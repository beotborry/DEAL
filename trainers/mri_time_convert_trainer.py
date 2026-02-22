from __future__ import print_function
import torch
import wandb
import numpy as np
import pandas as pd
import time

from torch.cuda.amp import GradScaler, autocast
from trainers.mri_trainer import Trainer as MRITrainer
from trainers.optim_utils import _get_optim_n_scheduler
from utils import save_model
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.linear_model.coxph import BreslowEstimator
from networks.monai_resnet_film import get_film_input
from copy import deepcopy

class MRITimeConvertTrainer(MRITrainer):
    def __init__(self, args, log_dir=None, log_name=None, save_dir=None, **kwargs):
        super().__init__(args, log_dir, log_name, save_dir, **kwargs)
        self.args = args
        self.age_mean = self.train_dataset.age_mean
        self.age_std = self.train_dataset.age_std

    def _init_model(self, args):
        if args.method == 'tab_concat':        
            from networks.monai_resnet_tab_concat import ResNetConcat
            self.model = super()._init_model(args)
            self.model = ResNetConcat(self.model, args.tabular_input, 1, clf_n_layers=args.num_layer_clf,
                                    act=args.film_aux_net_act,)
        elif args.method == 'tab_concat_lin_proj':
            from networks.monai_resnet_tab_concat_lin_proj import ResNetConcatLinProj
            self.model = super()._init_model(args)
            self.model = ResNetConcatLinProj(self.model, args.tabular_input, 1, clf_n_layers=args.num_layer_clf,
                                    act=args.film_aux_net_act,)

            self.model.age_mean = self.train_dataset.age_mean
            self.model.age_std = self.train_dataset.age_std

        elif args.method == 'tab_attention':
            from networks.monai_resnet_tab_attention import ResNetTabAttention
            self.model = super()._init_model(args)
            self.model = ResNetTabAttention(self.model, args.tabular_input, 1, clf_n_layers=args.num_layer_clf,
                                n_attn_blocks=args.n_attn_blocks, act=args.film_aux_net_act,)

            self.model.age_mean = self.train_dataset.age_mean
            self.model.age_std = self.train_dataset.age_std
            
        elif args.method == 'film_demo':
            if args.sagittal:
                n_input_channels = 3
                spatial_dims = 2
            else:
                n_input_channels = 1
                spatial_dims = 3
            num_outputs = 1 if self.num_classes == 2 else self.num_classes
            from networks.monai_resnet_film import resnet18
            self.model = resnet18(n_input_channels=n_input_channels, num_classes=num_outputs, spatial_dims=spatial_dims,
                    mri_modelpath=args.mri_modelpath, input_demo_for_film=args.tabular_input,
                    film_only_last=args.film_only_last, decouple=args.decouple,
                    age_mean=self.train_dataset.age_mean, age_std=self.train_dataset.age_std,
                    film_aux_net=args.film_aux_net, film_aux_net_act=args.film_aux_net_act,)

        elif args.method == 'tabular_only':
            from networks.mlp import MLP
            self.tabular_input = args.tabular_input
            if 'adas11' in self.tabular_input:
                self.tabular_input.append('has_adas11')
            if 'apoe4' in self.tabular_input:
                self.tabular_input.append('has_apoe4')
            if 'faq' in self.tabular_input:
                self.tabular_input.append('has_faq')
            self.tabular_input = deepcopy(self.tabular_input) 


            self.model = MLP(feature_size=len(self.tabular_input), hidden_dim=args.hidden_nodes, num_classes=2,
            num_layer=args.num_layer, with_dropout=not args.no_dropout_mlp)
        
        elif args.method == 'daft':
            from networks.monai_resnet_daft import resnet18
            if args.sagittal:
                n_input_channels = 3
                spatial_dims = 2
            else:
                n_input_channels = 1
                spatial_dims = 3
            num_outputs = 1 if self.num_classes == 2 else self.num_classes

            age_mean = self.train_dataset.age_mean
            age_std = self.train_dataset.age_std
            self.model = resnet18(n_input_channels=n_input_channels, num_classes=num_outputs,
                            spatial_dims=spatial_dims, daft_input=args.daft_input, tabular_input=args.tabular_input,
                            age_mean=age_mean, age_std=age_std, modelpath=args.modelpath)
        else:
            self.model = super()._init_model(args)

        if args.cuda:
            self.model = self.model.cuda()

        return self.model  
      

    def train(self, epochs, model=None, train_loader=None, val_loader=None, test_loader=None, logger=None,
                fold=None, target='ad_time'):
            model = self.model if model is None else model

            train_loader = self.train_loader if train_loader is None else train_loader
            val_loader = self.val_loader if val_loader is None else val_loader
            test_loader = self.test_loader if test_loader is None else test_loader

            optimizer, scheduler = _get_optim_n_scheduler(self.args, model, num_iters=len(train_loader))
            
            model = self._set_mode(model, eval=False)
            logger = self.logger if logger is None else logger
            scaler = GradScaler()

            results = {}
            for epoch in range(epochs):
                train_loss, train_per_g_loss, scaler = self._train_epoch(
                    model, epoch, epochs, train_loader, optimizer, scaler,
                    scheduler
                )

                if not self.last_eval_only:
                    results, eval_time, threshold = self.get_results(model, val_loader, results, set='val')
                    results, eval_time, threshold = self.get_results(model, test_loader, results, set='test')

                    self._print_state(results, eval_time, epoch, epochs)
                    self._update_log(logger, results, train_loss=train_loss)

                if scheduler is not None and 'cosine' in self.args.scheduler:
                    scheduler.step()

                results['train_loss'] = train_loss
                for g in range(self.num_groups):
                    results[f'train_g{g}_loss'] = train_per_g_loss[g]
                
                results['lr'] = scheduler.get_last_lr()[0]
                wandb.log(results)

            if self.args.save_model:
                save_model(model.state_dict(), self.save_dir, self.log_name, fold=fold, no_th=self.args.no_best_th)
                
            self.df_for_G_est_val = val_loader.dataset.df # use only val set for G_est
            self.df_for_G_est_val.reset_index(drop=True, inplace=True)
            self.df_for_G_est_val = self.df_for_G_est_val[['ad_time']]
            self.df_for_G_est_val.loc[:, 'event'] = self.df_for_G_est_val['ad_time'] > 0
            self.df_for_G_est_val.loc[:, 'time'] = np.abs(self.df_for_G_est_val['ad_time'])
            self.df_for_G_est_val.drop(columns=['ad_time'], inplace=True)
            self.df_for_G_est_val = self.df_for_G_est_val.to_records(index=False)

            self.df_for_G_est_test = test_loader.dataset.df # use only test set for G_est
            self.df_for_G_est_test.reset_index(drop=True, inplace=True)
            self.df_for_G_est_test = self.df_for_G_est_test[['ad_time']]
            self.df_for_G_est_test.loc[:, 'event'] = self.df_for_G_est_test['ad_time'] > 0
            self.df_for_G_est_test.loc[:, 'time'] = np.abs(self.df_for_G_est_test['ad_time'])
            self.df_for_G_est_test.drop(columns=['ad_time'], inplace=True)
            self.df_for_G_est_test = self.df_for_G_est_test.to_records(index=False)
            
            results, eval_time, threshold = self.get_results(model, train_loader, results, set='train', is_last=True)
            results, eval_time, threshold = self.get_results(model, val_loader, results, set='val', is_last=True)
            results, eval_time, threshold = self.get_results(model, test_loader, results, set='test', is_last=True)
            wandb.run.summary.update(results)

            print('Training Finished!')
            return model, logger

    def _train_epoch(self, model, epoch, epochs, train_loader, optimizer,
                    scaler=None, scheduler=None):
        model = self._set_mode(model, eval=False)
        running_loss = 0.0

        per_g_loss = {g: 0. for g in range(self.num_groups)}
        per_g_num = {g: 0. for g in range(self.num_groups)}

        epoch_loss = 0.0
        num_data = 0
        batch_start_time = time.time()

        for i, data in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)

            if self.args.method in ['tab_concat', 'daft', 'scratch', 'film_demo', 'tab_concat_lin_proj', 'tab_attention']:
                inputs, labels, groups, ad_time, demo_feature, idx = self._process_data_for_train(data, self.cuda,
                                                                                                self.binary)
                with autocast(enabled=True):
                    outputs = model(inputs, demo_feature)

                    if (not self.binary and outputs.dim() == 1) or len(outputs.shape) == 0:
                        outputs = outputs.unsqueeze(0)  
                    exp_outputs = torch.exp(outputs)
                    
                    loss = -((outputs - torch.log(((torch.abs(ad_time[:, None]) <= torch.abs(ad_time)) * exp_outputs).sum(dim=-1))) * (ad_time > 0)).mean()

            elif self.args.method == 'tabular_only':

                _, labels, groups, ad_time, demo_features, idx = self._process_data_for_train(data, self.cuda,
                                                                                                self.binary)
                    
                inputs = get_film_input(demo_features, input_demo_for_film=self.tabular_input,
                                        age_mean=self.age_mean, age_std=self.age_std)
                inputs = inputs.cuda()
                
                with autocast(enabled=True):
                    outputs = model(inputs)

                    if (not self.binary and outputs.dim() == 1) or len(outputs.shape) == 0:
                        outputs = outputs.unsqueeze(0)  
                    exp_outputs = torch.exp(outputs)
                    
                    loss = -((outputs - torch.log(((torch.abs(ad_time[:, None]) <= torch.abs(ad_time)) * exp_outputs).sum(dim=-1))) * (ad_time > 0)).mean()

            with torch.no_grad():
                for g in range(self.num_groups):
                    mask = groups == g
                    if mask.sum() != 0:
                        per_g_num[g] += mask.sum().item()
                        per_g_loss[g] += -(((outputs - torch.log(((torch.abs(ad_time[:, None]) <= torch.abs(ad_time)) * exp_outputs).sum(dim=-1))) * (ad_time > 0))[mask]).sum()

            running_loss += loss.detach()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and self.args.scheduler == 'cycle':
                scheduler.step()


            num_data += len(labels)
            epoch_loss += loss.detach() * len(labels)

            if i % self.term == 0:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                statement = f"[{epoch + 1}/{epochs}, {i:5d}] Train Loss: {running_loss / self.term:.3f} "
                statement += f" [{avg_batch_time / self.term:.3f} s/batch]"
                print(statement)
                running_loss = 0.0
                batch_start_time = time.time()

        for g in range(self.num_groups):
            per_g_loss[g] /= per_g_num[g]

        return epoch_loss.item() / num_data, per_g_loss, scaler

    def evaluate(self, model, loader, cuda=True, df=None, is_last = False):
        model = self._set_mode(model, eval=True)

        eval_loss = 0
        eval_data_count = 0

        per_g_loss = {g: 0. for g in range(self.num_groups)}
        per_g_num = {g: 0. for g in range(self.num_groups)}
        per_g_c_index = {g: 0. for g in range(self.num_groups)}



        with torch.no_grad():
            ys = []
            gs = []
            estimates = []
            ad_times = []
            
            if loader.drop_last:
                # reinitialize the loader
                loader = torch.utils.data.DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers, pin_memory=True,  drop_last=False)
            for j, eval_data in enumerate(loader):
                # Get the inputs
                if self.args.method in ['tab_concat', 'daft', 'scratch', 'film_demo', 'tab_concat_lin_proj', 'tab_attention']:
                    inputs, labels, groups, ad_time, demo_features, idx = self._process_data_for_train(eval_data, cuda,
                                                                                                    True)
                    gs.extend(groups.cpu().numpy())
                    ad_times.extend(ad_time.cpu().numpy())
                    outputs = model(inputs, demo_features)
                elif self.args.method == 'tabular_only':
                    _, labels, groups, ad_time, demo_features, idx = self._process_data_for_train(eval_data, cuda,
                                                                                                    True)
                    inputs = get_film_input(demo_features, input_demo_for_film=self.tabular_input,
                                            age_mean=self.age_mean, age_std=self.age_std)
                    inputs = inputs.cuda()
                    gs.extend(groups.cpu().numpy())
                    ad_times.extend(ad_time.cpu().numpy())
                    outputs = model(inputs)

                data_dict = {'SubjectID': np.array(demo_features[0]),
                            'label_time': ad_time.cpu().numpy(),
                            'label_transition': labels.cpu().numpy(),
                            'group': groups.cpu().numpy(),
                            'age': demo_features[1].cpu().numpy()}

                if (not self.binary and outputs.dim() == 1) or len(outputs.shape) == 0:
                    outputs = outputs.unsqueeze(0)  # for only one data

                estimates.extend(outputs.cpu().numpy())
                exp_outputs = torch.exp(outputs)
                loss = -((outputs - torch.log(((torch.abs(ad_time[:, None]) <= torch.abs(ad_time)) * exp_outputs).sum(dim=-1))) * (ad_time > 0)).mean()



                pred_dict = {'pred_time': outputs.cpu().numpy()}
                data_dict.update(pred_dict)
                if df is not None:
                    df = df.append(pd.DataFrame(data_dict))
                    df.reset_index(drop=True, inplace=True)

                ys.append(labels.cpu())

                eval_loss += loss
                eval_data_count += len(labels)

                for g in range(self.num_groups):
                    mask = groups == g
                    if mask.sum() != 0:
                        per_g_num[g] += mask.sum().item()
                        per_g_loss[g] += -(((outputs - torch.log(((torch.abs(ad_time[:, None]) <= torch.abs(ad_time)) * exp_outputs).sum(dim=-1))) * (ad_time > 0))[mask]).sum().cpu().item()
                        
            eval_loss = eval_loss / eval_data_count
            
            c_index = concordance_index_censored((np.array(ad_times)>0),np.abs(np.array(ad_times)),np.array(estimates))[0]
            for g in range(self.num_groups):
                per_g_loss[g] /= per_g_num[g]
                mask = torch.tensor(gs) == g
                per_g_c_index[g] = concordance_index_censored((np.array(ad_times)>0)[mask],np.abs(np.array(ad_times))[mask],np.array(estimates)[mask])[0]

        if is_last:
            try:
                return (eval_loss.item(), c_index, per_g_loss, per_g_c_index, estimates, ad_times, gs), df
            except:
                return (eval_loss, c_index, per_g_loss, per_g_c_index, estimates, ad_times, gs), df
        else:
            try:
                return (eval_loss.item(), c_index, per_g_loss, per_g_c_index), df
            except:
                return (eval_loss, c_index, per_g_loss, per_g_c_index), df
            
    def get_results(self, model, loader, results=None, threshold=None,
                    is_best=False, is_last=False, set='test'):

        suffix = '_best' if is_best else ''
        suffix += '_last' if is_last else ''
        if threshold is None:
            threshold = 0.5
        else:
            threshold = threshold
            suffix += '_optimal'

        df = None

        results = {} if results is None else results
        eval_start_time = time.time()

        if is_last:
            (results[f'{set}_loss' + suffix], results[f'{set}_c_index' + suffix], per_g_loss, per_g_c_index, estimates, ad_times, gs), df = \
                self.evaluate(model, loader, cuda=self.cuda, df=df, is_last=is_last)
        else:
            (results[f'{set}_loss' + suffix], results[f'{set}_c_index' + suffix], per_g_loss, per_g_c_index), df = \
                self.evaluate(model, loader, cuda=self.cuda, df=df, is_last=is_last)
                
        if is_last and set == 'train':
            self._baseline_model = BreslowEstimator().fit(np.array(estimates), np.array(ad_times) > 0, np.abs(ad_times))
            self.brier_eval_times = np.arange(30*6, 30*(36+1), 30)

        surv_info = pd.DataFrame({'event': np.array(ad_times) > 0, 'time': np.abs(np.array(ad_times))})
        surv_funcs = self._baseline_model.get_survival_function(np.array(estimates))
        surv_prob = np.vstack([fn(self.brier_eval_times) for fn in surv_funcs])

        if is_last and set == 'test':
            results[f'{set}_ibs' + suffix] = integrated_brier_score(self.df_for_G_est_test,
                                                                        surv_info[['event', 'time']].to_records(index=False),
                                                                        surv_prob, self.brier_eval_times)
        elif is_last and set == 'val':
            results[f'{set}_ibs' + suffix] = integrated_brier_score(self.df_for_G_est_val,
                                                                        surv_info[['event', 'time']].to_records(index=False),
                                                                        surv_prob, self.brier_eval_times)
                                                                    
        for g in range(loader.dataset.num_groups):
            results[f'{set}_group{g}_loss' + suffix] = per_g_loss[g]
            results[f'{set}_group{g}_c_index' + suffix] = per_g_c_index[g]
            if is_last and (set == 'val' or set == 'test'):
                mask = np.array(gs) == g
                surv_info = pd.DataFrame({'event': np.array(ad_times)[mask] > 0, 'time': np.abs(np.array(ad_times)[mask])})
                surv_funcs = self._baseline_model.get_survival_function(np.array(estimates)[mask])  
                surv_prob = np.vstack([fn(self.brier_eval_times) for fn in surv_funcs])

                if set == 'test':
                    results[f'{set}_group{g}_ibs' + suffix] = integrated_brier_score(self.df_for_G_est_test, 
                                                surv_info[['event', 'time']].to_records(index=False), surv_prob, self.brier_eval_times)
                elif set == 'val':
                    results[f'{set}_group{g}_ibs' + suffix] = integrated_brier_score(self.df_for_G_est_val, 
                                                surv_info[['event', 'time']].to_records(index=False), surv_prob, self.brier_eval_times)
            
        results[f'{set}_group_bal_c_index' + suffix] = np.mean([per_g_c_index[g] for g in range(loader.dataset.num_groups)])
        results[f'{set}_group_bal_loss' + suffix] = np.mean([per_g_loss[g] for g in range(loader.dataset.num_groups)])

        if is_last and (set == 'val' or set == 'test'):
            results[f'{set}_group_bal_ibs' + suffix] = np.mean([results[f'{set}_group{g}_ibs' + suffix] for g in range(loader.dataset.num_groups)])
            
        
        eval_end_time = time.time()

        return results, eval_end_time - eval_start_time, threshold

    def _print_state(self, results, eval_time, epoch, epochs):
        print_state = f'[{epoch + 1}/{epochs}] ' \
                    f'Val Loss {results["val_loss"]:.3f} Test Loss: {results["test_loss"]:.3f}'

        print(print_state + f' [{eval_time:.3f} s]')