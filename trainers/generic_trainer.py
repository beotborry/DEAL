from __future__ import print_function

import time
import os
import pandas as pd
from utils import get_accuracy, get_auroc
from collections import defaultdict
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.cuda.amp import GradScaler, autocast
from trainers.optim_utils import _get_optim_n_scheduler

import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from trainers.group_dro import CustomIndicesSampler, GroupWeightedLoss
from utils import get_accuracy, get_auroc, save_model, update_dict_with_suffix

from copy import deepcopy
import wandb


class GenericTrainer:
    def __init__(self, args, log_dir=None, log_name=None, save_dir=None, **kwargs):
        super().__init__()
        self.args = args
        self.last_eval_only = args.last_eval_only
        self.group_dro = args.group_dro
        self.group_weight_lr = args.group_weight_lr
        self.seed = args.seed
        self.decouple = args.decouple

        self.cuda = args.cuda
        self.target = args.target

        self.num_classes = 2
        if self.target == 'ad_time_cox':
            self.num_groups = 2
        else: self.num_groups = 4
        self.binary = True if self.num_classes == 2 else False

        self.mode = args.mode
        self.train_dataset, self.val_dataset, self.test_dataset = self._get_dataset(args)
        self.model = self._init_model(args)
        
        self.train_loader, self.val_loader, self.test_loader = self._get_dataloader(args,
                                                                                    self.train_dataset,
                                                                                    self.val_dataset,
                                                                                    self.test_dataset)

        self.logger = defaultdict(list)
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.log_name = log_name
        self.term = args.term
        
    def _get_dataset(self, args):
        raise NotImplementedError

    def _get_dataloader(self, args, train_dataset, val_dataset, test_dataset):
        def _init_fn(worker_id):
            np.random.seed(int(self.seed))

        bs = len(train_dataset) if args.batch_size == -1 else args.batch_size

        if self.group_dro:
            sampler = CustomIndicesSampler([], shuffle=True, seed=args.seed)
            train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler, shuffle=False,
                                      num_workers=args.num_workers, worker_init_fn=_init_fn,
                                      pin_memory=True, drop_last=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                      num_workers=args.num_workers, worker_init_fn=_init_fn,
                                      pin_memory=True, drop_last=True)

        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                                num_workers=args.num_workers, worker_init_fn=_init_fn, pin_memory=True)

        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False,
                                 num_workers=args.num_workers, worker_init_fn=_init_fn, pin_memory=True)
        return train_loader, val_loader, test_loader

    def _init_model(self, args):
        raise NotImplementedError

    def _set_mode(self, model, eval=False):
        if eval:
            model.eval()
        else:
            model.train()

        return model

    def _update_log(self, logger, results, is_last=False, is_best=False, fold=None, prefix=None, **kwargs):
        prefix = '' if prefix is None else prefix
        prefix += 'best_' if is_best else ''
        prefix += 'last_' if is_last else ''
        for key, value in results.items():
            logger[prefix + key].append(value)

        for key, value in kwargs.items():
            logger[prefix + key].append(value)

        if is_last and fold is None:
            suffix = "_log.pt"
            torch.save(logger, os.path.join(self.log_dir, self.log_name + suffix))

    def _get_loss(self, train_dataset, weighted=True):
        if self.group_dro:
            criterion = GroupWeightedLoss(criterion=torch.nn.BCEWithLogitsLoss(reduction='none'),
                                            num_groups=4,
                                            device=torch.device('cuda'),
                                            group_weight_lr=self.group_weight_lr,)
        else:
            if self.binary:
                if weighted:
                    print("In _get_loss fn...")
                    print(train_dataset.class_counts[0], train_dataset.class_counts[1])

                    n_data_per_label = torch.zeros(2)
                    n_data_per_label[0], n_data_per_label[1] = train_dataset.class_counts[0], \
                                                                train_dataset.class_counts[1]
                    pos_weight = n_data_per_label[0] / n_data_per_label[1]
                    self.pos_weight = pos_weight
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.cuda())
                else:
                    criterion = nn.BCEWithLogitsLoss()
            else:
                # weight is not supported currently, should be implemented when we consider multi-classes later.
                criterion = nn.CrossEntropyLoss(weight=None, reduction='none')
                class_counts = train_dataset.df["Dx_num"].value_counts()
                if self.args.mri_subg:
                    self.pos_weight = 1.
                else:
                    self.pos_weight = class_counts[0] / class_counts[1]

        time_criterion = nn.MSELoss()
        return criterion, time_criterion

    def train(self, epochs, model=None, train_loader=None, val_loader=None, test_loader=None, logger=None,
              fold=None, target='ad'):
        model = self.model if model is None else model

        train_loader = self.train_loader if train_loader is None else train_loader
        val_loader = self.val_loader if val_loader is None else val_loader
        test_loader = self.test_loader if test_loader is None else test_loader

        optimizer, scheduler = _get_optim_n_scheduler(self.args, model, num_iters=len(train_loader))
        criterion, time_criterion = self._get_loss(train_dataset=train_loader.dataset,
                                   weighted=not self.args.no_pos_weight)

        model = self._set_mode(model, eval=False if self.args.set_mode == 'train' else True)
        logger = self.logger if logger is None else logger
        best_loss = float('inf')

        scaler = GradScaler()

        best_group_bal_acc = -1
        best_worst_acc = -1
        results = {}
        best_loss_results, best_worst_acc_results, best_group_bal_acc_results = None, None, None

        for epoch in range(epochs):
            train_loss, train_acc, train_per_g_loss, scaler = self._train_epoch(
                model, epoch, epochs, train_loader, criterion, optimizer, scaler,
                scheduler
            )

            results['train_loss'], results['train_acc'] = train_loss, train_acc
            for g in range(self.num_groups):
                results[f'train_g{g}_loss'] = train_per_g_loss[g]

            if not self.last_eval_only:
                results, eval_time, threshold = self.get_results(model, val_loader, results, set='val')
                results, eval_time, threshold = self.get_results(model, test_loader, results, set='test')

                self._print_state(results, eval_time, epoch, epochs)
                self._update_log(logger, results)

                if results['val_group_bal_acc'] > best_group_bal_acc:
                    best_group_bal_acc = results['val_group_bal_acc']
                    best_group_bal_acc_results = deepcopy(results)
                    best_group_bal_acc_model = deepcopy(model.state_dict())

                # for Group DRO early stopping
                if results['val_worst_acc'] > best_worst_acc:
                    best_worst_acc = results['val_worst_acc']
                    best_worst_acc_results = deepcopy(results)
                    best_worst_acc_model = deepcopy(model.state_dict())

                self._update_log(logger, results, train_loss=train_loss, train_acc=train_acc)

            results['lr'] = scheduler.get_last_lr()[0]
            wandb.log(results)

            if scheduler is not None and 'cosine' in self.args.scheduler:
                scheduler.step()


        if not self.last_eval_only:
            best_group_bal_acc_results = update_dict_with_suffix(best_group_bal_acc_results, suffix='_at_best_gba')
            best_worst_acc_results = update_dict_with_suffix(best_worst_acc_results, suffix='_at_best_wa')
            wandb.run.summary.update(best_group_bal_acc_results)
            wandb.run.summary.update(best_worst_acc_results)

        if self.args.save_model:
            save_model(model.state_dict(), self.save_dir, self.log_name, fold=fold, no_th=self.args.no_best_th)

        train_loader.dataset.transform = test_loader.dataset.transform
        results, eval_time, threshold = self.get_results(model, train_loader, results, set='train', is_last=True)
        results, eval_time, threshold = self.get_results(model, val_loader, results, set='val', is_last=True)
        results, eval_time, threshold = self.get_results(model, test_loader, results, set='test', is_last=True)
        self._update_log(logger, results, is_last=True, fold=fold)
        wandb.run.summary.update(results)

        print('Training Finished!')
        return model, logger

    def cross_entropy_loss(self, input: torch.Tensor,
                           target: torch.Tensor
                           ) -> torch.Tensor:

        return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()

    def _train_epoch(self, model, epoch, epochs, train_loader, criterion, optimizer,
                     scaler=None, scheduler=None):
        model = self._set_mode(model, eval=False if self.args.set_mode == 'train' else True)

        running_acc = 0.0
        running_loss = 0.0

        per_g_loss = {g: 0. for g in range(self.num_groups)}
        per_g_num = {g: 0. for g in range(self.num_groups)}

        epoch_loss = 0.0
        epoch_acc = 0.0
        num_data = 0
        batch_start_time = time.time()

        if self.group_dro:
            train_loader.sampler.indices = random.choices(
                population=train_loader.dataset.base_indices,
                weights=train_loader.dataset.sampling_weights,
                k=len(train_loader.dataset)
            )
        for i, data in enumerate(train_loader, 1):
            optimizer.zero_grad()

            # Get the inputs
            inputs, labels, groups, ad_time, demo_feature, idx = self._process_data_for_train(data, self.cuda, self.binary)
            with autocast(enabled=True):
                outputs = model(inputs, demo_feature)
                if (not self.binary and outputs.dim() == 1) or len(outputs.shape) == 0:
                    outputs = outputs.unsqueeze(0)  # for only one data
                if self.group_dro:
                    loss = criterion(outputs, labels, groups)
                else:
                    loss = criterion(outputs, labels)

            with torch.no_grad():
                for g in range(self.num_groups):
                    mask = groups == g
                    if mask.sum() != 0:
                        per_g_loss[g] += nn.BCEWithLogitsLoss(reduction='sum')(outputs[mask], labels[mask]).item()
                        per_g_num[g] += mask.sum().item()

            running_loss += loss.detach()


            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None and self.args.scheduler == 'cycle':
                scheduler.step()
            num_data += len(labels)
            epoch_loss += loss.detach() * len(labels)
            acc = get_accuracy(outputs, labels, binary=self.binary)
            running_acc += acc
            epoch_acc += acc * len(labels)

            if i % self.term == 0:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                statement = f"[{epoch + 1}/{epochs}, {i:5d}] Train Loss: {running_loss / self.term:.3f}"
                statement += f" Train Acc: {running_acc / self.term:.3f}"
                statement += f" [{avg_batch_time / self.term:.3f} s/batch]"
                print(statement)

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()

        for g in range(self.num_groups):
            per_g_loss[g] /= per_g_num[g]

        return epoch_loss.item() / num_data, epoch_acc / num_data, per_g_loss, scaler

    def _process_data_for_train(self, data, cuda=True, binary=True):

        # type conversion & to cuda
        inputs, labels, groups, ad_time, demo_features, idx = data
        inputs = inputs.float()
        if cuda:
            inputs = inputs.cuda(non_blocking=True)

        labels = labels.long() if not binary else labels.float()
        groups = groups.long()
        ad_time = ad_time.float()
        if cuda:
            labels = labels.cuda(non_blocking=True)
            groups = groups.cuda(non_blocking=True)
            ad_time = ad_time.cuda(non_blocking=True)

        return inputs, labels, groups, ad_time, demo_features, idx

    def evaluate(self, model, loader, criterion, threshold=None, cuda=True, target='ad_transition', df=None):
        model = self._set_mode(model, eval=True)
        num_classes = loader.dataset.num_classes
        num_groups = loader.dataset.num_groups
        train_group_bincount = loader.dataset.train_group_bincount

        train_group_weight = train_group_bincount / train_group_bincount.sum()
        binary = num_classes == 2
        eval_acc = 0.0
        eval_loss = 0
        eval_data_count = 0
        bal_acc = 0.
        acc_per_class = np.zeros(num_classes)
        num_per_class = np.zeros(num_classes)

        roc_auc, fpr, tpr = None, None, None
        micro_f1, macro_f1 = 0., 0.
        sensitivity, specificity = 0., 0.
        ppv, npv = 0., 0.
        confusion = np.zeros((num_classes, num_classes))
        groupwise_acc = np.zeros(num_groups)
        groupwise_conf = np.zeros(num_groups)

        worst_acc, group_bal_acc, weighted_avg_acc = 0., 0., 0.
        per_g_loss = {g: 0. for g in range(self.num_groups)}
        per_g_num = {g: 0. for g in range(self.num_groups)}
        with autocast(enabled=True):
            with torch.no_grad():
                yhats = []
                ys = []
                gs = []
                for j, eval_data in enumerate(loader):
                    # Get the inputs
                    inputs, labels, groups, ad_time, demo_features, idx = self._process_data_for_train(eval_data, cuda, binary)
                    gs.extend(groups.cpu().numpy())
                    outputs = model(inputs, demo_features)
                    data_dict = {'SubjectID': np.array(demo_features[0]),
                               'label_time': ad_time.cpu().numpy(),
                               'label_transition': labels.cpu().numpy(),
                               'group': groups.cpu().numpy(),
                               'age': demo_features[1].cpu().numpy()}


                    if (not self.binary and outputs.dim() == 1) or len(outputs.shape) == 0:
                        outputs = outputs.unsqueeze(0)  # for only one data

                    pred_dict = {'pred_transition': torch.sigmoid(outputs).cpu().numpy()}

                    data_dict.update(pred_dict)
                    if df is not None:
                        df = df.append(pd.DataFrame(data_dict))
                        df.reset_index(drop=True, inplace=True)
                    loss = criterion(outputs, labels)
                    ys.append(labels)

                    eval_loss += loss
                    yhats.append(outputs)
                    eval_data_count += len(labels)

                    hits = get_accuracy(outputs, labels, reduction="none", binary=binary, threshold=threshold)
                    eval_acc += hits.sum()

                    if hits.dim() == 0:
                        hits = hits.unsqueeze(0)

                    for c in range(num_classes):
                        acc_per_class[c] += hits[(labels == c)].sum().data.cpu().numpy()
                        num_per_class[c] += (labels == c).sum().data.cpu().numpy()

                    for g in range(self.num_groups):
                        mask = groups == g
                        if mask.any():
                            per_g_loss[g] += criterion(outputs[mask], labels[mask])
                            per_g_num[g] += mask.sum().item()

                if binary:
                    roc_auc, fpr, tpr = get_auroc(yhats, ys)
                    threshold = 0.5 if threshold is None else threshold
                    ys = torch.cat(ys).long().cpu().tolist()
                    yhats = torch.cat(yhats).squeeze()
                    conf = torch.sigmoid(yhats).cpu()
                    yhats = (conf >= threshold).long().tolist()

                    confusion = confusion_matrix(ys, yhats, labels=[i for i in range(num_classes)])
                    eval_acc = accuracy_score(ys, yhats)
                    micro_f1 = f1_score(ys, yhats, average='binary')
                    macro_f1 = f1_score(ys, yhats, average='macro')
                    eps = 1e-7
                    sensitivity = confusion[1, 1] / (confusion[1, :].sum() if confusion[1, :].sum() != 0
                                                        else confusion[1, :].sum() + eps)
                    specificity = confusion[0, 0] / (confusion[0, :].sum() if confusion[0, :].sum() != 0
                                                        else confusion[0, :].sum() + eps)
                    ppv = confusion[1, 1] / (confusion[:, 1].sum() if confusion[:, 1].sum() != 0
                                                else confusion[:, 1].sum() + eps)
                    npv = confusion[0, 0] / (confusion[:, 0].sum() if confusion[:, 0].sum() != 0
                                                else confusion[:, 0].sum() + eps)

                    bal_acc = (sensitivity + specificity) / 2
                    acc_per_class[0] = specificity
                    acc_per_class[1] = sensitivity

                    is_correct = np.array(ys) == np.array(yhats)
                    gs = np.array(gs).astype(np.int32)
                    for g in range(num_groups):
                        groupwise_acc[g] += len(np.where(np.logical_and(is_correct == 1, gs == g))[0])
                        groupwise_acc[g] /= len(np.where(gs == g)[0])
                        groupwise_conf[g] = conf[gs == g].sum().numpy() / (gs == g).sum()

                    worst_acc = np.min(groupwise_acc)
                    group_bal_acc = np.mean(groupwise_acc)
                    weighted_avg_acc = np.sum(train_group_weight * groupwise_acc)

                eval_loss = eval_loss / eval_data_count
                for g in range(self.num_groups):
                    per_g_loss[g] /= per_g_num[g]

                try:
                    return (eval_loss.item(), per_g_loss), (eval_acc.item(), bal_acc, acc_per_class), \
                        (roc_auc, fpr, tpr), (micro_f1, macro_f1), (sensitivity, specificity, ppv, npv), confusion, threshold, \
                        (groupwise_acc, worst_acc, group_bal_acc, weighted_avg_acc, groupwise_conf), df
                except:
                    return (eval_loss, per_g_loss), (eval_acc, bal_acc, acc_per_class), \
                        (roc_auc, fpr, tpr), (micro_f1, macro_f1), (sensitivity, specificity, ppv, npv), confusion, threshold, \
                        (groupwise_acc, worst_acc, group_bal_acc, weighted_avg_acc, groupwise_conf), df

    def get_results(self, model, loader, results=None, criterion=nn.BCEWithLogitsLoss(reduction='sum'),
                    threshold=None,
                    is_best=False, is_last=False, set='test'):

        suffix = '_best' if is_best else ''
        suffix += '_last' if is_last else ''
        if threshold is None:
            threshold = 0.5
        else:
            threshold = threshold
            suffix += '_optimal'

        if is_last and not is_best:
            df = pd.DataFrame(columns=['SubjectID', 'label_time', 'label_transition', 'pred_time', 'pred_transition', 'group', 'age'])
            df_save_path = os.path.join(self.log_dir, self.log_name + f"_{set}.csv")
        else:
            df = None

        results = {} if results is None else results
        eval_start_time = time.time()

        (results[f'{set}_loss' + suffix], per_g_loss), accs, roc_scores, \
        f1, metrics, confusion, threshold, group_metrics, df = \
            self.evaluate(model, loader, criterion, threshold=threshold,
                          cuda=self.cuda, target=self.target, df=df)

        results[f'{set}_acc' + suffix], results[f'{set}_bal_acc' + suffix], results[f'{set}_acc_per_class' + suffix] = accs
        results[f'{set}_auroc' + suffix], results[f'{set}_fpr' + suffix], results[f'{set}_tpr' + suffix] = roc_scores
        results[f'{set}_micro_F1' + suffix], results[f'{set}_macro_F1' + suffix] = f1
        (results[f'{set}_sensitivity' + suffix], results[f'{set}_specificity' + suffix],
         results[f'{set}_ppv' + suffix], results[f'{set}_npv' + suffix]) = metrics
        results[f'{set}_threshold' + suffix] = threshold
        results[f'{set}_confusion' + suffix] = confusion

        for g in range(loader.dataset.num_groups):
            results[f'{set}_group{g}_acc' + suffix] = group_metrics[0][g]
            results[f'{set}_group{g}_loss' + suffix] = per_g_loss[g]
            results[f'{set}_group{g}_conf' + suffix] = group_metrics[-1][g]

        (results[f'{set}_worst_acc' + suffix], results[f'{set}_group_bal_acc' + suffix],
         results[f'{set}_weighted_avg_acc' + suffix]) = group_metrics[1:-1]

        eval_end_time = time.time()

        if df is not None:
            df.to_csv(df_save_path)

        return results, eval_end_time - eval_start_time, threshold

    def _print_state(self, results, eval_time, epoch, epochs):
        print_state = f'[{epoch + 1}/{epochs}] ' \
                      f'Val Loss {results["val_loss"]:.3f} Test Loss: {results["test_loss"]:.3f}'

        print_state += f' Val Acc: {results["val_acc"]:.3f} Test Acc: {results["test_acc"]:.3f}'
        print_state += f' Val F1 (micro/macro): {results["val_micro_F1"]:.3f}/{results["val_macro_F1"]:.3f} ' \
                        f' Test F1 (micro/macro): {results["test_micro_F1"]:.3f}/{results["test_macro_F1"]:.3f}'

        if self.binary:
            print_state += f' Val AUROC: {results["val_auroc"]:.3f}'
            print_state += f' Test AUROC: {results["test_auroc"]:.3f}'

        print(print_state + f' [{eval_time:.3f} s]')
