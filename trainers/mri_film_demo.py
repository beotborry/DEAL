from __future__ import print_function

from trainers.mri_trainer import Trainer as MRITrainer
from networks.monai_resnet_film import resnet18
from networks.filmed_net import FilmNet
import time
import pandas as pd
from utils import get_accuracy, get_auroc

from torch.cuda.amp import autocast
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix



class MRIFilmDemoTrainer(MRITrainer):
    def __init__(self, args, log_dir=None, log_name=None, save_dir=None, **kwargs):
        super().__init__(args, log_dir, log_name, save_dir, **kwargs)

    def _init_model(self, args):
        if args.sagittal:
            n_input_channels = 3
            spatial_dims = 2
        else:
            n_input_channels = 1
            spatial_dims = 3
        num_outputs = 1 if self.num_classes == 2 else self.num_classes
        if args.mri_arch in ['monai_resnet18']:
            model = resnet18(n_input_channels=n_input_channels, num_classes=num_outputs, spatial_dims=spatial_dims,
                                mri_modelpath=args.mri_modelpath, input_demo_for_film=args.tabular_input,
                                film_only_last=args.film_only_last, decouple=args.decouple,
                                age_mean=self.train_dataset.age_mean, age_std=self.train_dataset.age_std,
                                film_aux_net=args.film_aux_net, film_aux_net_act=args.film_aux_net_act, 
                                
                                )

        elif args.mri_arch in ["efficientnet", "vit_b"]:
            if args.mri_arch == "efficientnet":
                from networks.efficientnet import EfficientNetBNwClf
                feature_extractor = EfficientNetBNwClf(backbone_type=args.efficientnet_backbone,
                                                       num_classes=self.num_classes, pretrained=args.pretrained,
                                                       mri_modelpath=args.mri_modelpath, dropout_rate=args.drop_rate)

            elif args.mri_arch == "vit_b":
                from networks.vit_3d import ViT
                feature_extractor = ViT(image_size=192, image_patch_size=16, frames=192, frame_patch_size=16,
                            num_classes=num_outputs, dim=768, depth=12, heads=12, mlp_dim=768 * 4,
                            pool='cls', channels=1, dim_head=64,
                            dropout=args.tf_dropout, emb_dropout=args.emb_dropout,
                            freeze_embedding=args.freeze_embedding, modelpath=args.mri_modelpath)
            if args.cuda:
                feature_extractor = feature_extractor.cuda()
            model = FilmNet(feature_extractor, num_outputs, input_demo_for_film=args.tabular_input,
                            age_mean=self.train_dataset.age_mean, age_std=self.train_dataset.age_std,
                            decouple=args.decouple,
                            mri_arch=args.mri_arch,
                            share_ln=args.share_ln, film_aux_net=args.film_aux_net, dropout_rate=args.drop_rate)



        if args.cuda:
            model = model.cuda()
        return model

    def _train_epoch(self, model, epoch, epochs, train_loader, criterion, optimizer,
                     scaler=None, scheduler=None):
        model = self._set_mode(model, eval=False if self.args.set_mode == 'train' else True)
        running_acc = 0.0
        running_loss = 0.0
        per_g_loss = {g: 0. for g in range(self.num_groups)}
        per_g_num = {g: 0. for g in range(self.num_groups)}

        epoch_loss = 0.
        epoch_acc = 0.0
        num_data = 0
        batch_start_time = time.time()
        for i, data in enumerate(train_loader, 1):
            optimizer.zero_grad()
            inputs, labels, groups, ad_time, demo_feature, idx = self._process_data_for_train(data, self.cuda, self.binary)
            with autocast(enabled=True):
                outputs = model(inputs, demo_feature)
                if (not self.binary and outputs.dim() == 1) or len(outputs.shape) == 0:
                    outputs = outputs.unsqueeze(0)  # for only one data

                loss = criterion(outputs, labels)

            with torch.no_grad():
                for g in range(self.num_groups):
                    mask = groups == g
                    if mask.sum() != 0:
                        per_g_loss[g] += nn.BCEWithLogitsLoss(reduction='sum')(outputs[mask],
                                                                                labels[mask]).item()
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

        return epoch_loss.item() / num_data, epoch_acc / num_data, (per_g_loss), scaler

    def evaluate(self, model, loader, criterion, threshold=None, cuda=True, target='ad_transition',
                 df=None, set=set):
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
                    inputs, labels, groups, ad_time, demo_features, idx = self._process_data_for_train(eval_data, cuda,
                                                                                                       binary)
                    gs.extend(groups.cpu().numpy())

                    outputs = model(inputs, demo_features)
                    if (not self.binary and outputs.dim() == 1) or len(outputs.shape) == 0:
                        outputs = outputs.unsqueeze(0)  # for only one data
                    loss = criterion(outputs, labels)

                    data_dict = {'SubjectID': np.array(demo_features[0]),
                                 'label_time': ad_time.cpu().numpy(),
                                 'label_transition': labels.cpu().numpy(),
                                 'group': groups.cpu().numpy(),
                                 'age': demo_features[1].cpu().numpy()}


                    pred_dict = {'pred_transition': torch.sigmoid(outputs).cpu().numpy()}

                    data_dict.update(pred_dict)
                    if df is not None:
                        df = df.append(pd.DataFrame(data_dict))
                        df.reset_index(drop=True, inplace=True)


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
                    return (eval_loss.item(), per_g_loss), (
                    eval_acc.item(), bal_acc, acc_per_class), \
                        (roc_auc, fpr, tpr), (micro_f1, macro_f1), (
                    sensitivity, specificity, ppv, npv), confusion, threshold, \
                        (groupwise_acc, worst_acc, group_bal_acc, weighted_avg_acc, groupwise_conf), df
                except:
                    return (eval_loss, per_g_loss), (eval_acc, bal_acc, acc_per_class), \
                        (roc_auc, fpr, tpr), (micro_f1, macro_f1), (
                    sensitivity, specificity, ppv, npv), confusion, threshold, \
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
                          cuda=self.cuda, target=self.target, df=df, set=set)

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
        # if target == 'ad':
        print_state = f'[{epoch + 1}/{epochs}] ' \
                      f'Val Loss {results["val_loss"]:.3f} ' \
                      f'Test Loss: {results["test_loss"]:.3f}'


        if self.target != 'age':
            print_state += f' Val Acc: {results["val_acc"]:.3f} Test Acc: {results["test_acc"]:.3f}'
            print_state += f' Val F1 (micro/macro): {results["val_micro_F1"]:.3f}/{results["val_macro_F1"]:.3f} ' \
                           f' Test F1 (micro/macro): {results["test_micro_F1"]:.3f}/{results["test_macro_F1"]:.3f}'
            if self.binary:
                print_state += f' Val AUROC: {results["val_auroc"]:.3f}'
                print_state += f' Test AUROC: {results["test_auroc"]:.3f}'

        print(print_state + f' [{eval_time:.3f} s]')