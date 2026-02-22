import torch
import numpy as np
import random
import os
from os.path import join, expanduser
from scipy.io import loadmat
from typing import List
from sklearn import metrics
from sklearn.metrics import roc_curve
import pandas as pd

def list_dir(root: str, prefix: bool = False) -> List[str]:
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories


def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_id_from_mri_folder_name(folder_name):
    start_idx = folder_name.find('-') + 1
    end_idx = folder_name.find('X')
    sub_id = folder_name[start_idx:end_idx]
    return sub_id


def get_mri_id(root):
    subject_data = list_dir(root)
    mri_sub = []
    for sub in subject_data:
        mri_sub.append(get_id_from_mri_folder_name(sub))
    return set(mri_sub)

def get_accuracy(outputs, labels, binary=False, reduction='mean', threshold=None):
    if binary:
        outputs = torch.sigmoid(outputs)
        threshold = 0.5 if threshold is None else threshold
        predictions = (outputs >= threshold).float()
    else:
        predictions = torch.argmax(outputs, 1)
    c = (predictions == labels).float().squeeze()
    if reduction == 'none':
        return c
    else:
        accuracy = torch.mean(c)
        return accuracy.detach()


def get_auroc(outputs, labels, for_caat=False):
    try:
        y_trues = torch.cat(labels).long().cpu().detach()
        yhats = torch.cat(outputs).squeeze()
    except:
        y_trues = torch.stack(labels).long().cpu().detach()
        yhats = torch.stack(outputs).squeeze()

    y_preds = torch.sigmoid(yhats).cpu().detach()
    if torch.isnan(y_preds).any():
        print(yhats[torch.isnan(y_preds)])
    roc_auc = metrics.roc_auc_score(y_true=y_trues, y_score=y_preds)
    fpr, tpr, thresholds = roc_curve(y_trues, y_preds)
    return roc_auc, fpr, tpr


def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
        print("Failed to create directory!!")


def make_log_name(args):
    log_name = ''

    if args.mode == 'eval':
        log_name = args.modelpath.split('/')[-1]
        # remove .pt from name
        log_name = log_name[:-3]

    else:
        log_name += 'seed{}_tid{}_epochs{}_bs{}_lr{}_decay{}'.format(
            args.seed, args.test_set_id, args.epochs,
            args.batch_size, args.lr, args.weight_decay)

        if args.group_dro:
            log_name += '_gdro_lr{}'.format(args.group_weight_lr)
         
    return log_name


def save_model(state_dict, save_dir, log_name, is_best=False, fold=None, no_th=False):
    suffix = ''
    if fold is not None:
        suffix += '_fold{}'.format(fold)

    suffix += '_best' if is_best else '_last'

    if no_th:
        suffix += '_no_th'
    suffix += '.pt'

    model_savepath = os.path.join(save_dir, log_name + suffix)
    torch.save(state_dict, model_savepath)
    print('Model saved to %s' % model_savepath)


def load_model(model, save_dir, log_name, load_best=False, fold=None, no_th=False):
    suffix = ''
    if fold is not None:
        suffix += '_fold{}'.format(fold)
    if load_best:
        suffix += '_best'
        if no_th:
            suffix += '_no_th'
    suffix += '.pt'

    state_dict_path = os.path.join(save_dir, log_name + suffix)
    model.load_state_dict(torch.load(state_dict_path))
    return model

def update_dict_with_suffix(results_dict, suffix):
    new_key_dict = {}
    for key, value in results_dict.items():
        new_key_dict[key + suffix] = value
    return new_key_dict