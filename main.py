import torch
import numpy as np

from utils import check_log_dir, make_log_name, set_seed
from argument import get_args
import os
from importlib import import_module
from trainers.mri_feature_concat import MRIConcatTrainer
from trainers.mri_film import MRIFilmTrainer
from trainers.mri_film_demo import MRIFilmDemoTrainer
from trainers.mri_daft import MRIDaftTrainer
from trainers.mri_tabular_concat import MRITabConcatTrainer
from trainers.mri_tabular_concat_lin_proj import MRITabConcatLinProjTrainer
from trainers.tabular_only import TabularOnlyTrainer
from trainers.mri_time_convert_trainer import MRITimeConvertTrainer
from trainers.mri_tabular_attention import MRITabAttentionTrainer
import os
import wandb

wandb.login()

args = get_args()


def main():
    torch.backends.cudnn.enabled = True

    seed = args.seed
    set_seed(args.test_set_id)

    np.set_printoptions(precision=4)
    torch.set_printoptions(precision=4)
    wandb.init(
        project='DEAL',
        name=args.date,
    )
    wandb.config.update(args)

    log_name = make_log_name(args)
    target = args.target
    target += f'_year3'
    save_dir = os.path.join("./trained_models", args.date, 'mri', target)
    log_dir = os.path.join("./results", args.date, 'mri', target)
    check_log_dir(save_dir)
    check_log_dir(log_dir)

    if args.target == 'ad_time_cox':
        trainer = MRITimeConvertTrainer(args, log_dir, log_name, save_dir)
    else:
        if args.method == 'feature_concat':
            trainer = MRIConcatTrainer(args, log_dir, log_name, save_dir)
        elif args.method == 'film':
            trainer = MRIFilmTrainer(args, log_dir, log_name, save_dir)
        elif args.method == 'film_demo':
            trainer = MRIFilmDemoTrainer(args, log_dir, log_name, save_dir)
        elif args.method == 'daft':
            trainer = MRIDaftTrainer(args, log_dir, log_name, save_dir)
        elif args.method == 'tab_concat':
            trainer = MRITabConcatTrainer(args, log_dir, log_name, save_dir)
        elif args.method == 'tab_concat_lin_proj':
            trainer = MRITabConcatLinProjTrainer(args, log_dir, log_name, save_dir)
        elif args.method == 'tab_attention':
            trainer = MRITabAttentionTrainer(args, log_dir, log_name, save_dir)
        elif args.method =='tabular_only':
            trainer = TabularOnlyTrainer(args, log_dir, log_name, save_dir)
        else:
            Modality = import_module(f'trainers.mri_trainer')
            trainer = Modality.Trainer(args, log_dir, log_name, save_dir)

    wandb.config.update(args, allow_val_change=True)
    trainer.train(args.epochs, target=args.target)
    wandb.finish()
    print('Done.')


if __name__ == "__main__":
    main()
