from __future__ import print_function

from trainers.mri_trainer import Trainer as MRITrainer
from networks.monai_resnet_daft import resnet18

class MRIDaftTrainer(MRITrainer):
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

        age_mean = self.train_dataset.age_mean
        age_std = self.train_dataset.age_std
        model = resnet18(n_input_channels=n_input_channels, num_classes=num_outputs,
                         spatial_dims=spatial_dims, daft_input=args.daft_input, tabular_input=args.tabular_input,
                         age_mean=age_mean, age_std=age_std, modelpath=args.modelpath)
        print(age_mean, age_std)

        if args.cuda:
            model = model.cuda()
        return model