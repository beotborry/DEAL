from __future__ import print_function

from trainers.mri_trainer import Trainer as MRITrainer
from networks.monai_resnet_tab_attention import ResNetTabAttention

class MRITabAttentionTrainer(MRITrainer):
    def __init__(self, args, log_dir=None, log_name=None, save_dir=None, **kwargs):
        super().__init__(args, log_dir, log_name, save_dir, **kwargs)
        num_outputs = 1 if self.num_classes == 2 else self.num_classes
        self.model = ResNetTabAttention(self.model, args.tabular_input, num_outputs, clf_n_layers=args.num_layer_clf,
                                        n_attn_blocks=args.n_attn_blocks, act=args.film_aux_net_act,)
        self.model.cuda()
        self.model.age_mean = self.train_dataset.age_mean
        self.model.age_std = self.train_dataset.age_std
        print(self.model.age_mean, self.model.age_std)