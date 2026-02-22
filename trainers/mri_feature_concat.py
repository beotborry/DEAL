from __future__ import print_function

from trainers.mri_trainer import Trainer as MRITrainer
import torch
from networks.fea_concatnet import FeatureConcatNet
from copy import deepcopy


class MRIConcatTrainer(MRITrainer):
    def __init__(self, args, log_dir=None, log_name=None, save_dir=None, **kwargs):
        super().__init__(args, log_dir, log_name, save_dir, **kwargs)
        sub_net = self._get_pretrained_sub_net()
        self.model = FeatureConcatNet(self.model, sub_net, self.args.hidden_nodes_clf, self.args.num_layer_clf,
                                      self.args.decouple, self.args.adapter_dim, self.args.adapter_n_layer,
                                      self.args.linear_probing, self.args.film_aux_net_act)
        if self.args.cuda:
            self.model = self.model.cuda()

    def _get_pretrained_sub_net(self):
        assert self.args.sub_modelpath is not None
        args = deepcopy(self.args)
        args.decouple = False
        args.mri_modelpath = None
        sub_net = self._init_model(args)

        # load pretrained model
        sub_net_weight = torch.load(args.sub_modelpath)
        sub_net.load_state_dict(sub_net_weight)

        return sub_net

