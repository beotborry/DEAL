from __future__ import print_function

from trainers.mri_trainer import Trainer as MRITrainer
import torch
from networks.filmed_net import FilmNet
from copy import deepcopy

class MRIFilmTrainer(MRITrainer):
    def __init__(self, args, log_dir=None, log_name=None, save_dir=None, **kwargs):
        super().__init__(args, log_dir, log_name, save_dir, **kwargs)
        age_net = None
        ad_time_net = None
        age_mean, age_std = None, None
        if self.args.film_input != 'gender':
            age_net = self._get_pretrained_net(self.args.age_modelpath)
            age_mean, age_std = self._get_age_statistics()
            if self.args.ad_time_modelpath is not None:
                ad_time_net = self._get_pretrained_net(self.args.ad_time_modelpath)

        self.model = FilmNet(self.model, age_net, 1, age_mean, age_std, self.args.film_input, self.decouple,
                             ad_time_net, self.args.use_ad_time_feature, self.args.linear_probing)
        if self.args.cuda:
            self.model = self.model.cuda()

    def _get_pretrained_net(self, modelpath):
        args = deepcopy(self.args)
        args.decouple = False
        pretrained_model = self._init_model(args)
        # load pretrained model
        pretrained_weight = torch.load(modelpath)
        pretrained_model.load_state_dict(pretrained_weight)
        for n, p in pretrained_model.named_parameters():
            p.requires_grad = False
        pretrained_model.eval()
        return pretrained_model

    def _get_age_statistics(self):
        args = deepcopy(self.args)
        args.target = 'age'
        args.normalize = True
        args.seed = 0
        args.test_set_id = 0
        age_train_dataset, _, _ = self._get_dataset(args)
        age_mean = age_train_dataset.age_mean
        age_std = age_train_dataset.age_std
        return age_mean, age_std