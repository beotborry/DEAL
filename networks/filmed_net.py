import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.monai_resnet_film import get_film_input
from copy import deepcopy

class FiLM(nn.Module):
    """
    A Feature-wise Linear Modulation Layer from
     'FiLM: Visual Reasoning with a General Conditioning Layer'
    """
    def forward(self, x, gammas, betas):
        return (gammas * x) + betas


class FilmNet(nn.Module):
    def __init__(self, feature_extractor, num_outputs, input_demo_for_film='emd', age_mean=None, age_std=None,
                 decouple=False, mri_arch='monai_resnet18', share_ln=False, film_aux_net='linear', dropout_rate=0.0):
        super(FilmNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.age_mean = age_mean
        self.age_std = age_std
        self.input_demo_for_film = input_demo_for_film
        film_input_dim = len(input_demo_for_film)
        self.share_ln = share_ln
        self.mri_arch = mri_arch
        self.feature_extractor.eval()
        if self.feature_extractor.__class__.__name__ == 'ViT':
            dummy_input = torch.randn((1, 1, 192, 192, 192)).cuda()
        else:
            dummy_input = torch.randn((1, 1, 182, 218, 182)).cuda()
        with torch.no_grad():
            fea, _ = self.feature_extractor(dummy_input, get_inter=True)
        feature_dim = fea.shape[-1]
        self.feature_extractor.train()

        if 'daft' in input_demo_for_film:
            film_input_dim = 15
        else:
            if 'adas11' in input_demo_for_film:
                input_demo_for_film.extend(['has_adas11'])               
            film_input_dim = len(input_demo_for_film)
            
        if film_aux_net =='linear':
            self.film_generator = nn.Linear(film_input_dim, 2 * feature_dim)
        else:
            self.film_generator = nn.Sequential(
                nn.Linear(film_input_dim, int(feature_dim/4)),
                nn.ReLU(),
                nn.Linear(int(feature_dim/4), 2 * feature_dim)
            )

        self.film = FiLM()
        if mri_arch == 'efficientnet':
            self.dropout = nn.Dropout(dropout_rate)
        self.decouple = decouple
        if decouple:
            self.clf = nn.ModuleList()
            for _ in range(2):
                if 'vit' in mri_arch:
                    if share_ln:
                        self.layer_norm = nn.LayerNorm(feature_dim)
                        self.clf.append(nn.Linear(feature_dim, num_outputs))
                    else:
                        self.clf.append(nn.Sequential(
                            nn.LayerNorm(feature_dim),
                            nn.Linear(feature_dim, num_outputs)
                        ))
                else:
                    self.clf.append(nn.Linear(feature_dim, num_outputs))
        else:
            if 'vit' in mri_arch:
                self.clf = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, num_outputs)
                )
            else:
                self.clf = nn.Linear(feature_dim, num_outputs)

    def forward(self, x, demo_feature=None, get_inter=False):

        feature, _ = self.feature_extractor(x, demo_feature, get_inter=True)
        if self.share_ln:
            feature = self.layer_norm(feature)

        film_input = get_film_input(demo_feature, self.input_demo_for_film, self.age_mean, self.age_std)

        film_vector = self.film_generator(film_input).view(feature.size(0), feature.size(1), 2)
        beta = film_vector[:, :, 0]
        gamma = film_vector[:, :, 1]
        feature = self.film(feature, gamma, beta)
        if self.mri_arch == 'efficientnet':
            feature = self.dropout(feature)

        if self.decouple:
            age = demo_feature[1]
            if self.age_mean is not None:
                age = age * self.age_std + self.age_mean
            age_mask = (age >= 75).cuda()
            tmp = 0
            for i in range(2):
                tmp += self.clf[i](feature).squeeze() * (i == age_mask)
            y_hat = tmp.squeeze()
        else:
            y_hat = self.clf(feature).squeeze()


        if get_inter:
            return feature, y_hat

        else:
            return y_hat