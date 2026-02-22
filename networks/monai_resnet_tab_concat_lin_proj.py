# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn
from networks.monai_resnet_film import get_film_input


from copy import deepcopy

class ResNetConcatLinProj(nn.Module):
    def __init__(self, backbone, tabular_input, num_classes, clf_n_layers=1, act='relu') -> None:
        super().__init__()
        self.backbone = backbone
        self.tabular_input = tabular_input
        if 'adas11' in self.tabular_input:
            self.tabular_input.append('has_adas11')
        if 'moca' in self.tabular_input:
            self.tabular_input.append('has_moca')
        if 'moca_from_mmse' in self.tabular_input:
            self.tabular_input.append('has_moca_from_mmse')
        if 'adcoms' in self.tabular_input:
            self.tabular_input.append('has_adcoms')
        if 'apoe4' in self.tabular_input:
            self.tabular_input.append('has_apoe4')
        if 'faq' in self.tabular_input:
            self.tabular_input.append('has_faq')
        self.tabular_input = deepcopy(self.tabular_input) 
        
        
        tabular_dim = len(self.tabular_input)
        self.tab_proj = nn.Linear(tabular_dim, 512)

        clf_layers = []
        for i in range(clf_n_layers):
            in_dim = 512 + 512 if i == 0 else 512
            out_dim = num_classes if i == clf_n_layers - 1 else 512
            clf_layers.append(nn.Linear(in_dim, out_dim))
            if i != clf_n_layers - 1:
                if act == 'relu':
                    clf_layers.append(nn.ReLU())
                elif act == 'silu':
                    clf_layers.append(nn.SiLU())
        self.clf = nn.Sequential(*clf_layers)

    def forward(self, x: torch.Tensor, demo_feature=None, get_inter=False) -> torch.Tensor:

        mri_feature, _ = self.backbone(x, get_inter=True)
        assert len(self.tabular_input) == 9 or len(self.tabular_input) == 7, print(self.tabular_input)
        tab_feature = get_film_input(demo_feature, self.tabular_input)
        tab_feature = self.tab_proj(tab_feature)
        input_feature = torch.cat((mri_feature, tab_feature), dim=1)
        y_hat = self.clf(input_feature).squeeze()

        if get_inter:
            return input_feature, y_hat
        else:
            return y_hat
