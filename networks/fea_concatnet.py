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

import torch
import torch.nn as nn
from networks.mlp import MLP

class AdapterBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(AdapterBlock, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.adapter(x)

class Adapter(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer):
        super(Adapter, self).__init__()
   
        self.adapter_blocks = nn.ModuleList()
        for _ in range(n_layer):
            self.adapter_blocks.append(AdapterBlock(in_dim, out_dim))
            in_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        for adapter_block in self.adapter_blocks:
            out = adapter_block(x)
            x = out + x

        x = self.linear(x)
        return x
   

class FeatureConcatNet(nn.Module):
    def __init__(self, net, sub_net, clf_hidden_dim, clf_n_layer, decouple, adapter_dim, adapter_n_layer,
                 linear_probing, act='relu'):
        super(FeatureConcatNet, self).__init__()

        self.net = net
        self.sub_net = sub_net
        self.decouple = decouple
        self.adapter_dim = adapter_dim
        self.adapter_n_layer = adapter_n_layer
        self.linear_probing = linear_probing
        self.clf = self._make_clf_layer(hidden_dim=clf_hidden_dim, n_layer=clf_n_layer)
        self.act = act
        
        if adapter_n_layer > 0:
            self.adapter = Adapter(in_dim=512, out_dim=adapter_dim, n_layer=adapter_n_layer)
            
        for n, p in self.sub_net.named_parameters():
            p.requires_grad = False
        if linear_probing:
            for n, p in self.net.named_parameters():
                p.requires_grad = False

    def _make_clf_layer(self, hidden_dim, n_layer):
        if self.decouple:
            clf = nn.ModuleList()
            for i in range(2):
                clf.append(
                    MLP(feature_size=512 * 2, hidden_dim=hidden_dim, num_classes=1, num_layer=n_layer,
                        with_dropout=True, adv=False, act=self.act)
                )

        else:
            clf = MLP(feature_size=512 * 2, hidden_dim=hidden_dim, num_classes=1, num_layer=n_layer,
                           with_dropout=True, adv=False, act=self.act)
        return clf

    def forward(self, x, demo_feature=None, get_inter=False):
        with torch.no_grad():
            sub_feature, _ = self.sub_net(x, demo_feature, get_inter=True)

        if self.linear_probing:
            with torch.no_grad():
                feature, _ = self.net(x, demo_feature, get_inter=True)
        else:
            feature, _ = self.net(x, demo_feature, get_inter=True)

        if self.adapter_n_layer > 0:
            sub_feature = self.adapter(sub_feature)

        concat_feature = torch.cat((sub_feature, feature), dim=1)

        if self.decouple:
            _, age, _, _ = demo_feature
            age_mask = (age >= 75).cuda()
            tmp = 0
            for i in range(2):
                tmp += self.clf[i](concat_feature).squeeze() * (i == age_mask)
            y_hat = tmp.squeeze()
        else:
            y_hat = self.clf(concat_feature).squeeze()

        if get_inter:
            return concat_feature, y_hat
        else:
            return y_hat