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

from einops import rearrange

from networks.resnet_attention_generic import CategoricalFeatureTokenizer, NumericalFeatureTokenizer

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        inner_dim = dim_head * heads
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, query_in, key_in, value_in):
        q = self.to_q(query_in)
        k = self.to_k(key_in)
        v = self.to_v(value_in)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        out = out.transpose(1, 2).contiguous()
        out = out.view(out.shape[0], out.shape[1], -1)
        return self.to_out(out)
    


class ResNetTabAttention(nn.Module):
    def __init__(self, backbone, tabular_input, num_classes, clf_n_layers=1, n_attn_blocks=4, act='relu') -> None:
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
        
        self.attn = nn.ModuleList([Attention(dim=512, heads=8, dim_head=64, dropout=0.0) for _ in range(n_attn_blocks)])
        
        clf_layers = []
        for i in range(clf_n_layers):
            in_dim = 512
            out_dim = num_classes if i == clf_n_layers - 1 else 512
            clf_layers.append(nn.Linear(in_dim, out_dim))
            if i != clf_n_layers - 1:
                if act == 'relu':
                    clf_layers.append(nn.ReLU())
                elif act == 'silu':
                    clf_layers.append(nn.SiLU())
        self.clf = nn.Sequential(*clf_layers)
        
        self.numerical_feature_tokenizer = NumericalFeatureTokenizer(6, 512, True, 'uniform')
        self.gender_tokenizer = CategoricalFeatureTokenizer([2], 512, True, 'uniform')
        self.has_adas11_tokenizer = CategoricalFeatureTokenizer([2], 512, True, 'uniform')
        self.has_faq_tokenizer = CategoricalFeatureTokenizer([2], 512, True, 'uniform')
        
        
    def tokenize_tabular_input(self, demo_feature):
        if len(self.tabular_input) == 9:
            numerical_features = torch.stack([demo_feature[:, idx].unsqueeze(1) for idx in [0, 1, 2, 3, 5, 6]], dim=1).squeeze(-1)
            gender_feature = demo_feature[:, 4].unsqueeze(1).long()
            has_adas11_feature = demo_feature[:, 7].unsqueeze(1).long()
            has_faq_feature = demo_feature[:, 8].unsqueeze(1).long()
            
            numerical_features = self.numerical_feature_tokenizer(numerical_features)
            gender_feature = self.gender_tokenizer(gender_feature)
            has_adas11_feature = self.has_adas11_tokenizer(has_adas11_feature)
            has_faq_feature = self.has_faq_tokenizer(has_faq_feature)
        
            return torch.cat((numerical_features, gender_feature, has_adas11_feature, has_faq_feature), dim=1)
        
        elif len(self.tabular_input) == 7:
            # ['mmse', 'cdrsb', 'age', 'gender', 'educat', 'faq', 'has_faq']
            numerical_features = torch.stack([demo_feature[:, idx].unsqueeze(1) for idx in [0, 1, 2, 4, 5]], dim=1).squeeze(-1)
            gender_feature = demo_feature[:, 3].unsqueeze(1).long()
            has_faq_feature = demo_feature[:, 6].unsqueeze(1).long()
            
            numerical_features = self.numerical_feature_tokenizer(numerical_features)
            gender_feature = self.gender_tokenizer(gender_feature)
            has_faq_feature = self.has_faq_tokenizer(has_faq_feature)
            
            return torch.cat((numerical_features, gender_feature, has_faq_feature), dim=1)

        else:
            raise ValueError(f"Unsupported tabular input length: {len(self.tabular_input)}")
        
    def forward(self, x: torch.Tensor, demo_feature=None, get_inter=False) -> torch.Tensor:

        mri_feature = self.backbone(x, get_before_avg=True)
        assert len(self.tabular_input) == 9 or len(self.tabular_input) == 7, print(self.tabular_input)
        tab_feature = get_film_input(demo_feature, self.tabular_input)
        tab_feature = self.tokenize_tabular_input(tab_feature)
        input_feature = rearrange(mri_feature, 'b c h w d -> b (h w d) c')
        for attn in self.attn:
            input_feature = attn(input_feature, tab_feature, tab_feature)
            
        input_feature = input_feature.mean(dim=1)
        y_hat = self.clf(input_feature).squeeze()

        if get_inter:
            return input_feature, y_hat
        else:
            return y_hat
