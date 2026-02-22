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

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_pool_layer
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option
from networks.sequential import Sequential
from copy import deepcopy
from collections import OrderedDict


__all__ = [
    "ResNet",
    "ResNetBlock",
    "ResNetBottleneck",
    "resnet10",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet200",
]


def get_inplanes():
    return [64, 128, 256, 512]


def get_avgpool():
    return [0, 1, (1, 1), (1, 1, 1)]


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: nn.Module | partial | None = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels.
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for first conv layer.
            downsample: which downsample layer to use.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = norm_type(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.downsample = downsample
        self.stride = stride
        self.planes = planes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DaftBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: nn.Module | partial | None = None,
        daft_input='original', tabular_input=None,
        age_mean=None, age_std=None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels.
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for first conv layer.
            downsample: which downsample layer to use.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]


        self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = norm_type(planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.relu = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.downsample = downsample
        self.stride = stride
        self.planes = planes

        self.daft_input = daft_input
        self.tabular_input = tabular_input
        if daft_input == 'original':
            self.tabular_size = 15
        else:
            self.tabular_size = len(self.tabular_input)
        
        bottleneck_factor = 7
        n_basefilters = get_inplanes()[0]
        bdim = int((4 * n_basefilters + self.tabular_size) / bottleneck_factor)
        self.bottleneck_dim = bdim
        aux_input_dims = in_planes
        # shift and scale decoding
        self.split_size = in_planes
        self.scale = None
        self.shift = None
        self.film_dims = 2 * in_planes
        # create aux net
        layers = [
            ("aux_base", nn.Linear(self.tabular_size + aux_input_dims, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, self.film_dims, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))
        self.age_mean = age_mean
        self.age_std = age_std

    def rescale_features(self, feature_map, demo_feature):

        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        x_aux = self._get_aux_input(demo_feature)
        squeeze = torch.cat((squeeze, x_aux), dim=1)
        attention = self.aux(squeeze)
        v_scale, v_shift = torch.split(attention, self.split_size, dim=1)
        v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
        v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        return (v_scale * feature_map) + v_shift

    def _get_aux_input(self, demo_feature):
        (sub_id, age, gender, educat, cdrsb, adas11,
        mmse, apoe4, abeta42, tau, ptau, av45, fdg, group, faq) = demo_feature
        

        if self.daft_input == 'original':
            has_apoe4 = torch.ones_like(apoe4)
            has_apoe4[torch.isnan(apoe4)] = 0
            apoe4[torch.isnan(apoe4)] = 0

            has_abeta42 = torch.ones_like(abeta42)
            has_abeta42[torch.isnan(abeta42)] = 0
            abeta42[torch.isnan(abeta42)] = 0

            has_tau = torch.ones_like(tau)
            has_tau[torch.isnan(tau)] = 0
            tau[torch.isnan(tau)] = 0

            has_ptau = torch.ones_like(ptau)
            has_ptau[torch.isnan(ptau)] = 0
            ptau[torch.isnan(ptau)] = 0

            has_av45 = torch.ones_like(av45)
            has_av45[torch.isnan(av45)] = 0
            av45[torch.isnan(av45)] = 0

            has_fdg = torch.ones_like(fdg)
            has_fdg[torch.isnan(fdg)] = 0
            fdg[torch.isnan(fdg)] = 0

            x_aux = torch.stack((age, gender, educat, apoe4, has_apoe4, abeta42, has_abeta42,
                                 tau, has_tau, ptau, has_ptau, av45, has_av45, fdg, has_fdg), dim=1)
        else:
            demo_feature = {k: v for k, v in zip(['sub_id', 'age', 'gender', 'educat', 'cdrsb', 'adas11',
                                                'mmse', 'apoe4', 'abeta42', 'tau', 'ptau', 'av45', 'fdg', 'group', 'faq'], demo_feature)}


            if 'age_dummy' in self.tabular_input:
                assert self.age_mean is not None and self.age_std is not None
                denormalized_age = age * self.age_std + self.age_mean
                age_dummy = (denormalized_age >= 75).float()
                demo_feature['age_dummy'] = age_dummy

            if 'adas11' in self.tabular_input:
                has_adas11 = torch.ones_like(adas11)
                has_adas11[torch.isnan(adas11)] = 0
                adas11[torch.isnan(adas11)] = 0
                demo_feature.update({'has_adas11': has_adas11, 'adas11': adas11})

            if 'apoe4' in self.tabular_input:
                has_apoe4 = torch.ones_like(apoe4)
                has_apoe4[torch.isnan(apoe4)] = 0
                apoe4[torch.isnan(apoe4)] = 0
                demo_feature.update({'has_apoe4': has_apoe4, 'apoe4': apoe4})
            
            if 'faq' in self.tabular_input:
                has_faq = torch.ones_like(faq)
                has_faq[torch.isnan(faq)] = 0
                faq[torch.isnan(faq)] = 0
                demo_feature.update({'has_faq': has_faq, 'faq': faq})
            
            x_aux = torch.stack([demo_feature[k] for k in self.tabular_input], dim=1)
        x_aux = x_aux.cuda()
        return x_aux

    def forward(self, x: torch.Tensor, demo_feature) -> torch.Tensor:
        residual = x

        x = self.rescale_features(x, demo_feature)
        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        downsample: nn.Module | partial | None = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels (taking expansion into account).
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for second conv layer.
            downsample: which downsample layer to use.
        """

        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_type(planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_type(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet based on: `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? <https://arxiv.org/pdf/1711.09577.pdf>`_.
    Adapted from `<https://github.com/kenshohara/3D-ResNets-PyTorch/tree/master/models>`_.

    Args:
        block: which ResNet block to use, either Basic or Bottleneck.
            ResNet block class or str.
            for Basic: ResNetBlock or 'basic'
            for Bottleneck: ResNetBottleneck or 'bottleneck'
        layers: how many layers to use.
        block_inplanes: determine the size of planes at each step. Also tunable with widen_factor.
        spatial_dims: number of spatial dimensions of the input image.
        n_input_channels: number of input channels for first convolutional layer.
        conv1_t_size: size of first convolution layer, determines kernel and padding.
        conv1_t_stride: stride of first convolution layer.
        no_max_pool: bool argument to determine if to use maxpool layer.
        shortcut_type: which downsample block to use. Options are 'A', 'B', default to 'B'.
            - 'A': using `self._downsample_basic_block`.
            - 'B': kernel_size 1 conv + norm.
        widen_factor: widen output for each layer.
        num_classes: number of output (classifications).
        feed_forward: whether to add the FC layer for the output, default to `True`.
        bias_downsample: whether to use bias term in the downsampling block when `shortcut_type` is 'B', default to `True`.

    """

    def __init__(
        self,
        block: type[ResNetBlock | ResNetBottleneck] | str,
        layers: list[int],
        block_inplanes: list[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: tuple[int] | int = 7,
        conv1_t_stride: tuple[int] | int = 2,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 400,
        feed_forward: bool = True,
        bias_downsample: bool = True,  # for backwards compatibility (also see PR #5477)
        daft_input='original',
        tabular_input = None,
        age_mean=None,
        age_std=None,
        modelpath=None,
    ) -> None:
        super().__init__()

        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)

        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        norm_type: type[nn.BatchNorm1d | nn.BatchNorm2d | nn.BatchNorm3d] = Norm[Norm.BATCH, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        avgp_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        block_avgpool = get_avgpool()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.bias_downsample = bias_downsample
        self.tabular_input = tabular_input
        if daft_input != 'original':
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
        
        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,  # type: ignore
            stride=conv1_stride,  # type: ignore
            padding=tuple(k // 2 for k in conv1_kernel_size),  # type: ignore
            bias=False,
        )
        self.bn1 = norm_type(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = pool_type(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2)
        self.layer4 = self._make_layer(DaftBlock, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2,
                                       daft_input=daft_input, tabular_input=self.tabular_input, age_mean=age_mean, age_std=age_std)
        self.avgpool = avgp_type(block_avgpool[spatial_dims])
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes) if feed_forward else None

        for n, m in self.named_modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, norm_type):
                if m.weight is not None:
                    nn.init.constant_(torch.as_tensor(m.weight), 1)
                    nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(torch.as_tensor(m.bias), 0)

        if modelpath is not None:
            self.load_state_dict(torch.load(modelpath), strict=False)
            self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes) if feed_forward else None
            print("load model from", modelpath)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(
        self,
        block: type[ResNetBlock | ResNetBottleneck | DaftBlock],
        planes: int,
        blocks: int,
        spatial_dims: int,
        shortcut_type: str,
        stride: int = 1,
        daft_input = None,
        tabular_input = None,
        age_mean = None,
        age_std = None
    ) -> Sequential:
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        downsample: nn.Module | partial | None = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(
                        self.in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=self.bias_downsample,
                    ),
                    norm_type(planes * block.expansion),
                )
        if daft_input is not None:
            layers = [
                block(
                    in_planes=self.in_planes, planes=planes, spatial_dims=spatial_dims,
                    stride=stride, downsample=downsample, daft_input=daft_input, tabular_input=tabular_input, age_mean=age_mean, age_std=age_std
                )
            ]
        else:
            layers = [
                block(
                    in_planes=self.in_planes, planes=planes, spatial_dims=spatial_dims,
                    stride=stride, downsample=downsample,
                )
            ]

        self.in_planes = planes * block.expansion
        for _i in range(1, blocks):
            if daft_input is not None:
                layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims, daft_input=daft_input, tabular_input=tabular_input,
                                    age_mean=age_mean, age_std=age_std))
            else:
                layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims))
        if daft_input is not None:
            return Sequential(*layers)
        else:
            return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, demo_feature=None, get_inter=False) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x, demo_feature)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y_hat = self.fc(x).squeeze()

        if get_inter:
            return x, y_hat
        else:
            return y_hat

def _resnet(
    arch: str,
    block: type[ResNetBlock | ResNetBottleneck],
    layers: list[int],
    block_inplanes: list[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model: ResNet = ResNet(block, layers, block_inplanes, **kwargs)
    if pretrained:
        # Author of paper zipped the state_dict on googledrive,
        # so would need to download, unzip and read (2.8gb file for a ~150mb state dict).
        # Would like to load dict from url but need somewhere to save the state dicts.
        raise NotImplementedError(
            "Currently not implemented. You need to manually download weights provided by the paper's author"
            " and load then to the model with `state_dict`. See https://github.com/Tencent/MedicalNet"
            "Please ensure you pass the appropriate `shortcut_type` and `bias_downsample` args. as specified"
            "here: https://github.com/Tencent/MedicalNet/tree/18c8bb6cd564eb1b964bffef1f4c2283f1ae6e7b#update20190730"
        )
    return model


def resnet10(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-10 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet10", ResNetBlock, [1, 1, 1, 1], get_inplanes(), pretrained, progress, **kwargs)


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", ResNetBlock, [2, 2, 2, 2], get_inplanes(), pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-34 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", ResNetBlock, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-50 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", ResNetBottleneck, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 8 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", ResNetBottleneck, [3, 4, 23, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 8 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", ResNetBottleneck, [3, 8, 36, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet200(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-200 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 8 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet200", ResNetBottleneck, [3, 24, 36, 3], get_inplanes(), pretrained, progress, **kwargs)