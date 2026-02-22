import torch
import torch.nn as nn
from monai.networks.nets import EfficientNetBN
from monai.networks.layers.factories import Act


class EfficientNetBNwClf(nn.Module):
    def __init__(self, backbone_type, num_classes, pretrained=False, mri_modelpath=None, dropout_rate = 0.2, **kwargs):
        super(EfficientNetBNwClf, self).__init__()
        num_outputs = 1 if num_classes == 2 else num_classes

        self.backbone = EfficientNetBN(
            model_name = backbone_type,
            pretrained = False,
            spatial_dims=3,
            num_classes=num_outputs,
            in_channels=1,
        )        
        self.backbone = nn.Sequential(*(list(self.backbone.children())[:-4]))
        self.dropout = nn.Dropout(dropout_rate)
        self.clf = nn.Linear(1280, num_outputs, bias=True)
        self.swish = Act['memswish'](inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=1)

        if pretrained:
            self._load_pretrained_models(mri_modelpath=mri_modelpath)
            self.clf = nn.Linear(1280, num_outputs, bias=True)

    def _load_pretrained_models(self, mri_modelpath=None):
        if mri_modelpath is not None:
            mri_emb_weight = torch.load(mri_modelpath)
            for n, p in self.backbone.named_parameters():
                assert p.data.shape == mri_emb_weight['backbone.' + n].shape
                p.data = mri_emb_weight['backbone.' + n]
            # self.backbone.eval()


    def forward(self, x, demo_feature=None, get_inter=False):
        x = self.backbone(x)
        x = self.avg_pool(x)
        fea = x.flatten(start_dim=1)
        x = self.dropout(fea)
        y_hat = self.clf(x).squeeze()
        if get_inter:
            return fea, y_hat
        else:
            return y_hat

