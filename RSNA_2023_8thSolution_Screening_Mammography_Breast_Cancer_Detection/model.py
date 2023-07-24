import torch
import torch.nn as nn
import torch.nn.functional as F

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

from torch.nn.parameter import Parameter
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (self.__class__.__name__  + f"(p={self.p.data.tolist()[0]:.4f},eps={self.eps})")

import sys
sys.path.append("/home/br/workspace/RSNA2023/input/pytorch-image-models-main/")
from timm import create_model, list_models
from torch.nn import Module, Linear, Sequential, ModuleList, ReLU, Dropout, Flatten

class BreastCancerModel(Module):
    def __init__(self, model_arch, dropout=0.0, fc_dropout=0.0):
        super().__init__()
        self.model = create_model(
            model_arch, 
            pretrained=True,
            num_classes=0, 
            drop_rate=dropout,
            global_pool="", 
            # in_chans=1, # for 1 channel images
            )
        self.num_feats = self.model.num_features

        self.cancer_logits = Linear(self.num_feats, 1)
        
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.global_pool = GeM(p_trainable=True)

    def forward(self, x):
        x = self.model(x) # (bs, num_feats) /  (bs, num_feats, 16, 16)
        x = self.global_pool(x) # (bs, num_feats, 1, 1)
        x = x[:,:,0,0] # # (bs, num_feats)
        cancer_logits = self.cancer_logits(self.fc_dropout(x)).squeeze() # (bs)
        return cancer_logits