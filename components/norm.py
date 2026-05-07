import torch
import torch.nn as nn
from mmdet.registry import MODELS


@MODELS.register_module('IBN')
class IBN(nn.Module):

    def __init__(self, num_features, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(num_features * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(num_features - self.half)

    def forward(self, x):
        # x: [B, C, H, W]
        split = torch.split(x, [self.half, x.shape[1] - self.half], dim=1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        return torch.cat((out1, out2), 1)
