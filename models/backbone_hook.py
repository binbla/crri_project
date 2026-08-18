from mmdet.models.backbones.resnet import ResNet
from mmdet.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmengine.dist import all_reduce, get_dist_info
from mmengine import dist as mm_dist
import numpy as np
from torch.distributed import all_reduce, is_initialized, ReduceOp


@MODELS.register_module()
class CRRI(BaseModule):

    def __init__(self, in_channels=256, bank_size=1024, momentum=0.01, eps=1e-5, init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.bank_size = bank_size
        self.momentum = momentum
        self.eps = eps

        # [1, C, bank_size] buffer
        self.register_buffer('style_bank', torch.zeros(1, in_channels, bank_size))
        self.register_buffer('initialized', torch.tensor(False))

    def _update_bank(self, x_sorted):
        if x_sorted.numel() == 0:
            return

        x_resampled = F.interpolate(x_sorted,
                                    size=self.bank_size,
                                    mode='linear',
                                    align_corners=True)

        local_mean_dist = x_resampled.mean(dim=0, keepdim=True)

        if torch.distributed.is_initialized():
            from mmengine.dist import all_reduce
            local_mean_dist = local_mean_dist.float().contiguous()
            all_reduce(local_mean_dist, op='mean')

        local_mean_dist = local_mean_dist.to(self.style_bank.dtype)

        if not self.initialized:
            self.style_bank.data.copy_(local_mean_dist)
            self.initialized.fill_(True)
        else:
            new_bank = (1 - self.momentum) * self.style_bank + \
                       self.momentum * local_mean_dist
            self.style_bank.data.copy_(new_bank)

    def forward(self, x, batch_data_samples=None):
        """
        Args:
            x: [B, C, H, W]
        """
        if x.numel() == 0:
            return x, None

        B, C, H, W = x.shape
        x_view = x.view(B, C, -1)

        if x_view.isnan().any():
            x_view = torch.nan_to_num(x_view, nan=0.0)

        if self.training or self.initialized:
            noise = (torch.rand_like(x_view) * 2 - 1) * self.eps
            x_for_sort = x_view + noise
        else:
            x_for_sort = x_view

        x_sorted, x_idx = torch.sort(x_for_sort, dim=-1)

        if self.training:
            with torch.no_grad():
                self._update_bank(x_sorted)

        if not self.initialized:
            return x, None

        N = H * W
        style_vals = F.interpolate(self.style_bank, size=N, mode='linear',
                                   align_corners=True).to(x.dtype)

        style_vals = style_vals.expand(B, -1, -1)

        out = torch.empty_like(x_view)
        out.scatter_(dim=-1, index=x_idx, src=style_vals)

        return out.view(B, C, H, W), None

