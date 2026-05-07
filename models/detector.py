import copy
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import os
import sys
import torch.nn.functional as F
import math

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.models import TwoStageDetector, FasterRCNN, BaseDetector, SemiBaseDetector, DINO
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from mmengine.model import ExponentialMovingAverage


@MODELS.register_module()
class HookFasterRCNN(FasterRCNN):

    def __init__(self,
                 res0_adv_cfg=None,
                 res1_adv_cfg=None,
                 res2_adv_cfg=None,
                 res3_adv_cfg=None,
                 res4_adv_cfg=None,
                 vis_debug_cfg=None,
                 **kwargs):
        super().__init__(**kwargs)
        if res0_adv_cfg is not None:
            self.res0_adv = MODELS.build(res0_adv_cfg)
        else:
            self.res0_adv = None

        if res1_adv_cfg is not None:
            self.res1_adv = MODELS.build(res1_adv_cfg)
        else:
            self.res1_adv = None
        if res2_adv_cfg is not None:
            self.res2_adv = MODELS.build(res2_adv_cfg)
        else:
            self.res2_adv = None
        if res3_adv_cfg is not None:
            self.res3_adv = MODELS.build(res3_adv_cfg)
        else:
            self.res3_adv = None
        if res4_adv_cfg is not None:
            self.res4_adv = MODELS.build(res4_adv_cfg)
        else:
            self.res4_adv = None

        # ==============================

    def extract_feat(self, batch_inputs, batch_data_samples=None):
        """
        
        Returns:
            tuple: (outs, style_losses)
        """

        losses = {}
        x_0 = self.backbone.extract_stage0(batch_inputs)
        if self.res0_adv is not None:
            x_0, res0_loss = self.res0_adv(x_0, batch_data_samples)
            if self.training and res0_loss is not None:
                losses['res0_loss'] = res0_loss
        # outs = self.backbone.forward_from_stage0(x_0)

        # x_1 = self.backbone.extract_stage1(batch_inputs)

        x_1 = self.backbone.extract_stage1(x_0)
        # ori = x_1  #
        if self.res1_adv is not None:
            x_1, res1_loss = self.res1_adv(x_1, batch_data_samples)
            if self.training and res1_loss is not None:
                losses['res1_loss'] = res1_loss

        x_2 = self.backbone.extract_stage2(x_1)
        if self.res2_adv is not None:
            x_2, res2_loss = self.res2_adv(x_2, batch_data_samples)
            if self.training and res2_loss is not None:
                losses['res2_loss'] = res2_loss

        x_3 = self.backbone.extract_stage3(x_2)
        if self.res3_adv is not None:
            x_3, res3_loss = self.res3_adv(x_3, batch_data_samples)
            if self.training and res3_loss is not None:
                losses['res3_loss'] = res3_loss

        x_4 = self.backbone.extract_stage4(x_3)
        if self.res4_adv is not None:
            x_4, res4_loss = self.res4_adv(x_4, batch_data_samples)
            if self.training and res4_loss is not None:
                losses['res4_loss'] = res4_loss

        outs = self.backbone.forward_from_stage4(x_1, x_2, x_3, x_4)

        # self.save_pt(batch_data_samples, [x1_ori, x_1, x_2, x_3, x_4]) # for vis
        # self.save_pt(batch_data_samples, [ori, x_1]) # for vis

        if self.with_neck:
            outs = self.neck(outs)
        # return outs, losses
        return outs

    def extract_feat_wo_neck(self, batch_inputs, batch_data_samples=None):
        """
        
        Returns:
            tuple: (outs, style_losses)
        """

        losses = {}

        x_1 = self.backbone.extract_stage1(batch_inputs)
        # x1_ori = x_1
        if self.res1_adv is not None:
            x_1, res1_loss = self.res1_adv(x_1, batch_data_samples)
            if self.training and res1_loss is not None:
                losses['res1_loss'] = res1_loss

        x_2 = self.backbone.extract_stage2(x_1)
        if self.res2_adv is not None:
            x_2, res2_loss = self.res2_adv(x_2, batch_data_samples)
            if self.training and res2_loss is not None:
                losses['res2_loss'] = res2_loss

        x_3 = self.backbone.extract_stage3(x_2)
        if self.res3_adv is not None:
            x_3, res3_loss = self.res3_adv(x_3, batch_data_samples)
            if self.training and res3_loss is not None:
                losses['res3_loss'] = res3_loss

        x_4 = self.backbone.extract_stage4(x_3)
        if self.res4_adv is not None:
            x_4, res4_loss = self.res4_adv(x_4, batch_data_samples)
            if self.training and res4_loss is not None:
                losses['res4_loss'] = res4_loss

        outs = self.backbone.forward_from_stage4(x_1, x_2, x_3, x_4)

        # self.save_pt(batch_data_samples, [x1_ori, x_1, x_2, x_3, x_4])

        return outs, losses

    def loss(self, batch_inputs, batch_data_samples):
        losses = {}
        # x, hook_losses = self.extract_feat(batch_inputs, batch_data_samples)
        x = self.extract_feat(batch_inputs, batch_data_samples)
        # losses = hook_losses

        rpn_roi_losses = self.loss_from_feat(x, batch_data_samples)
        losses.update(rpn_roi_losses)

        return losses

    def predict(self, batch_inputs, batch_data_samples, rescale=True):

        assert self.with_bbox, 'Bbox head must be implemented.'
        # x, _ = self.extract_feat(batch_inputs, batch_data_samples)
        x = self.extract_feat(batch_inputs, batch_data_samples)

        results_list = self.predict_from_feat(x, batch_data_samples, rescale=rescale)
        return results_list

    def loss_from_feat(self, x, batch_data_samples):
        losses = {}
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)
            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(x,
                                                                          rpn_data_samples,
                                                                          proposal_cfg=proposal_cfg)
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]

        roi_losses = self.roi_head.loss(x, rpn_results_list, batch_data_samples)
        losses.update(roi_losses)

        return losses

    def predict_from_feat(self, x, batch_data_samples, rescale=True):

        assert self.with_bbox, 'Bbox head must be implemented.'

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [data_sample.proposals for data_sample in batch_data_samples]

        results_list = self.roi_head.predict(x,
                                             rpn_results_list,
                                             batch_data_samples,
                                             rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return batch_data_samples

    def save_pt(
            self,
            batch_data_samples,
            feats,
            save_dir="/tmp/feats/"):
        os.makedirs(save_dir, exist_ok=True)

        for i, data_sample in enumerate(batch_data_samples):

            img_path = data_sample.metainfo['img_path']
            name = os.path.splitext(os.path.basename(img_path))[0]

            for stage_id, feat in enumerate(feats, 0):

                feat_i = feat[i].detach().cpu()

                save_path = os.path.join(save_dir, f"{name}_stage{stage_id}.pt")

                torch.save(feat_i, save_path)
