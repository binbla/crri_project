# Copyright (c) OpenMMLab. All rights reserved.
import copy
import itertools
import logging
from typing import Dict, List, Optional, Sequence

from mmengine.logging import print_log
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS, MODELS
from mmengine.runner.checkpoint import load_state_dict
from mmengine.hooks import Hook
from mmengine.hooks.hook import DATA_BATCH


@HOOKS.register_module()
class MultiEMAHook(Hook):
    """Hook for maintaining multiple EMA weights.

    Features:
    1) `ema_momentum_list` supports maintaining multiple EMA weights with different momentums during a single training session.
    2) `eval_ema_index` specifies which EMA weight to use during validation/testing.
    3) All EMA states are saved in the checkpoint, with compatibility for the single-EMA `ema_state_dict`.

    Note:
        - `begin_iter` and `begin_epoch` cannot be set simultaneously.
        - By default, the EMA corresponding to `eval_ema_index` will be swapped with the `state_dict` and saved,
          facilitating direct loading for deployment/evaluation.
    """

    priority = 'NORMAL'

    def __init__(self,
                 ema_type: str = 'ExponentialMovingAverage',
                 strict_load: bool = False,
                 begin_iter: int = 0,
                 begin_epoch: int = 0,
                 ema_momentum_list: Sequence[float] = (0.0002, ),
                 eval_ema_index: int = 0,
                 **kwargs):
        self.strict_load = strict_load

        assert not (begin_iter != 0 and begin_epoch != 0), (
            '`begin_iter` and `begin_epoch` should not be both set.')
        assert begin_iter >= 0, ('`begin_iter` must larger than or equal to 0, '
                                 f'but got begin_iter: {begin_iter}')
        assert begin_epoch >= 0, ('`begin_epoch` must larger than or equal to 0, '
                                  f'but got begin_epoch: {begin_epoch}')

        self.begin_iter = begin_iter
        self.begin_epoch = begin_epoch
        self.enabled_by_epoch = self.begin_epoch > 0

        if 'momentum' in kwargs:
            print_log('`momentum` in kwargs is ignored because `ema_momentum_list` is used.',
                      'current', logging.WARNING)
            kwargs.pop('momentum')

        assert len(ema_momentum_list) > 0, '`ema_momentum_list` cannot be empty.'
        self.ema_momentum_list: List[float] = [float(m) for m in ema_momentum_list]
        for m in self.ema_momentum_list:
            assert 0 < m < 1, f'EMA momentum must be in (0, 1), but got {m}.'

        self.eval_ema_index = int(eval_ema_index)
        self._normalize_eval_index()

        self.ema_type = ema_type
        self.ema_kwargs = kwargs

        self.src_model = None
        self.ema_models = []

    def _normalize_eval_index(self):
        n = len(self.ema_momentum_list)
        if self.eval_ema_index < 0:
            self.eval_ema_index = n + self.eval_ema_index
        assert 0 <= self.eval_ema_index < n, (
            f'`eval_ema_index` out of range, got {self.eval_ema_index}, '
            f'valid range is [0, {n - 1}]')

    @property
    def selected_ema_model(self):
        return self.ema_models[self.eval_ema_index]

    def before_run(self, runner) -> None:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        self.src_model = model

        self.ema_models = []
        for m in self.ema_momentum_list:
            ema_cfg = dict(type=self.ema_type, momentum=m, **self.ema_kwargs)
            ema_model = MODELS.build(ema_cfg, default_args=dict(model=self.src_model))
            self.ema_models.append(ema_model)

    def before_train(self, runner) -> None:
        if self.enabled_by_epoch:
            assert self.begin_epoch <= runner.max_epochs, (
                'self.begin_epoch should be smaller than or equal to '
                f'runner.max_epochs: {runner.max_epochs}, but got '
                f'begin_epoch: {self.begin_epoch}')
        else:
            assert self.begin_iter <= runner.max_iters, (
                'self.begin_iter should be smaller than or equal to '
                f'runner.max_iters: {runner.max_iters}, but got '
                f'begin_iter: {self.begin_iter}')

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:
        if self._ema_started(runner):
            for ema_model in self.ema_models:
                ema_model.update_parameters(self.src_model)
        else:
            src_params = self.src_model.state_dict()
            for ema_model in self.ema_models:
                ema_params = ema_model.module.state_dict()
                for k, p in ema_params.items():
                    p.data.copy_(src_params[k].data)

    def before_val_epoch(self, runner) -> None:
        self._swap_ema_parameters()

    def after_val_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        self._swap_ema_parameters()

    def before_test_epoch(self, runner) -> None:
        self._swap_ema_parameters()

    def after_test_epoch(self, runner, metrics: Optional[Dict[str, float]] = None) -> None:
        self._swap_ema_parameters()

    def before_save_checkpoint(self, runner, checkpoint: dict) -> None:
        checkpoint['ema_state_dicts'] = [m.state_dict() for m in self.ema_models]
        checkpoint['ema_momentum_list'] = list(self.ema_momentum_list)
        checkpoint['ema_eval_index'] = self.eval_ema_index

        checkpoint['ema_state_dict'] = self.selected_ema_model.state_dict()

        self._swap_selected_ema_state_dict(checkpoint)

    def after_load_checkpoint(self, runner, checkpoint: dict) -> None:
        if 'ema_state_dicts' in checkpoint and runner._resume:
            self._swap_selected_ema_state_dict(checkpoint)

            ckpt_ema_list = checkpoint['ema_state_dicts']
            if len(ckpt_ema_list) != len(self.ema_models):
                print_log(
                    'Length mismatch between checkpoint `ema_state_dicts` and current '
                    '`ema_momentum_list`. Will load min(length).', 'current', logging.WARNING)

            n = min(len(ckpt_ema_list), len(self.ema_models))
            for i in range(n):
                self.ema_models[i].load_state_dict(ckpt_ema_list[i], strict=self.strict_load)

            for i in range(n, len(self.ema_models)):
                load_state_dict(self.ema_models[i].module,
                                copy.deepcopy(checkpoint['state_dict']),
                                strict=self.strict_load)

        elif 'ema_state_dict' in checkpoint and runner._resume:
            self._swap_selected_ema_state_dict(checkpoint)
            self.selected_ema_model.load_state_dict(checkpoint['ema_state_dict'],
                                                    strict=self.strict_load)

            for i, ema_model in enumerate(self.ema_models):
                if i == self.eval_ema_index:
                    continue
                load_state_dict(ema_model.module,
                                copy.deepcopy(checkpoint['state_dict']),
                                strict=self.strict_load)
        else:
            if runner._resume:
                print_log(
                    'There is no EMA state in checkpoint. '
                    'All EMA models will copy from `state_dict`.', 'current', logging.WARNING)
            for ema_model in self.ema_models:
                load_state_dict(ema_model.module,
                                copy.deepcopy(checkpoint['state_dict']),
                                strict=self.strict_load)

    def _swap_ema_parameters(self) -> None:
        ema_model = self.selected_ema_model

        avg_param = (itertools.chain(ema_model.module.parameters(), ema_model.module.buffers())
                     if ema_model.update_buffers else ema_model.module.parameters())
        src_param = (itertools.chain(self.src_model.parameters(), self.src_model.buffers())
                     if ema_model.update_buffers else self.src_model.parameters())

        for p_avg, p_src in zip(avg_param, src_param):
            tmp = p_avg.data.clone()
            p_avg.data.copy_(p_src.data)
            p_src.data.copy_(tmp)

    def _swap_selected_ema_state_dict(self, checkpoint: dict) -> None:
        model_state = checkpoint['state_dict']
        ema_state = checkpoint['ema_state_dict']  # selected EMA
        for k in ema_state:
            if k.startswith('module.'):
                model_k = k[7:]
                if model_k in model_state:
                    tmp = ema_state[k]
                    ema_state[k] = model_state[model_k]
                    model_state[model_k] = tmp

    def _ema_started(self, runner) -> bool:
        if self.enabled_by_epoch:
            return runner.epoch + 1 >= self.begin_epoch
        else:
            return runner.iter + 1 >= self.begin_iter


# custom_hooks = [
#     dict(
#         type='MultiEMAHook',
#         ema_type='ExponentialMovingAverage',
#         ema_momentum_list=[0.0001, 0.0002, 0.001],
#         eval_ema_index=1,
#         begin_iter=0,
#         strict_load=False)
# ]
