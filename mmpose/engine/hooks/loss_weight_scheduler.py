# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Optional

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmpose.registry import HOOKS
from mmpose.utils.hooks import rgetattr, rsetattr


def _smooth_interp(a: float, b: float, t: float, mode: str = 'cosine') -> float:
    t = max(0.0, min(1.0, float(t)))
    if mode == 'cosine':
        t = 0.5 - 0.5 * math.cos(math.pi * t)
    return a + (b - a) * t


@HOOKS.register_module()
class LossWeightSchedulerHook(Hook):
    priority = 51

    def __init__(self,
                 switch_epoch: int,
                 w_heat_front: float = 0.7,
                 w_cons_front: float = 0.3,
                 w_heat_back: float = 0.3,
                 w_cons_back: float = 0.5,
                 smooth: str = 'cosine',
                 verbose: bool = True,
                 head_path: str = 'head',
                 min_w_heat: Optional[float] = None,
                 zero_w_heat_epoch: int = -1):
        assert smooth in ('linear', 'cosine')
        self.switch_epoch = int(switch_epoch)
        self.w_heat_front = float(w_heat_front)
        self.w_cons_front = float(w_cons_front)
        self.w_heat_back = float(w_heat_back)
        self.w_cons_back = float(w_cons_back)
        self.smooth = smooth
        self.verbose = verbose
        self.head_path = str(head_path)
        self.min_w_heat = None if min_w_heat is None else float(min_w_heat)
        self.zero_w_heat_epoch = int(zero_w_heat_epoch)

    def before_train_epoch(self, runner: Runner):
        # unwrap DDP
        model = runner.model
        if is_model_wrapper(model):
            model = model.module

        # 注意：rgetattr 没有 default 参数，这里用 try/except 兜底
        try:
            head = rgetattr(model, self.head_path)
        except Exception:
            head = None

        if head is None:
            if hasattr(runner, 'logger') and runner.logger:
                runner.logger.warning(
                    f'[LossWeightSchedulerHook] Cannot find head at "{self.head_path}".'
                )
            return

        cur = runner.epoch
        maxe = runner.max_epochs

        # 计算权重
        if cur < self.switch_epoch or self.switch_epoch >= maxe:
            w_heat = self.w_heat_front
            w_cons = self.w_cons_front
        else:
            denom = max(1, (maxe - self.switch_epoch))
            t = (cur - self.switch_epoch) / denom
            w_heat = _smooth_interp(self.w_heat_front, self.w_heat_back, t, self.smooth)
            w_cons = _smooth_interp(self.w_cons_front, self.w_cons_back, t, self.smooth)

        if self.zero_w_heat_epoch >= 0 and cur >= self.zero_w_heat_epoch:
            w_heat = 0.0
        if self.min_w_heat is not None:
            w_heat = max(self.min_w_heat, w_heat)

        # 写回（DualHeadRTMCC 需有 w_heat / w_cons 属性）
        try:
            if hasattr(head, 'w_heat'):
                rsetattr(model, f'{self.head_path}.w_heat', float(w_heat))
            if hasattr(head, 'w_cons'):
                rsetattr(model, f'{self.head_path}.w_cons', float(w_cons))
        except Exception as e:
            if hasattr(runner, 'logger') and runner.logger:
                runner.logger.warning(
                    f'[LossWeightSchedulerHook] Failed to set weights: {e}'
                )

        if self.verbose and hasattr(runner, 'logger') and runner.logger:
            runner.logger.info(
                f'[LossWeightSchedulerHook] epoch={cur} '
                f'w_heat={w_heat:.4f}, w_cons={w_cons:.4f} '
                f'(switch={self.switch_epoch}, mode={self.smooth})'
            )
