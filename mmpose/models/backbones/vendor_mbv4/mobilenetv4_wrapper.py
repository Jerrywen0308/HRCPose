# mobilenetv4_wrapper.py
from __future__ import annotations
import torch
import torch.nn as nn

from mmpose.registry import MODELS   # MMPose 1.x / mmengine 风格注册
from mmengine.model import BaseModule

# 相对导入你刚才改好的实现与配置
from .mobilenetv4 import MobileNetV4
from .model_config import MODEL_SPECS




import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange





@MODELS.register_module()
class MobileNetV4Wrapper(BaseModule):
    """
    将 vendor 版 MobileNetV4（去掉 layer5，仅返回 (x1,x2,x3,x4)）包装为 MMPose backbone。

    Args:
        model (str): 变体名，必须在 MODEL_SPECS 里，如：
            - "MobileNetV4ConvSmall"
            - "MobileNetV4ConvMedium"
            - "MobileNetV4ConvLarge"
            - "MobileNetV4HybridMedium"
            - "MobileNetV4HybridLarge"
        out_indices (tuple[int]): 想要输出的阶段索引，范围 0..3，对应 (x1,x2,x3,x4)。
        frozen_stages (int): 冻结到哪一层，-1 表示不冻结；
            0 冻结 conv0，1 还会冻结 layer1，依此类推（最多到 4）。
        norm_eval (bool): 训练时将 BN 切到 eval 模式（不更新均值方差）。
        init_cfg (dict|None): mmengine 的初始化配置（可选）。
    """
    def __init__(
        self,
        model: str = 'MobileNetV4ConvSmall',
        out_indices: tuple[int, ...] = (2,),   # 常用：只取 x3(/16) 或 (3,) 取 x4(/32)
        frozen_stages: int = -1,
        norm_eval: bool = False,
        init_cfg: dict | None = None,
        use_p5_dwaspp: bool = False,
        p5_dilations=(1,2,3),
        p5_gamma_init: float = 0.0,
        
    ):
        super().__init__(init_cfg=init_cfg)

        assert model in MODEL_SPECS, f'Unknown model spec: {model}'
        self.model_name = model

        # 校验 out_indices
        self.out_indices = tuple(out_indices)
        for i in self.out_indices:
            assert 0 <= i <= 3, 'out_indices must be in range [0, 3] (x1..x4)'

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # 构建 vendor backbone（你已去掉 layer5，forward 返回 (x1,x2,x3,x4)）
#         self.backbone = MobileNetV4(model=self.model_name)
        self.backbone = MobileNetV4(
            model=self.model_name,
            use_p5_dwaspp=use_p5_dwaspp,
            p5_dilations=p5_dilations,
            p5_gamma_init=p5_gamma_init
        )

        # 记录每个输出阶段的通道数，方便你在配置里确认 neck.in_channels
        self._stage_channels = self._infer_stage_channels(MODEL_SPECS[self.model_name])
        # 公开一个属性（很多头/neck会用到）
        self.out_channels = tuple(self._stage_channels)

        # 冻结参数（若设置了 frozen_stages）
        if self.frozen_stages >= 0:
            self._freeze_stages()

    # --------- util: 从 spec 推断各 stage 最后输出通道 ----------
    @staticmethod
    def _infer_stage_channels(spec: dict) -> list[int]:
        stage_keys = ['layer1', 'layer2', 'layer3', 'layer4']
        chs = []
        for k in stage_keys:
            last = spec[k]['block_specs'][-1]
            assert isinstance(last, list), f'{k} last block must be a list spec'
            # 不同 block 的第二个元素都是 oup
            chs.append(int(last[1]))
        return chs  # [C1, C2, C3, C4] -> 分别对应 x1..x4

    # --------- 冻结阶段 ----------
    def _freeze_stages(self):
        # 顺序：conv0(0), layer1(1), layer2(2), layer3(3), layer4(4)
        stages = [
            self.backbone.conv0,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        ]
        max_idx = min(self.frozen_stages, len(stages) - 1)
        for i in range(max_idx + 1):
            m = stages[i]
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    # --------- 训练模式开关（可固定 BN） ----------
    def train(self, mode: bool = True):
        super().train(mode)
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    # --------- 前向 ----------
    def forward(self, x: torch.Tensor):
        # vendor backbone 返回 (x1, x2, x3, x4)
        feats = self.backbone(x)
        # 选取需要的 indices
        outs = tuple(feats[i] for i in self.out_indices)
        return outs

    
    
    
    
    
    
    
    
    
    
