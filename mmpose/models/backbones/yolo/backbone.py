
# Copyright (c) 2025.
# YOLOv12-turbo backbone for MMPose/MMEngine style
# Requires: block.py providing Conv, C3k2, A2C2f in the same package.
# Default outputs: P3/P4/P5 (strides 8/16/32)
#Gated Spatial Fusion：GSF模块的全称（也就是sppfdw）

from typing import List, Sequence, Tuple, Optional
import math
import torch
from torch import nn, Tensor
from torch.nn.modules.batchnorm import _BatchNorm

from mmengine.model import BaseModule
from mmpose.registry import MODELS

# 使用你已有的模块
try:
    from .block import Conv, C3k2, SPPF, C2PSA,A2C2f,LAE,DSC3k2,SPPF_DW,C2f,SPPF_DW_Lite,C2PSA,SPPF,DWASPP,C3k2UIB
except Exception:
    from block import Conv, C3k2, SPPF, C2PSA,A2C2f,LAE,DSC3k2,SPPF_DW,C2f,SPPF_DW_Lite,C2PSA,SPPF,DWASPP,C3k2UIB  # 允许: python backbone.py

def _make_divisible(x: int, divisor: int = 8) -> int:
    """Round channel number to be divisible by `divisor` (default 8)."""
    return int(math.ceil(x / divisor) * divisor)


@MODELS.register_module()
class YOLOv12TurboBackbone(BaseModule):
    """
    YOLOv12-turbo style backbone (P3-P5 outputs) for MMPose.

    Mapping to your YAML (backbone part only):
      0: Conv(64, 3, 2)                 -> layer0  (P1/2)
      1: Conv(128, 3, 2, p=1, g=2)      -> layer1  (P2/4)
      2: [*2] C3k2(256, False, e=0.25)  -> stage2
      3: Conv(256, 3, 2, p=1, g=4)      -> layer3  (P3/8)
      4: [*2] C3k2(512, False, e=0.25)  -> stage4  (==> P3 out)
      5: Conv(512, 3, 2)                -> layer5  (P4/16)
      6: [*4] A2C2f(512, a2=True, area=4)-> stage6 (==> P4 out)
      7: Conv(1024, 3, 2)               -> layer7  (P5/32)
      8: [*4] A2C2f(1024, a2=True, area=1)-> stage8 (==> P5 out)

    Args:
        deepen_factor (float): depth scaling for repeats (affects 2,4,6,8).
        widen_factor (float): width scaling applied to base channels.
        max_channels (int): channel cap after scaling.
        out_indices (Sequence[int]): which layer indices to output (default P3/P4/P5 -> (4,6,8)).
        norm_eval (bool): set BN to eval during training (freeze stats).
        init_cfg (dict): init config (Kaiming by default).

    Note:
        - This backbone **does not include head/neck**; only returns feature maps for P3/P4/P5.
        - Make sure your neck's `in_channels` matches the returned channels.
    """

    def __init__(self,
                 deepen_factor: float = 0.50,   # per YAML 'n' scale
                 widen_factor: float = 0.25,   # per YAML 'n' scale
                 max_channels: int = 1024,     # per YAML scales
                 out_indices: Sequence[int] = (4, 6, 8),
                 norm_eval: bool = False,
                 init_cfg: Optional[dict] = dict(
                     type='Kaiming', layer='Conv2d', a=math.sqrt(5),
                     distribution='uniform', mode='fan_in',
                     nonlinearity='leaky_relu')) -> None:
        super().__init__(init_cfg=init_cfg)

        self.out_indices = tuple(out_indices)
        self.norm_eval = norm_eval

        # ---- base channels (before width scaling) ----
        # keep alignment with YAML progression
        ch0 = 64
        ch1 = 128
        c_s2 = 256
        c_p3 = 256     # conv before P3 stage
        c3 = 512       # P3 out
        c_p4 = 512     # conv before P4 stage
        c4 = 512       # P4 out
        c_p5 = 1024    # conv before P5 stage
        c5 = 1024      # P5 out

        # ---- apply width scaling & cap ----
        def C(x):  # channel scaler
            return min(max_channels, _make_divisible(int(round(x * widen_factor)), 8))

        ch0, ch1 = C(ch0), C(ch1)
        c_s2, c_p3, c3 = C(c_s2), C(c_p3), C(c3)
        c_p4, c4 = C(c_p4), C(c4)
        c_p5, c5 = C(c_p5), C(c5)

        # ---- compute repeats (depth scaling) ----
        rep_2 = max(1, round(2 * deepen_factor))  # stage2
        rep_4 = max(1, round(2 * deepen_factor))  # stage4
        rep_6 = max(1, round(4 * deepen_factor))  # stage6
        rep_8 = max(1, round(4 * deepen_factor))  # stage8

        # ---- build layers ----
        # 0: Conv [64, 3, 2]
        self.layer0 = Conv(3, ch0, 3, 2)

        # 1: Conv [128, 3, 2, 1, 2]
        self.layer1 = Conv(ch0, ch1, 3, 2, 1, 2)

        # 2: repeats=2, C3k2 [256, False, 0.25]
        stage2_blocks = []
        in_c = ch1
        for i in range(rep_2):
            blk = C3k2(in_c, c_s2, n=1, c3k=False, e=0.25)
            stage2_blocks.append(blk)
            in_c = c_s2
        self.stage2 = nn.Sequential(*stage2_blocks)

        # 3: Conv [256, 3, 2, 1, 4]  -> P3/8
        self.layer3 = Conv(c_s2, c_p3, 3, 2, 1, 4)

        # 4: repeats=2, C3k2 [512, False, 0.25]  -> P3 out
        stage4_blocks = []
        in_c = c_p3
        for i in range(rep_4):
            blk = C3k2(in_c, c3, n=1, c3k=True, e=0.25, k=5)  # 仅改为 c3k=True, k=5
            stage4_blocks.append(blk)
            in_c = c3
        self.stage4 = nn.Sequential(*stage4_blocks)
        
    


        # 5: Conv [512, 3, 2]  -> P4/16
        self.layer5 = Conv(c3, c_p4, 3, 2)

        # 6: repeats=4, A2C2f [512, True, 4]  -> P4 out
        stage6_blocks = []
        in_c = c_p4
        for i in range(rep_6):
            blk = A2C2f(in_c, c4, n=1, a2=True, area=4, residual=True)  # default e=0.5 inside
            stage6_blocks.append(blk)
            in_c = c4
        self.stage6 = nn.Sequential(*stage6_blocks)

        # 7: Conv [1024, 3, 2]  -> P5/32
        self.layer7 = Conv(c4, c_p5, 3, 2)

        # 8: repeats=4, A2C2f [1024, True, 1] -> P5 out
        stage8_blocks = []
        in_c = c_p5
        for i in range(rep_8):
            blk = A2C2f(in_c, c5, n=1, a2=True, area=1, residual=True)
            stage8_blocks.append(blk)
            in_c = c5
        self.stage8 = nn.Sequential(*stage8_blocks)

        # 按 YAML 顺序登记，方便用 out_indices 直接对齐 [0..8]
        self.layers = [
            'layer0',  # 0
            'layer1',  # 1
            'stage2',  # 2
            'layer3',  # 3 (P3/8 pre)
            'stage4',  # 4 (P3 out)
            'layer5',  # 5 (P4/16 pre)
            'stage6',  # 6 (P4 out)
            'layer7',  # 7 (P5/32 pre)
            'stage8',  # 8 (P5 out)
        ]

        # 记录输出通道（按 out_indices 排序）
        all_out_channels = [ch0, ch1, c_s2, c_p3, c3, c_p4, c4, c_p5, c5]
        self.out_channels: List[int] = [all_out_channels[i] for i in self.out_indices]

    # --- training/eval BN 处理（与 MM 风格一致） ---
    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    # --- forward ---
    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs: List[Tensor] = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    
    


@MODELS.register_module()
class YOLOv12FastBackbone(BaseModule):
    """
    YOLOv12FastBackbone (P3/P4/P5 输出)

    设计目标：在 0.7 GFLOPs / ~3M 参数级别，最大化 GPU/TRT FPS，同时尽量不牺牲 AP。

    关键优化：
      - layer1/layer3 取消 group conv；
      - stage4 使用 C3k(k=3)；
      - stage6 默认 **关闭 a2** 或用 **C2f**；
      - stage8 保留 a2(area=1)（可开关）；
      - 通道按 32 对齐；
      - 可选全局替换 Conv 默认激活为 ReLU / Hard-Swish。

    Args:
        deepen_factor (float): 深度缩放，控制重复次数（stage2/4/6/8）。
        widen_factor (float): 宽度缩放，控制通道数。
        max_channels (int): 通道上限。
        out_indices (Sequence[int]): 输出层索引（默认 (4,6,8) -> P3/P4/P5）。
        norm_eval (bool): 训练时 BN 用 eval 模式（冻结统计）。
        set_global_conv_act (bool): 是否覆盖 Conv.default_act 为给定激活。
        act_type (str): "relu" | "hswish" | "silu"；仅当 set_global_conv_act=True 时生效。
        use_a2_p4 (bool): stage6 是否启用 area-attention（默认 False）。
        use_c2f_p4 (bool): stage6 是否直接使用 C2f（比 A2C2f(a2=False) 更简洁）。
        area_p4 (int): stage6 的 area（a2=True 时有效）。
        use_a2_p5 (bool): stage8 是否启用 area-attention（默认 True）。
        area_p5 (int): stage8 的 area。
        k_stage4 (int): stage4 的 C3k 大核尺寸（默认 3，更快）。
    """

    def __init__(
        self,
        deepen_factor: float = 0.50,
        widen_factor: float = 0.25,
        max_channels: int = 1024,
        out_indices: Sequence[int] = (4, 6, 8),
        norm_eval: bool = False,
        set_global_conv_act: bool = True,
        act_type: str = "silu",  # "relu" | "hswish" | "silu"
        use_a2_p4: bool = False,
        use_c2f_p4: bool = True,
        area_p4: int = 4,
        use_a2_p5: bool = True,
        area_p5: int = 1,
        k_stage4: int = 3,
        init_cfg: Optional[dict] = dict(
            type='Kaiming', layer='Conv2d', a=math.sqrt(5),
            distribution='uniform', mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        from types import SimpleNamespace
        self.cfg = SimpleNamespace(
            deepen_factor=deepen_factor, widen_factor=widen_factor,
            max_channels=max_channels, out_indices=tuple(out_indices),
            norm_eval=norm_eval, set_global_conv_act=set_global_conv_act,
            act_type=act_type, use_a2_p4=use_a2_p4, use_c2f_p4=use_c2f_p4,
            area_p4=area_p4, use_a2_p5=use_a2_p5, area_p5=area_p5,
            k_stage4=k_stage4,
        )
        self.out_indices = self.cfg.out_indices
        self.norm_eval = norm_eval

        # -------- 选择激活 --------
        act_layer: nn.Module
        if act_type.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act_type.lower() in ("hswish", "hardswish"):
            act_layer = nn.Hardswish()
        elif act_type.lower() in ("silu", "swish"):
            act_layer = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported act_type: {act_type}")

        # 可选：全局替换 Conv 的默认激活
        try:
            if set_global_conv_act:
                from inspect import isclass
                # Conv.default_act 期望一个 module 实例
                import types as _types
                # 防御式：某些工程 Conv.default_act 是类或实例，这里统一覆盖为实例
                import builtins  # noqa
                Conv.default_act = act_layer  # type: ignore  # noqa
        except Exception:
            pass

        # ---- base channels (before width scaling) ----
        ch0 = 64
        ch1 = 128
        c_s2 = 256
        c_p3 = 256
        c3 = 512
        c_p4 = 512
        c4 = 512
        c_p5 = 1024
        c5 = 1024

        # ---- apply width scaling & 32-align ----
        def C(x: int) -> int:
            return min(max_channels, _make_divisible(int(round(x * widen_factor)), 32))

        ch0, ch1 = C(ch0), C(ch1)
        c_s2, c_p3, c3 = C(c_s2), C(c_p3), C(c3)
        c_p4, c4 = C(c_p4), C(c4)
        c_p5, c5 = C(c_p5), C(c5)

        # ---- compute repeats (depth scaling) ----
        rep_2 = max(1, round(2 * deepen_factor))  # stage2
        rep_4 = max(1, round(2 * deepen_factor))  # stage4
        rep_6 = max(1, round(4 * deepen_factor))  # stage6
        rep_8 = max(1, round(4 * deepen_factor))  # stage8

        # ---- build layers ----
        # 0: Conv [64, 3, 2]
        self.layer0 = Conv(3, ch0, 3, 2, act=act_layer)

        # 1: Conv [128, 3, 2]  —— 去分组 (g=1)
        self.layer1 = Conv(ch0, ch1, 3, 2, 1, 1, act=act_layer)  # g=1 更快

        # 2: repeats=2, C3k2 [256, False, 0.25]
        stage2_blocks = []
        in_c = ch1
        for _ in range(rep_2):
            blk = C3k2(in_c, c_s2, n=1, c3k=False, e=0.25)
            stage2_blocks.append(blk)
            in_c = c_s2
        self.stage2 = nn.Sequential(*stage2_blocks)

        # 3: Conv [256, 3, 2]  —— 去分组 (g=1)
        self.layer3 = Conv(c_s2, c_p3, 3, 2, 1, 1, act=act_layer)  # g=1 更快

        # 4: repeats=2, C3k2  -> 使用 k=3 提速
        stage4_blocks = []
        in_c = c_p3
        for _ in range(rep_4):
            blk = C3k2(in_c, c3, n=1, c3k=False, e=0.25)
            stage4_blocks.append(blk)
            in_c = c3
        self.stage4 = nn.Sequential(*stage4_blocks)

        # 5: Conv [512, 3, 2]
        self.layer5 = Conv(c3, c_p4, 3, 2, act=act_layer)

        # 6: repeats=4, P4 —— 默认关闭 a2 或改用 C2f（速度优先）
        stage6_blocks = []
        in_c = c_p4
        for _ in range(rep_6):
            if self.cfg.use_c2f_p4:
                blk = C2f(in_c, c4, n=1, shortcut=True)
            else:
                blk = A2C2f(in_c, c4, n=1, a2=self.cfg.use_a2_p4, area=self.cfg.area_p4, residual=True)
            stage6_blocks.append(blk)
            in_c = c4
        self.stage6 = nn.Sequential(*stage6_blocks)

        # 7: Conv [1024, 3, 2]
        self.layer7 = Conv(c4, c_p5, 3, 2, act=act_layer)

        # 8: repeats=4, P5 —— 轻量 a2（area=1），可开关
        stage8_blocks = []
        in_c = c_p5
        for _ in range(rep_8):
            blk = A2C2f(in_c, c5, n=1, a2=self.cfg.use_a2_p5, area=self.cfg.area_p5, residual=True)
            stage8_blocks.append(blk)
            in_c = c5
        self.stage8 = nn.Sequential(*stage8_blocks)

        # 顺序登记，便于按 out_indices 取出
        self.layers = [
            'layer0',  # 0
            'layer1',  # 1
            'stage2',  # 2
            'layer3',  # 3 (P3/8 pre)
            'stage4',  # 4 (P3 out)
            'layer5',  # 5 (P4/16 pre)
            'stage6',  # 6 (P4 out)
            'layer7',  # 7 (P5/32 pre)
            'stage8',  # 8 (P5 out)
        ]

        all_out_channels = [ch0, ch1, c_s2, c_p3, c3, c_p4, c4, c_p5, c5]
        self.out_channels: List[int] = [all_out_channels[i] for i in self.out_indices]

    # --- 训练/评估时的 BN 处理 ---
    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):  # type: ignore
                    m.eval()

    # --- forward ---
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs: List[torch.Tensor] = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)



@MODELS.register_module()
class YOLOv12FasterBackbone(BaseModule):
    """
    YOLOv12FastBackbone (P3/P4/P5 输出)

    设计目标：在 0.7 GFLOPs / ~3M 参数级别，最大化 GPU/TRT FPS，同时尽量不牺牲 AP。

    关键优化：
      - layer1/layer3 取消 group conv；
      - stage4 使用 C3k(k=3)；
      - stage6 默认 **关闭 a2** 或用 **C2f**；
      - stage8 保留 a2(area=1)（可开关）；
      - 通道按 32 对齐；
      - 可选全局替换 Conv 默认激活为 ReLU / Hard-Swish。

    Args:
        deepen_factor (float): 深度缩放，控制重复次数（stage2/4/6/8）。
        widen_factor (float): 宽度缩放，控制通道数。
        max_channels (int): 通道上限。
        out_indices (Sequence[int]): 输出层索引（默认 (4,6,8) -> P3/P4/P5）。
        norm_eval (bool): 训练时 BN 用 eval 模式（冻结统计）。
        set_global_conv_act (bool): 是否覆盖 Conv.default_act 为给定激活。
        act_type (str): "relu" | "hswish" | "silu"；仅当 set_global_conv_act=True 时生效。
        use_a2_p4 (bool): stage6 是否启用 area-attention（默认 False）。
        use_c2f_p4 (bool): stage6 是否直接使用 C2f（比 A2C2f(a2=False) 更简洁）。
        area_p4 (int): stage6 的 area（a2=True 时有效）。
        use_a2_p5 (bool): stage8 是否启用 area-attention（默认 True）。
        area_p5 (int): stage8 的 area。
        k_stage4 (int): stage4 的 C3k 大核尺寸（默认 3，更快）。
    """

    def __init__(
        self,
        deepen_factor: float = 0.50,
        widen_factor: float = 0.25,
        max_channels: int = 1024,
        out_indices: Sequence[int] = (4, 6, 8),
        norm_eval: bool = False,
        set_global_conv_act: bool = True,
        act_type: str = "silu",  # "relu" | "hswish" | "silu"
        use_a2_p4: bool = False,
        use_c2f_p4: bool = True,
        area_p4: int = 4,
        use_a2_p5: bool = True,
        area_p5: int = 1,
        k_stage4: int = 3,
        init_cfg: Optional[dict] = dict(
            type='Kaiming', layer='Conv2d', a=math.sqrt(5),
            distribution='uniform', mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        from types import SimpleNamespace
        self.cfg = SimpleNamespace(
            deepen_factor=deepen_factor, widen_factor=widen_factor,
            max_channels=max_channels, out_indices=tuple(out_indices),
            norm_eval=norm_eval, set_global_conv_act=set_global_conv_act,
            act_type=act_type, use_a2_p4=use_a2_p4, use_c2f_p4=use_c2f_p4,
            area_p4=area_p4, use_a2_p5=use_a2_p5, area_p5=area_p5,
            k_stage4=k_stage4,
        )
        self.out_indices = self.cfg.out_indices
        self.norm_eval = norm_eval

        # -------- 选择激活 --------
        act_layer: nn.Module
        if act_type.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act_type.lower() in ("hswish", "hardswish"):
            act_layer = nn.Hardswish()
        elif act_type.lower() in ("silu", "swish"):
            act_layer = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported act_type: {act_type}")

        # 可选：全局替换 Conv 的默认激活
        try:
            if set_global_conv_act:
                from inspect import isclass
                # Conv.default_act 期望一个 module 实例
                import types as _types
                # 防御式：某些工程 Conv.default_act 是类或实例，这里统一覆盖为实例
                import builtins  # noqa
                Conv.default_act = act_layer  # type: ignore  # noqa
        except Exception:
            pass

        # ---- base channels (before width scaling) ----
        ch0 = 64
        ch1 = 128
        c_s2 = 256
        c_p3 = 256
        c3 = 512
        c_p4 = 512
        c4 = 512
        c_p5 = 1024
        c5 = 1024

        # ---- apply width scaling & 32-align ----
        def C(x: int) -> int:
            return min(max_channels, _make_divisible(int(round(x * widen_factor)), 32))

        ch0, ch1 = C(ch0), C(ch1)
        c_s2, c_p3, c3 = C(c_s2), C(c_p3), C(c3)
        c_p4, c4 = C(c_p4), C(c4)
        c_p5, c5 = C(c_p5), C(c5)

        # ---- compute repeats (depth scaling) ----
        rep_2 = max(1, round(2 * deepen_factor))  # stage2
        rep_4 = max(1, round(2 * deepen_factor))  # stage4
        rep_6 = max(1, round(4 * deepen_factor))  # stage6
        rep_8 = max(1, round(4 * deepen_factor))  # stage8

        # ---- build layers ----
        # 0: Conv [64, 3, 2]
        self.layer0 = Conv(3, ch0, 3, 2, act=act_layer)

        # 1: Conv [128, 3, 2]  —— 去分组 (g=1)
        self.layer1 = Conv(ch0, ch1, 3, 2, 1, 1, act=act_layer)  # g=1 更快

        # 2: repeats=2, C3k2 [256, False, 0.25]
        stage2_blocks = []
        in_c = ch1
        for _ in range(rep_2):
            blk = C3k2(in_c, c_s2, n=1, c3k=False, e=0.25)
            stage2_blocks.append(blk)
            in_c = c_s2
        self.stage2 = nn.Sequential(*stage2_blocks)

        # 3: Conv [256, 3, 2]  —— 去分组 (g=1)
        self.layer3 = Conv(c_s2, c_p3, 3, 2, 1, 1, act=act_layer)  # g=1 更快

        # 4: repeats=2, C3k2  -> 使用 k=3 提速
        stage4_blocks = []
        in_c = c_p3
        for _ in range(rep_4):
            blk = C3k2(in_c, c3, n=1, c3k=False, e=0.25)
            stage4_blocks.append(blk)
            in_c = c3
        self.stage4 = nn.Sequential(*stage4_blocks)

        # 5: Conv [512, 3, 2]
        self.layer5 = Conv(c3, c_p4, 3, 2, act=act_layer)

        # 6: repeats=4, P4 —— 默认关闭 a2 或改用 C2f（速度优先）
        stage6_blocks = []
        in_c = c_p4
        for _ in range(rep_6):
            if self.cfg.use_c2f_p4:
                blk = C3k2(in_c, c4, n=1, c3k=False, e=0.5)
#                 blk = C2f(in_c, c4, n=1, shortcut=True)
            else:
                blk = A2C2f(in_c, c4, n=1, a2=self.cfg.use_a2_p4, area=self.cfg.area_p4, residual=True)
            stage6_blocks.append(blk)
            in_c = c4
        self.stage6 = nn.Sequential(*stage6_blocks)

        # 7: Conv [1024, 3, 2]
        self.layer7 = Conv(c4, c_p5, 3, 2, act=act_layer)

        # 8: repeats=4, P5 —— 轻量 a2（area=1），可开关
        stage8_blocks = []
        in_c = c_p5
        for _ in range(rep_8):
            blk = A2C2f(in_c, c5, n=1, a2=self.cfg.use_a2_p5, area=self.cfg.area_p5, residual=True)
            stage8_blocks.append(blk)
            in_c = c5
        self.stage8 = nn.Sequential(*stage8_blocks)

        # 顺序登记，便于按 out_indices 取出
        self.layers = [
            'layer0',  # 0
            'layer1',  # 1
            'stage2',  # 2
            'layer3',  # 3 (P3/8 pre)
            'stage4',  # 4 (P3 out)
            'layer5',  # 5 (P4/16 pre)
            'stage6',  # 6 (P4 out)
            'layer7',  # 7 (P5/32 pre)
            'stage8',  # 8 (P5 out)
        ]

        all_out_channels = [ch0, ch1, c_s2, c_p3, c3, c_p4, c4, c_p5, c5]
        self.out_channels: List[int] = [all_out_channels[i] for i in self.out_indices]

    # --- 训练/评估时的 BN 处理 ---
    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):  # type: ignore
                    m.eval()

    # --- forward ---
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs: List[torch.Tensor] = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    
    
@MODELS.register_module()
class YOLOv12FastSPPFDWBackbone(BaseModule):
    """
    YOLOv12FastBackbone (P3/P4/P5 输出)

    设计目标：在 0.7 GFLOPs / ~3M 参数级别，最大化 GPU/TRT FPS，同时尽量不牺牲 AP。

    关键优化：
      - layer1/layer3 取消 group conv；
      - stage4 使用 C3k(k=3)；
      - stage6 默认 **关闭 a2** 或用 **C2f**；
      - stage8 保留 a2(area=1)（可开关）；
      - 通道按 32 对齐；
      - 可选全局替换 Conv 默认激活为 ReLU / Hard-Swish。

    Args:
        deepen_factor (float): 深度缩放，控制重复次数（stage2/4/6/8）。
        widen_factor (float): 宽度缩放，控制通道数。
        max_channels (int): 通道上限。
        out_indices (Sequence[int]): 输出层索引（默认 (4,6,8) -> P3/P4/P5）。
        norm_eval (bool): 训练时 BN 用 eval 模式（冻结统计）。
        set_global_conv_act (bool): 是否覆盖 Conv.default_act 为给定激活。
        act_type (str): "relu" | "hswish" | "silu"；仅当 set_global_conv_act=True 时生效。
        use_a2_p4 (bool): stage6 是否启用 area-attention（默认 False）。
        use_c2f_p4 (bool): stage6 是否直接使用 C2f（比 A2C2f(a2=False) 更简洁）。
        area_p4 (int): stage6 的 area（a2=True 时有效）。
        use_a2_p5 (bool): stage8 是否启用 area-attention（默认 True）。
        area_p5 (int): stage8 的 area。
        k_stage4 (int): stage4 的 C3k 大核尺寸（默认 3，更快）。
    """

    def __init__(
        self,
        deepen_factor: float = 0.50,
        widen_factor: float = 0.25,
        max_channels: int = 1024,
        out_indices: Sequence[int] = (4, 6, 8),
        norm_eval: bool = False,
        set_global_conv_act: bool = True,
        act_type: str = "silu",  # "relu" | "hswish" | "silu"
        use_a2_p4: bool = False,
        use_c2f_p4: bool = True,
        area_p4: int = 4,
        use_a2_p5: bool = True,
        area_p5: int = 1,
        k_stage4: int = 3,
        init_cfg: Optional[dict] = dict(
            type='Kaiming', layer='Conv2d', a=math.sqrt(5),
            distribution='uniform', mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        from types import SimpleNamespace
        self.cfg = SimpleNamespace(
            deepen_factor=deepen_factor, widen_factor=widen_factor,
            max_channels=max_channels, out_indices=tuple(out_indices),
            norm_eval=norm_eval, set_global_conv_act=set_global_conv_act,
            act_type=act_type, use_a2_p4=use_a2_p4, use_c2f_p4=use_c2f_p4,
            area_p4=area_p4, use_a2_p5=use_a2_p5, area_p5=area_p5,
            k_stage4=k_stage4,
        )
        self.out_indices = self.cfg.out_indices
        self.norm_eval = norm_eval

        # -------- 选择激活 --------
        act_layer: nn.Module
        if act_type.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act_type.lower() in ("hswish", "hardswish"):
            act_layer = nn.Hardswish()
        elif act_type.lower() in ("silu", "swish"):
            act_layer = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported act_type: {act_type}")

        # 可选：全局替换 Conv 的默认激活
        try:
            if set_global_conv_act:
                from inspect import isclass
                # Conv.default_act 期望一个 module 实例
                import types as _types
                # 防御式：某些工程 Conv.default_act 是类或实例，这里统一覆盖为实例
                import builtins  # noqa
                Conv.default_act = act_layer  # type: ignore  # noqa
        except Exception:
            pass

        # ---- base channels (before width scaling) ----
        ch0 = 64
        ch1 = 128
        c_s2 = 256
        c_p3 = 256
        c3 = 512
        c_p4 = 512
        c4 = 512
        c_p5 = 1024
        c5 = 1024

        # ---- apply width scaling & 32-align ----
        def C(x: int) -> int:
            return min(max_channels, _make_divisible(int(round(x * widen_factor)), 32))

        ch0, ch1 = C(ch0), C(ch1)
        c_s2, c_p3, c3 = C(c_s2), C(c_p3), C(c3)
        c_p4, c4 = C(c_p4), C(c4)
        c_p5, c5 = C(c_p5), C(c5)

        # ---- compute repeats (depth scaling) ----
        rep_2 = max(1, round(2 * deepen_factor))  # stage2
        rep_4 = max(1, round(2 * deepen_factor))  # stage4
        rep_6 = max(1, round(4 * deepen_factor))  # stage6
        rep_8 = max(1, round(4 * deepen_factor))  # stage8

        # ---- build layers ----
        # 0: Conv [64, 3, 2]
        self.layer0 = Conv(3, ch0, 3, 2, act=act_layer)

        # 1: Conv [128, 3, 2]  —— 去分组 (g=1)
        self.layer1 = Conv(ch0, ch1, 3, 2, 1, 1, act=act_layer)  # g=1 更快

        # 2: repeats=2, C3k2 [256, False, 0.25]
        stage2_blocks = []
        in_c = ch1
        for _ in range(rep_2):
            blk = C3k2(in_c, c_s2, n=1, c3k=False, e=0.25)
            stage2_blocks.append(blk)
            in_c = c_s2
        self.stage2 = nn.Sequential(*stage2_blocks)

        # 3: Conv [256, 3, 2]  —— 去分组 (g=1)
        self.layer3 = Conv(c_s2, c_p3, 3, 2, 1, 1, act=act_layer)  # g=1 更快

        # 4: repeats=2, C3k2  -> 使用 k=3 提速
        stage4_blocks = []
        in_c = c_p3
        for _ in range(rep_4):
            blk = C3k2(in_c, c3, n=1, c3k=False, e=0.25)
            stage4_blocks.append(blk)
            in_c = c3
        self.stage4 = nn.Sequential(*stage4_blocks)

        # 5: Conv [512, 3, 2]
        self.layer5 = Conv(c3, c_p4, 3, 2, act=act_layer)

        # 6: repeats=4, P4 —— 默认关闭 a2 或改用 C2f（速度优先）
        stage6_blocks = []
        in_c = c_p4
        for _ in range(rep_6):
            if self.cfg.use_c2f_p4:
                blk = C2f(in_c, c4, n=1, shortcut=True)
            else:
                blk = A2C2f(in_c, c4, n=1, a2=self.cfg.use_a2_p4, area=self.cfg.area_p4, residual=True)
            stage6_blocks.append(blk)
            in_c = c4
        self.stage6 = nn.Sequential(*stage6_blocks)

        # 7: Conv [1024, 3, 2]
        self.layer7 = Conv(c4, c_p5, 3, 2, act=act_layer)
                # ★ 新增：P5 的 SPPF_DW（放在最底层）
        self.sppf_p5 = SPPF_DW_Lite(c_p5, k=5, e=0.5)

        # 8: repeats=4, P5 —— 轻量 a2（area=1），可开关
        stage8_blocks = []
        stage8_blocks.append(self.sppf_p5)
        in_c = c_p5
        for _ in range(rep_8):
            blk = A2C2f(in_c, c5, n=1, a2=self.cfg.use_a2_p5, area=self.cfg.area_p5, residual=True)
            stage8_blocks.append(blk)
            in_c = c5
        self.stage8 = nn.Sequential(*stage8_blocks)

        # 顺序登记，便于按 out_indices 取出
        self.layers = [
            'layer0',  # 0
            'layer1',  # 1
            'stage2',  # 2
            'layer3',  # 3 (P3/8 pre)
            'stage4',  # 4 (P3 out)
            'layer5',  # 5 (P4/16 pre)
            'stage6',  # 6 (P4 out)
            'layer7',  # 7 (P5/32 pre)
            'stage8',  # 8 (P5 out)
        ]

        all_out_channels = [ch0, ch1, c_s2, c_p3, c3, c_p4, c4, c_p5, c5]
        self.out_channels: List[int] = [all_out_channels[i] for i in self.out_indices]

    # --- 训练/评估时的 BN 处理 ---
    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):  # type: ignore
                    m.eval()

    # --- forward ---
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs: List[torch.Tensor] = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

@MODELS.register_module()
class YOLOv12FasterSPPFDWBackbone(BaseModule):
    """
    YOLOv12FastBackbone (P3/P4/P5 输出)

    设计目标：在 0.7 GFLOPs / ~3M 参数级别，最大化 GPU/TRT FPS，同时尽量不牺牲 AP。

    关键优化：
      - layer1/layer3 取消 group conv；
      - stage4 使用 C3k(k=3)；
      - stage6 默认 **关闭 a2** 或用 **C2f**；
      - stage8 保留 a2(area=1)（可开关）；
      - 通道按 32 对齐；
      - 可选全局替换 Conv 默认激活为 ReLU / Hard-Swish。

    Args:
        deepen_factor (float): 深度缩放，控制重复次数（stage2/4/6/8）。
        widen_factor (float): 宽度缩放，控制通道数。
        max_channels (int): 通道上限。
        out_indices (Sequence[int]): 输出层索引（默认 (4,6,8) -> P3/P4/P5）。
        norm_eval (bool): 训练时 BN 用 eval 模式（冻结统计）。
        set_global_conv_act (bool): 是否覆盖 Conv.default_act 为给定激活。
        act_type (str): "relu" | "hswish" | "silu"；仅当 set_global_conv_act=True 时生效。
        use_a2_p4 (bool): stage6 是否启用 area-attention（默认 False）。
        use_c2f_p4 (bool): stage6 是否直接使用 C2f（比 A2C2f(a2=False) 更简洁）。
        area_p4 (int): stage6 的 area（a2=True 时有效）。
        use_a2_p5 (bool): stage8 是否启用 area-attention（默认 True）。
        area_p5 (int): stage8 的 area。
        k_stage4 (int): stage4 的 C3k 大核尺寸（默认 3，更快）。
    """

    def __init__(
        self,
        deepen_factor: float = 0.50,
        widen_factor: float = 0.25,
        max_channels: int = 1024,
        out_indices: Sequence[int] = (4, 6, 8),
        norm_eval: bool = False,
        set_global_conv_act: bool = True,
        act_type: str = "silu",  # "relu" | "hswish" | "silu"
        use_a2_p4: bool = False,
        use_c2f_p4: bool = True,
        area_p4: int = 4,
        use_a2_p5: bool = True,
        area_p5: int = 1,
        k_stage4: int = 3,
        init_cfg: Optional[dict] = dict(
            type='Kaiming', layer='Conv2d', a=math.sqrt(5),
            distribution='uniform', mode='fan_in',
            nonlinearity='leaky_relu')
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        from types import SimpleNamespace
        self.cfg = SimpleNamespace(
            deepen_factor=deepen_factor, widen_factor=widen_factor,
            max_channels=max_channels, out_indices=tuple(out_indices),
            norm_eval=norm_eval, set_global_conv_act=set_global_conv_act,
            act_type=act_type, use_a2_p4=use_a2_p4, use_c2f_p4=use_c2f_p4,
            area_p4=area_p4, use_a2_p5=use_a2_p5, area_p5=area_p5,
            k_stage4=k_stage4,
        )
        self.out_indices = self.cfg.out_indices
        self.norm_eval = norm_eval

        # -------- 选择激活 --------
        act_layer: nn.Module
        if act_type.lower() == "relu":
            act_layer = nn.ReLU(inplace=True)
        elif act_type.lower() in ("hswish", "hardswish"):
            act_layer = nn.Hardswish()
        elif act_type.lower() in ("silu", "swish"):
            act_layer = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unsupported act_type: {act_type}")

        # 可选：全局替换 Conv 的默认激活
        try:
            if set_global_conv_act:
                from inspect import isclass
                # Conv.default_act 期望一个 module 实例
                import types as _types
                # 防御式：某些工程 Conv.default_act 是类或实例，这里统一覆盖为实例
                import builtins  # noqa
                Conv.default_act = act_layer  # type: ignore  # noqa
        except Exception:
            pass

        # ---- base channels (before width scaling) ----
        ch0 = 64
        ch1 = 128
        c_s2 = 256
        c_p3 = 256
        c3 = 512
        c_p4 = 512
        c4 = 512
        c_p5 = 1024
        c5 = 1024

        # ---- apply width scaling & 32-align ----
        def C(x: int) -> int:
            return min(max_channels, _make_divisible(int(round(x * widen_factor)), 32))

        ch0, ch1 = C(ch0), C(ch1)
        c_s2, c_p3, c3 = C(c_s2), C(c_p3), C(c3)
        c_p4, c4 = C(c_p4), C(c4)
        c_p5, c5 = C(c_p5), C(c5)

        # ---- compute repeats (depth scaling) ----
        rep_2 = max(1, round(2 * deepen_factor))  # stage2
        rep_4 = max(1, round(2 * deepen_factor))  # stage4
        rep_6 = max(1, round(4 * deepen_factor))  # stage6
        rep_8 = max(1, round(4 * deepen_factor))  # stage8

        # ---- build layers ----
        # 0: Conv [64, 3, 2]
        self.layer0 = Conv(3, ch0, 3, 2, act=act_layer)

        # 1: Conv [128, 3, 2]  —— 去分组 (g=1)
        self.layer1 = Conv(ch0, ch1, 3, 2, 1, 1, act=act_layer)  # g=1 更快

        # 2: repeats=2, C3k2 [256, False, 0.25]
        stage2_blocks = []
        in_c = ch1
        for _ in range(rep_2):
            blk = C3k2(in_c, c_s2, n=1, c3k=False, e=0.25)
            stage2_blocks.append(blk)
            in_c = c_s2
        self.stage2 = nn.Sequential(*stage2_blocks)

        # 3: Conv [256, 3, 2]  —— 去分组 (g=1)
        self.layer3 = Conv(c_s2, c_p3, 3, 2, 1, 1, act=act_layer)  # g=1 更快

        # 4: repeats=2, C3k2  -> 使用 k=3 提速
        stage4_blocks = []
        in_c = c_p3
        for _ in range(rep_4):
            blk = C3k2(in_c, c3, n=1, c3k=False, e=0.25)
            stage4_blocks.append(blk)
            in_c = c3
        self.stage4 = nn.Sequential(*stage4_blocks)

        # 5: Conv [512, 3, 2]
        self.layer5 = Conv(c3, c_p4, 3, 2, act=act_layer)

        # 6: repeats=4, P4 —— 默认关闭 a2 或改用 C2f（速度优先）
        stage6_blocks = []
        in_c = c_p4
        for _ in range(rep_6):
            if self.cfg.use_c2f_p4:
                blk = C3k2(in_c, c4, n=1, c3k=False, e=0.5)
#                 blk = C2f(in_c, c4, n=1, shortcut=True)
            else:
                blk = A2C2f(in_c, c4, n=1, a2=self.cfg.use_a2_p4, area=self.cfg.area_p4, residual=True)
            stage6_blocks.append(blk)
            in_c = c4
        self.stage6 = nn.Sequential(*stage6_blocks)

        # 7: Conv [1024, 3, 2]
        self.layer7 = Conv(c4, c_p5, 3, 2, act=act_layer)
                # ★ 新增：P5 的 SPPF_DW（放在最底层）
        self.sppf_p5 = SPPF_DW_Lite(c_p5, k=5, e=0.5)

        # 8: repeats=4, P5 —— 轻量 a2（area=1），可开关
        stage8_blocks = []
        stage8_blocks.append(self.sppf_p5)
        in_c = c_p5
        for _ in range(rep_8):
            blk = A2C2f(in_c, c5, n=1, a2=self.cfg.use_a2_p5, area=self.cfg.area_p5, residual=True)
            stage8_blocks.append(blk)
            in_c = c5
        self.stage8 = nn.Sequential(*stage8_blocks)

        # 顺序登记，便于按 out_indices 取出
        self.layers = [
            'layer0',  # 0
            'layer1',  # 1
            'stage2',  # 2
            'layer3',  # 3 (P3/8 pre)
            'stage4',  # 4 (P3 out)
            'layer5',  # 5 (P4/16 pre)
            'stage6',  # 6 (P4 out)
            'layer7',  # 7 (P5/32 pre)
            'stage8',  # 8 (P5 out)
        ]

        all_out_channels = [ch0, ch1, c_s2, c_p3, c3, c_p4, c4, c_p5, c5]
        self.out_channels: List[int] = [all_out_channels[i] for i in self.out_indices]

    # --- 训练/评估时的 BN 处理 ---
    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):  # type: ignore
                    m.eval()

    # --- forward ---
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outs: List[torch.Tensor] = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    
@MODELS.register_module()
class YOLOv12SPPFDWBackbone(BaseModule):
    """
    YOLOv12-turbo style backbone (P3-P5 outputs) for MMPose.

    Mapping to your YAML (backbone part only):
      0: Conv(64, 3, 2)                 -> layer0  (P1/2)
      1: Conv(128, 3, 2, p=1, g=2)      -> layer1  (P2/4)
      2: [*2] C3k2(256, False, e=0.25)  -> stage2
      3: Conv(256, 3, 2, p=1, g=4)      -> layer3  (P3/8)
      4: [*2] C3k2(512, False, e=0.25)  -> stage4  (==> P3 out)
      5: Conv(512, 3, 2)                -> layer5  (P4/16)
      6: [*4] A2C2f(512, a2=True, area=4)-> stage6 (==> P4 out)
      7: Conv(1024, 3, 2)               -> layer7  (P5/32)
      8: [*4] A2C2f(1024, a2=True, area=1)-> stage8 (==> P5 out)

    Args:
        deepen_factor (float): depth scaling for repeats (affects 2,4,6,8).
        widen_factor (float): width scaling applied to base channels.
        max_channels (int): channel cap after scaling.
        out_indices (Sequence[int]): which layer indices to output (default P3/P4/P5 -> (4,6,8)).
        norm_eval (bool): set BN to eval during training (freeze stats).
        init_cfg (dict): init config (Kaiming by default).

    Note:
        - This backbone **does not include head/neck**; only returns feature maps for P3/P4/P5.
        - Make sure your neck's `in_channels` matches the returned channels.
    """

    def __init__(self,
                 deepen_factor: float = 0.50,   # per YAML 'n' scale
                 widen_factor: float = 0.25,   # per YAML 'n' scale
                 max_channels: int = 1024,     # per YAML scales
                 out_indices: Sequence[int] = (4, 6, 8),
                 norm_eval: bool = False,
                 init_cfg: Optional[dict] = dict(
                     type='Kaiming', layer='Conv2d', a=math.sqrt(5),
                     distribution='uniform', mode='fan_in',
                     nonlinearity='leaky_relu')) -> None:
        super().__init__(init_cfg=init_cfg)

        self.out_indices = tuple(out_indices)
        self.norm_eval = norm_eval

        # ---- base channels (before width scaling) ----
        # keep alignment with YAML progression
        ch0 = 64
        ch1 = 128
        c_s2 = 256
        c_p3 = 256     # conv before P3 stage
        c3 = 512       # P3 out
        c_p4 = 512     # conv before P4 stage
        c4 = 512       # P4 out
        c_p5 = 1024    # conv before P5 stage
        c5 = 1024      # P5 out

        # ---- apply width scaling & cap ----
        def C(x):  # channel scaler
            return min(max_channels, _make_divisible(int(round(x * widen_factor)), 8))

        ch0, ch1 = C(ch0), C(ch1)
        c_s2, c_p3, c3 = C(c_s2), C(c_p3), C(c3)
        c_p4, c4 = C(c_p4), C(c4)
        c_p5, c5 = C(c_p5), C(c5)

        # ---- compute repeats (depth scaling) ----
        rep_2 = max(1, round(2 * deepen_factor))  # stage2
        rep_4 = max(1, round(2 * deepen_factor))  # stage4
        rep_6 = max(1, round(4 * deepen_factor))  # stage6
        rep_8 = max(1, round(4 * deepen_factor))  # stage8

        # ---- build layers ----
        # 0: Conv [64, 3, 2]
        self.layer0 = Conv(3, ch0, 3, 2)

        # 1: Conv [128, 3, 2, 1, 2]
        self.layer1 = Conv(ch0, ch1, 3, 2, 1, 2)

        # 2: repeats=2, C3k2 [256, False, 0.25]
        stage2_blocks = []
        in_c = ch1
        for i in range(rep_2):
            blk = C3k2(in_c, c_s2, n=1, c3k=False, e=0.25)
            stage2_blocks.append(blk)
            in_c = c_s2
        self.stage2 = nn.Sequential(*stage2_blocks)

        # 3: Conv [256, 3, 2, 1, 4]  -> P3/8
        self.layer3 = Conv(c_s2, c_p3, 3, 2, 1, 4)

        # 4: repeats=2, C3k2 [512, False, 0.25]  -> P3 out
        stage4_blocks = []
        in_c = c_p3
        for i in range(rep_4):
            blk = C3k2(in_c, c3, n=1, c3k=True, e=0.25, k=5)  # 仅改为 c3k=True, k=5
            stage4_blocks.append(blk)
            in_c = c3
        self.stage4 = nn.Sequential(*stage4_blocks)
        
    


        # 5: Conv [512, 3, 2]  -> P4/16
        self.layer5 = Conv(c3, c_p4, 3, 2)

        # 6: repeats=4, A2C2f [512, True, 4]  -> P4 out
        stage6_blocks = []
        in_c = c_p4
        for i in range(rep_6):
            blk = A2C2f(in_c, c4, n=1, a2=True, area=4, residual=True)  # default e=0.5 inside
            stage6_blocks.append(blk)
            in_c = c4
        self.stage6 = nn.Sequential(*stage6_blocks)

        # 7: Conv [1024, 3, 2]  -> P5/32
        self.layer7 = Conv(c4, c_p5, 3, 2)
        
        # ★ 新增：P5 的 SPPF_DW（放在最底层）
        self.sppf_p5 = SPPF_DW(c_p5, c_p5, k=5, e=0.5)

        # 8: repeats=4, A2C2f [1024, True, 1] -> P5 out
        stage8_blocks = []
        # 先接入 SPPF
        stage8_blocks.append(self.sppf_p5)
        in_c = c_p5
        for i in range(rep_8):
            blk = A2C2f(in_c, c5, n=1, a2=True, area=1, residual=True)
            stage8_blocks.append(blk)
            in_c = c5
        self.stage8 = nn.Sequential(*stage8_blocks)

        # 按 YAML 顺序登记，方便用 out_indices 直接对齐 [0..8]
        self.layers = [
            'layer0',  # 0
            'layer1',  # 1
            'stage2',  # 2
            'layer3',  # 3 (P3/8 pre)
            'stage4',  # 4 (P3 out)
            'layer5',  # 5 (P4/16 pre)
            'stage6',  # 6 (P4 out)
            'layer7',  # 7 (P5/32 pre)
            'stage8',  # 8 (P5 out)
        ]

        # 记录输出通道（按 out_indices 排序）
        all_out_channels = [ch0, ch1, c_s2, c_p3, c3, c_p4, c4, c_p5, c5]
        self.out_channels: List[int] = [all_out_channels[i] for i in self.out_indices]

    # --- training/eval BN 处理（与 MM 风格一致） ---
    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    # --- forward ---
    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs: List[Tensor] = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    
    
    
    
    



@MODELS.register_module()
class YOLOv11Backbone(BaseModule):
    """YOLOv11-n style backbone with P3/P4/P5 outputs."""
    def __init__(self,
                 deepen_factor: float = 0.50,   # scales.n depth
                 widen_factor: float = 0.25,    # scales.n width
                 max_channels: int = 1024,
                 out_indices: Sequence[int] = (4, 6, 10),  # P3, P4, P5
                 norm_eval: bool = False,
                 init_cfg: Optional[dict] = dict(
                     type='Kaiming', layer='Conv2d', a=math.sqrt(5),
                     distribution='uniform', mode='fan_in',
                     nonlinearity='leaky_relu')) -> None:
        super().__init__(init_cfg=init_cfg)
        self.out_indices = tuple(out_indices)
        self.norm_eval = norm_eval

        # ---- base channels before width scaling (from yaml) ----
        ch0, ch1 = 64, 128
        c_s2, c_p3, c3 = 256, 256, 512
        c_p4, c4 = 512, 512
        c_p5, c5 = 1024, 1024

        def C(x):  # width scale + cap + 8-divisible
            return min(max_channels, _make_divisible(int(round(x * widen_factor)), 8))

        ch0, ch1 = C(ch0), C(ch1)
        c_s2, c_p3, c3 = C(c_s2), C(c_p3), C(c3)
        c_p4, c4 = C(c_p4), C(c4)
        c_p5, c5 = C(c_p5), C(c5)

        # repeats (depth scaling): yaml 为 2/2/2/2，这里按 0.5 缩成 1
        rep_2 = max(1, round(2 * deepen_factor))  # stage2
        rep_4 = max(1, round(2 * deepen_factor))  # stage4
        rep_6 = max(1, round(2 * deepen_factor))  # stage6
        rep_8 = max(1, round(2 * deepen_factor))  # stage8
        rep_10 = max(1, round(2 * deepen_factor)) # stage10 (C2PSA)

        # 0: Conv [64,3,2] -> P1/2
        self.layer0 = Conv(3, ch0, 3, 2)
        # 1: Conv [128,3,2] -> P2/4
        self.layer1 = Conv(ch0, ch1, 3, 2)
        # 2: C3k2 x rep_2 -> [256, False, 0.25]
        self.stage2 = nn.Sequential(*[
            C3k2(ch1 if i == 0 else c_s2, c_s2, n=1, c3k=False, e=0.25)
            for i in range(rep_2)
        ])
        # 3: Conv [256,3,2] -> P3/8
        self.layer3 = Conv(c_s2, c_p3, 3, 2)
        # 4: C3k2 x rep_4 -> [512, False, 0.25]  => P3 out
        self.stage4 = nn.Sequential(*[
            C3k2(c_p3 if i == 0 else c3, c3, n=1, c3k=False, e=0.25)
            for i in range(rep_4)
        ])
        # 5: Conv [512,3,2] -> P4/16
        self.layer5 = Conv(c3, c_p4, 3, 2)
        # 6: C3k2 x rep_6 -> [512, True]        => P4 out
        self.stage6 = nn.Sequential(*[
            C3k2(c_p4 if i == 0 else c4, c4, n=1, c3k=True)
            for i in range(rep_6)
        ])
        # 7: Conv [1024,3,2] -> P5/32
        self.layer7 = Conv(c4, c_p5, 3, 2)
        
        
        # 8: C3k2 x rep_8 -> [1024, True]
        self.stage8 = nn.Sequential(*[
            C3k2(c_p5 if i == 0 else c5, c5, n=1, c3k=True)
            for i in range(rep_8)
        ])
        # 9: SPPF [1024, 5]
        self.layer9 = SPPF(c5, c5, k=5)
        # 10: C2PSA x rep_10 -> [1024]          => P5 out
        self.stage10 = nn.Sequential(*[
            C2PSA(c5, c5, n=1)
            for _ in range(rep_10)
        ])

        self.layers = [
            'layer0',  # 0
            'layer1',  # 1
            'stage2',  # 2
            'layer3',  # 3 (P3/8 pre)
            'stage4',  # 4 (P3 out)
            'layer5',  # 5 (P4/16 pre)
            'stage6',  # 6 (P4 out)
            'layer7',  # 7 (P5/32 pre)
            'stage8',  # 8
            'layer9',  # 9
            'stage10', # 10 (P5 out)
        ]
        all_out_channels = [ch0, ch1, c_s2, c_p3, c3, c_p4, c4, c_p5, c5, c5, c5]
        self.out_channels: List[int] = [all_out_channels[i] for i in self.out_indices]

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs: List[Tensor] = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    
@MODELS.register_module()
class YOLOv11SPPFDWBackbone(BaseModule):
    """YOLOv11-n style backbone with P3/P4/P5 outputs."""
    def __init__(self,
                 deepen_factor: float = 0.50,   # scales.n depth
                 widen_factor: float = 0.25,    # scales.n width
                 max_channels: int = 1024,
                 out_indices: Sequence[int] = (4, 6, 10),  # P3, P4, P5
                 norm_eval: bool = False,
                 init_cfg: Optional[dict] = dict(
                     type='Kaiming', layer='Conv2d', a=math.sqrt(5),
                     distribution='uniform', mode='fan_in',
                     nonlinearity='leaky_relu')) -> None:
        super().__init__(init_cfg=init_cfg)
        self.out_indices = tuple(out_indices)
        self.norm_eval = norm_eval

        # ---- base channels before width scaling (from yaml) ----
        ch0, ch1 = 64, 128
        c_s2, c_p3, c3 = 256, 256, 512
        c_p4, c4 = 512, 512
        c_p5, c5 = 1024, 1024

        def C(x):  # width scale + cap + 8-divisible
            return min(max_channels, _make_divisible(int(round(x * widen_factor)), 8))

        ch0, ch1 = C(ch0), C(ch1)
        c_s2, c_p3, c3 = C(c_s2), C(c_p3), C(c3)
        c_p4, c4 = C(c_p4), C(c4)
        c_p5, c5 = C(c_p5), C(c5)

        # repeats (depth scaling): yaml 为 2/2/2/2，这里按 0.5 缩成 1
        rep_2 = max(1, round(2 * deepen_factor))  # stage2
        rep_4 = max(1, round(2 * deepen_factor))  # stage4
        rep_6 = max(1, round(2 * deepen_factor))  # stage6
        rep_8 = max(1, round(2 * deepen_factor))  # stage8
        rep_10 = max(1, round(2 * deepen_factor)) # stage10 (C2PSA)

        # 0: Conv [64,3,2] -> P1/2
        self.layer0 = Conv(3, ch0, 3, 2)
        # 1: Conv [128,3,2] -> P2/4
        self.layer1 = Conv(ch0, ch1, 3, 2)
        # 2: C3k2 x rep_2 -> [256, False, 0.25]
        self.stage2 = nn.Sequential(*[
            C3k2(ch1 if i == 0 else c_s2, c_s2, n=1, c3k=False, e=0.25)
            for i in range(rep_2)
        ])
        # 3: Conv [256,3,2] -> P3/8
        self.layer3 = Conv(c_s2, c_p3, 3, 2)
        # 4: C3k2 x rep_4 -> [512, False, 0.25]  => P3 out
        self.stage4 = nn.Sequential(*[
            C3k2(c_p3 if i == 0 else c3, c3, n=1, c3k=False, e=0.25)
            for i in range(rep_4)
        ])
        # 5: Conv [512,3,2] -> P4/16
        self.layer5 = Conv(c3, c_p4, 3, 2)
        # 6: C3k2 x rep_6 -> [512, True]        => P4 out
        self.stage6 = nn.Sequential(*[
            C3k2(c_p4 if i == 0 else c4, c4, n=1, c3k=True)
            for i in range(rep_6)
        ])
        # 7: Conv [1024,3,2] -> P5/32
        self.layer7 = Conv(c4, c_p5, 3, 2)
        
        
        # 8: C3k2 x rep_8 -> [1024, True]
        self.stage8 = nn.Sequential(*[
            C3k2(c_p5 if i == 0 else c5, c5, n=1, c3k=True)
            for i in range(rep_8)
        ])
        # 9: SPPF [1024, 5]
#         self.layer9 = SPPF(c5, c5, k=5)
#         self.layer9 = SPPF_DW(c5,c5,k=5)
#         self.layer9 = SPPF_DW_Lite(c5, k=5, e=0.5,delta=2)
        self.layer9 = DWASPP(c5, c5, dilations=(1,2,3), gamma_init=0.0)
        
        # 10: C2PSA x rep_10 -> [1024]          => P5 out
        self.stage10 = nn.Sequential(*[
            C2PSA(c5, c5, n=1)
            for _ in range(rep_10)
        ])

        self.layers = [
            'layer0',  # 0
            'layer1',  # 1
            'stage2',  # 2
            'layer3',  # 3 (P3/8 pre)
            'stage4',  # 4 (P3 out)
            'layer5',  # 5 (P4/16 pre)
            'stage6',  # 6 (P4 out)
            'layer7',  # 7 (P5/32 pre)
            'stage8',  # 8
            'layer9',  # 9
            'stage10', # 10 (P5 out)
        ]
        all_out_channels = [ch0, ch1, c_s2, c_p3, c3, c_p4, c4, c_p5, c5, c5, c5]
        self.out_channels: List[int] = [all_out_channels[i] for i in self.out_indices]

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs: List[Tensor] = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

@MODELS.register_module()
class YOLOv11UIBBackbone(BaseModule):
    """YOLOv11-n style backbone with P3/P4/P5 outputs."""
    def __init__(self,
                 deepen_factor: float = 0.50,   # scales.n depth
                 widen_factor: float = 0.25,    # scales.n width
                 max_channels: int = 1024,
                 out_indices: Sequence[int] = (4, 6, 10),  # P3, P4, P5
                 norm_eval: bool = False,
                 init_cfg: Optional[dict] = dict(
                     type='Kaiming', layer='Conv2d', a=math.sqrt(5),
                     distribution='uniform', mode='fan_in',
                     nonlinearity='leaky_relu')) -> None:
        super().__init__(init_cfg=init_cfg)
        self.out_indices = tuple(out_indices)
        self.norm_eval = norm_eval

        # ---- base channels before width scaling (from yaml) ----
        ch0, ch1 = 64, 128
        c_s2, c_p3, c3 = 256, 256, 512
        c_p4, c4 = 512, 512
        c_p5, c5 = 1024, 1024

        def C(x):  # width scale + cap + 8-divisible
            return min(max_channels, _make_divisible(int(round(x * widen_factor)), 8))

        ch0, ch1 = C(ch0), C(ch1)
        c_s2, c_p3, c3 = C(c_s2), C(c_p3), C(c3)
        c_p4, c4 = C(c_p4), C(c4)
        c_p5, c5 = C(c_p5), C(c5)

        # repeats (depth scaling): yaml 为 2/2/2/2，这里按 0.5 缩成 1
        rep_2 = max(1, round(2 * deepen_factor))  # stage2
        rep_4 = max(1, round(2 * deepen_factor))  # stage4
        rep_6 = max(1, round(2 * deepen_factor))  # stage6
        rep_8 = max(1, round(2 * deepen_factor))  # stage8
        rep_10 = max(1, round(2 * deepen_factor)) # stage10 (C2PSA)

        # 0: Conv [64,3,2] -> P1/2
        self.layer0 = Conv(3, ch0, 3, 2)
        # 1: Conv [128,3,2] -> P2/4
        self.layer1 = Conv(ch0, ch1, 3, 2)

        # 2: C3k2UIB x rep_2 -> [256, False, 0.25]
        self.stage2 = nn.Sequential(*[
            C3k2UIB(ch1 if i == 0 else c_s2, c_s2, n=1, c3k=False, e=0.25,k=3)  # CHANGED
            for i in range(rep_2)
        ])

        # 3: Conv [256,3,2] -> P3/8
        self.layer3 = Conv(c_s2, c_p3, 3, 2)

        # 4: C3k2UIB x rep_4 -> [512, False, 0.25]  => P3 out
        self.stage4 = nn.Sequential(*[
            C3k2UIB(c_p3 if i == 0 else c3, c3, n=1, c3k=False, e=0.25,k=3)     # CHANGED
            for i in range(rep_4)
        ])

        # 5: Conv [512,3,2] -> P4/16
        self.layer5 = Conv(c3, c_p4, 3, 2)

        # 6: C3k2UIB x rep_6 -> [512, True]        => P4 out
        self.stage6 = nn.Sequential(*[
            C3k2UIB(c_p4 if i == 0 else c4, c4, n=1, c3k=True)              # CHANGED
            for i in range(rep_6)
        ])

        # 7: Conv [1024,3,2] -> P5/32
        self.layer7 = Conv(c4, c_p5, 3, 2)

        # 8: C3k2UIB x rep_8 -> [1024, True]
        self.stage8 = nn.Sequential(*[
            C3k2UIB(c_p5 if i == 0 else c5, c5, n=1, c3k=True)              # CHANGED
            for i in range(rep_8)
        ])

        # 9: SPPF [1024, 5]
        self.layer9 = SPPF(c5, c5, k=5)

        # 10: C2PSA x rep_10 -> [1024]          => P5 out
        self.stage10 = nn.Sequential(*[
            C2PSA(c5, c5, n=1)
            for _ in range(rep_10)
        ])

        self.layers = [
            'layer0',  # 0
            'layer1',  # 1
            'stage2',  # 2
            'layer3',  # 3 (P3/8 pre)
            'stage4',  # 4 (P3 out)
            'layer5',  # 5 (P4/16 pre)
            'stage6',  # 6 (P4 out)
            'layer7',  # 7 (P5/32 pre)
            'stage8',  # 8
            'layer9',  # 9
            'stage10', # 10 (P5 out)
        ]
        all_out_channels = [ch0, ch1, c_s2, c_p3, c3, c_p4, c4, c_p5, c5, c5, c5]
        self.out_channels: List[int] = [all_out_channels[i] for i in self.out_indices]

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs: List[Tensor] = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    
    
    
@MODELS.register_module()
class YOLOv12LAEBackbone(BaseModule):
    """
    YOLOv12-LAE style backbone (P3-P5 outputs) for MMPose.

    改动要点：
      - 除前两层 Conv 外，所有下采样层统一替换为 LAE（通道不变，stride=2）
      - 去除下采样前后的中间通道，通道变化由 C3k2 / A2C2f 完成
      - P3 段采用大核 C3k2(k=7)

    YAML 对应（简化后的语义）：
      0: Conv(64, 3, 2)              -> layer0  (P1/2)
      1: Conv(128, 3, 2, p=1, g=2)   -> layer1  (P2/4)
      2: [*2] C3k2(256, False, 0.25) -> stage2
      3: LAE                         -> layer3  (P3/8)
      4: [*2] C3k2(→ 512, k=7)       -> stage4  (P3 out)
      5: LAE                         -> layer5  (P4/16)
      6: [*4] A2C2f(→ 512, area=4)   -> stage6  (P4 out)
      7: LAE                         -> layer7  (P5/32)
      8: [*4] A2C2f(→ 1024, area=1)  -> stage8  (P5 out)
    """

    def __init__(self,
                 deepen_factor: float = 0.50,   # 对应 n 尺度
                 widen_factor: float = 0.25,    # 对应 n 尺度
                 max_channels: int = 1024,
                 out_indices: Sequence[int] = (4, 6, 8),  # P3, P4, P5
                 norm_eval: bool = False,
                 init_cfg: Optional[dict] = dict(
                     type='Kaiming', layer='Conv2d', a=math.sqrt(5),
                     distribution='uniform', mode='fan_in',
                     nonlinearity='leaky_relu')) -> None:
        super().__init__(init_cfg=init_cfg)

        self.out_indices = tuple(out_indices)
        self.norm_eval = norm_eval

        # ---- base channels (before width scaling) ----
        # 与原 yolov12n 一致的基线通道
        ch0 = 64     # after layer0
        ch1 = 128    # after layer1
        c_s2 = 256   # stage2 out -> 输入给 LAE(P3)
        c3   = 512   # stage4 out (P3 out)
        c4   = 512   # stage6 out (P4 out)
        c5   = 1024  # stage8 out (P5 out)

        # ---- width scaling & cap ----
        def C(x: int) -> int:
            return min(max_channels, _make_divisible(int(round(x * widen_factor)), 8))

        ch0, ch1 = C(ch0), C(ch1)
        c_s2, c3, c4, c5 = C(c_s2), C(c3), C(c4), C(c5)

        # ---- depth scaling (repeats) ----
        rep_2 = max(1, round(2 * deepen_factor))  # stage2
        rep_4 = max(1, round(2 * deepen_factor))  # stage4
        rep_6 = max(1, round(4 * deepen_factor))  # stage6
        rep_8 = max(1, round(4 * deepen_factor))  # stage8

        # ---------------- build layers ----------------
        # 0: Conv [64, 3, 2] -> P1/2
        self.layer0 = Conv(3, ch0, 3, 2)

        # 1: Conv [128, 3, 2, 1, 2] -> P2/4
        self.layer1 = Conv(ch0, ch1, 3, 2, 1, 2)

        # 2: C3k2 堆叠（不改变分辨率），输出通道 -> c_s2
        stage2_blocks = []
        in_c = ch1
        for _ in range(rep_2):
            stage2_blocks.append(C3k2(in_c, c_s2, n=1, c3k=False, e=0.25))  # 常规 C3k2
            in_c = c_s2
        self.stage2 = nn.Sequential(*stage2_blocks)

        # 3: 下采样到 P3/8 —— 用 LAE（不改通道）
        #    输入来自 c_s2，输出仍为 c_s2
        self.layer3 = LAE(c_s2)

        # 4: P3 段 C3k2（大核 k=7），输出通道 -> c3
        stage4_blocks = []
        in_c = c_s2
        for _ in range(rep_4):
            # 原：DSC3k2（深度可分离 + 7x7 大核）
            # blk = DSC3k2(in_c, c3, n=1, dsc3k=True, e=0.25, k1=3, k2=7, d2=1, shortcut=True)

            # 现：普通 C3k2，使用 5x5 大核
            blk = C3k2(
                in_c, c3,
                n=1,
                c3k=True,   # 开启 k-variant
                e=0.25,
                k=5         # 大核改为 5x5
                # 如果你的 C3k2 实现支持 shortcut 参数且默认关闭，需要就手动加上：
                # shortcut=True
            )
            stage4_blocks.append(blk)
            in_c = c3
        self.stage4 = nn.Sequential(*stage4_blocks)


        # 5: 下采样到 P4/16 —— 用 LAE（不改通道）
        #    输入 c3，输出仍为 c3
        self.layer5 = LAE(c3)

        # 6: P4 段 A2C2f（注意力版 ELAN），输出通道 -> c4
        stage6_blocks = []
        in_c = c3
        for _ in range(rep_6):
            stage6_blocks.append(A2C2f(in_c, c4, n=1, a2=True, area=4, residual=True))  # residual=True 更稳
            in_c = c4
        self.stage6 = nn.Sequential(*stage6_blocks)

        # 7: 下采样到 P5/32 —— 用 LAE（不改通道）
        #    输入 c4，输出仍为 c4
        self.layer7 = LAE(c4)

        # 8: P5 段 A2C2f，输出通道 -> c5
        stage8_blocks = []
        in_c = c4
        for _ in range(rep_8):
            stage8_blocks.append(A2C2f(in_c, c5, n=1, a2=True, area=1, residual=False))
            in_c = c5
        self.stage8 = nn.Sequential(*stage8_blocks)

        # 顺序表（便于按照 out_indices 取出）
        self.layers = [
            'layer0',  # 0
            'layer1',  # 1
            'stage2',  # 2
            'layer3',  # 3 (P3/8 via LAE)
            'stage4',  # 4 (P3 out)
            'layer5',  # 5 (P4/16 via LAE)
            'stage6',  # 6 (P4 out)
            'layer7',  # 7 (P5/32 via LAE)
            'stage8',  # 8 (P5 out)
        ]

        # 记录输出通道（与上面各层一一对应）
        all_out_channels = [ch0, ch1, c_s2, c_s2, c3, c3, c4, c4, c5]
        self.out_channels: List[int] = [all_out_channels[i] for i in self.out_indices]

    # --- keep BN in eval if norm_eval True ---
    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    # --- forward ---
    def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
        outs: List[Tensor] = []
        for i, name in enumerate(self.layers):
            x = getattr(self, name)(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    


# @MODELS.register_module()
# class YOLOv11Backbone(BaseModule):
#     """YOLOv11-n style backbone with P3/P4/P5 outputs.

#     Args:
#         deepen_factor (float): depth scale (default 0.50)
#         widen_factor (float): width scale (default 0.25)
#         max_channels (int): channel cap (default 1024)
#         out_indices (Sequence[int]): output layer indices (default (4, 6, 10))
#         norm_eval (bool): keep BN in eval during training
#     """

#     def __init__(
#         self,
#         deepen_factor: float = 0.50,
#         widen_factor: float = 0.25,
#         max_channels: int = 1024,
#         out_indices: Sequence[int] = (4, 6, 10),
#         norm_eval: bool = False,
#         init_cfg: Optional[dict] = dict(
#             type='Kaiming', layer='Conv2d', a=math.sqrt(5),
#             distribution='uniform', mode='fan_in',
#             nonlinearity='leaky_relu')
#     ) -> None:
#         super().__init__() if BaseModule is nn.Module else super().__init__(init_cfg=init_cfg)
#         self.out_indices = tuple(out_indices)
#         self.norm_eval = norm_eval

#         # ---- base channels before width scaling (from YAML) ----
#         ch0, ch1 = 64, 128
#         c_s2, c_p3, c3 = 256, 256, 512
#         c_p4, c4 = 512, 512
#         c_p5, c5 = 1024, 1024

#         def C(x: int) -> int:
#             return min(max_channels, _make_divisible(int(round(x * widen_factor)), 8))

#         ch0, ch1 = C(ch0), C(ch1)
#         c_s2, c_p3, c3 = C(c_s2), C(c_p3), C(c3)
#         c_p4, c4 = C(c_p4), C(c4)
#         c_p5, c5 = C(c_p5), C(c5)

#         # repeats (depth scaling): YAML nominally 2; with 0.50 -> 1
#         rep_2 = max(1, round(2 * deepen_factor))  # stage2
#         rep_4 = max(1, round(2 * deepen_factor))  # stage4
#         rep_6 = max(1, round(2 * deepen_factor))  # stage6
#         rep_8 = max(1, round(2 * deepen_factor))  # stage8
#         rep_10 = max(1, round(2 * deepen_factor)) # stage10

#         # 0: Conv [64,3,2] -> P1/2
#         self.layer0 = Conv(3, ch0, 3, 2)
#         # 1: Conv [128,3,2] -> P2/4
#         self.layer1 = Conv(ch0, ch1, 3, 2)
#         # 2: C3k2 x rep_2 -> [256, False, 0.25]
#         self.stage2 = nn.Sequential(*[
#             C3k2(ch1 if i == 0 else c_s2, c_s2, n=1, c3k=False, e=0.25)
#             for i in range(rep_2)
#         ])
#         # 3: Conv [256,3,2] -> P3/8
#         self.layer3 = Conv(c_s2, c_p3, 3, 2)
#         # 4: C3k2 x rep_4 -> [512, False, 0.25]  => P3 out
#         self.stage4 = nn.Sequential(*[
#             C3k2(c_p3 if i == 0 else c3, c3, n=1, c3k=False, e=0.25)
#             for i in range(rep_4)
#         ])
#         # 5: Conv [512,3,2] -> P4/16
#         self.layer5 = Conv(c3, c_p4, 3, 2)
#         # 6: C3k2 x rep_6 -> [512, True]        => P4 out
#         self.stage6 = nn.Sequential(*[
#             C3k2(c_p4 if i == 0 else c4, c4, n=1, c3k=True)
#             for i in range(rep_6)
#         ])
#         # 7: Conv [1024,3,2] -> P5/32
#         self.layer7 = Conv(c4, c_p5, 3, 2)
#         # 8: C3k2 x rep_8 -> [1024, True]
#         self.stage8 = nn.Sequential(*[
#             C3k2(c_p5 if i == 0 else c5, c5, n=1, c3k=True)
#             for i in range(rep_8)
#         ])
#         # 9: SPPF [1024, 5]
#         self.layer9 = SPPF(c5, c5, k=5)
#         # 10: C2PSA x rep_10 -> [1024]          => P5 out
#         self.stage10 = nn.Sequential(*[
#             C2PSA(c5, c5, n=1)
#             for _ in range(rep_10)
#         ])

#         self.layers = [
#             'layer0',  # 0
#             'layer1',  # 1
#             'stage2',  # 2
#             'layer3',  # 3 (P3/8 pre)
#             'stage4',  # 4 (P3 out)
#             'layer5',  # 5 (P4/16 pre)
#             'stage6',  # 6 (P4 out)
#             'layer7',  # 7 (P5/32 pre)
#             'stage8',  # 8
#             'layer9',  # 9
#             'stage10', # 10 (P5 out)
#         ]
#         all_out_channels = [ch0, ch1, c_s2, c_p3, c3, c_p4, c4, c_p5, c5, c5, c5]
#         self.out_channels: List[int] = [all_out_channels[i] for i in self.out_indices]

#     def train(self, mode: bool = True) -> None:
#         super().train(mode)
#         if mode and self.norm_eval:
#             for m in self.modules():
#                 if isinstance(m, _BatchNorm):
#                     m.eval()

#     def forward(self, x: Tensor) -> Tuple[Tensor, ...]:
#         outs: List[Tensor] = []
#         for i, name in enumerate(self.layers):
#             x = getattr(self, name)(x)
#             if i in self.out_indices:
#                 outs.append(x)
#         return tuple(outs)

    
    
    
    
    
    
    
    
    
# quick self-test
if __name__ == "__main__":
    net = YOLOv11Backbone(deepen_factor=0.50, widen_factor=0.25, max_channels=1024,
                          out_indices=(4, 6, 10), norm_eval=False)
    x = torch.randn(1, 3, 640, 640)
    ys = net(x)
    print("Output shapes:", [y.shape for y in ys])
    print("out_channels:", net.out_channels)

