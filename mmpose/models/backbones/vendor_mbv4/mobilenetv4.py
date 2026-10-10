from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_config import MODEL_SPECS

def make_divisible(
        value: float,
        divisor: int,
        min_value: Optional[float] = None,
        round_down_protect: bool = True,
    ) -> int:
    """
    This function is copied from here 
    "https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_layers.py"
    
    This is to ensure that all layers have channels that are divisible by 8.

    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
        will be allowed.

    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU())
    return conv

class SqueezeExcitation(nn.Module):
    """True SE block: GAP -> FC(r) -> ReLU -> FC(C) -> Sigmoid -> scale."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        w = self.pool(x)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.gate(w)
        return x * w


class InvertedResidual(nn.Module):
    """
    清理后的 Fused-IB/IB 变体：
    - 先 1x1 expand（不降采样）
    - 再 3x3 depthwise（可 stride=1/2，下采样只在这里发生）
    - 可选 SE
    - 最后 1x1 projection（默认无激活）
    - 残差：stride==1 且 输入输出通道一致
    依赖上游的 conv_2d 工厂：Conv(+BN)[+ReLU]，groups=inp 时为 DWConv。
    """
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: float,
        act: bool = False,                # 是否在 projection 处保留激活（默认 False 更常见）
        squeeze_excitation: bool = False, # 是否添加真实 SE
        se_reduction: int = 4
    ):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))

        layers = nn.Sequential()

        # 1) 1x1 expand（不在这里做下采样）
        if expand_ratio != 1:
            layers.add_module(
                'exp_1x1',
                conv_2d(inp, hidden_dim, kernel_size=1, stride=1, bias=False, norm=True, act=True)
            )
        else:
            hidden_dim = inp  # 无扩展时直接保持通道

        # 2) 3x3 depthwise（唯一的下采样位置）
        layers.add_module(
            'dw_3x3',
            conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, bias=False, norm=True, act=True)
        )

        # 3) Squeeze-and-Excitation（可选，真正的 SE）
        if squeeze_excitation:
            layers.add_module('se', SqueezeExcitation(hidden_dim, reduction=se_reduction))

        # 4) 1x1 projection（通常无激活；若想保留激活，可将 act=True）
        layers.add_module(
            'proj_1x1',
            conv_2d(hidden_dim, oup, kernel_size=1, stride=1, bias=False, norm=True, act=act)
        )

        self.block = layers
        self.use_res_connect = (stride == 1 and inp == oup)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res_connect else out


class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self,
            inp,
            oup,
            start_dw_kernel_size,
            middle_dw_kernel_size,
            middle_dw_downsample,
            stride,
            expand_ratio,
            use_layer_scale=False
        ):
        super().__init__()
        self.use_res_connect = (inp == oup and stride == 1)

        # ---- 放在这里：下采样一致性检查（尚未创建任何层之前） ----
        if stride == 2:
            # 不能同时在 start 和 middle 都下采样
            if start_dw_kernel_size and middle_dw_kernel_size:
                assert not middle_dw_downsample, \
                    "Downsample at either start_dw OR middle_dw, not both."
            # 至少要有一处进行下采样（否则 stride=2 无处生效）
            assert (bool(start_dw_kernel_size) or
                    (bool(middle_dw_kernel_size) and bool(middle_dw_downsample))), \
                "When stride=2, downsample must occur at start_dw or middle_dw."

        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(
                inp, inp, kernel_size=start_dw_kernel_size,
                stride=stride_, groups=inp, act=False
            )

        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)

        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(
                expand_filters, expand_filters, kernel_size=middle_dw_kernel_size,
                stride=stride_, groups=expand_filters
            )

        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)

        self._use_layer_scale = use_layer_scale
        if self._use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale = MNV4LayerScale(oup, self.layer_scale_init_value)

    def forward(self, x):
        shortcut = x
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
            # print("_start_dw_", x.shape)
        x = self._expand_conv(x)
        # print("_expand_conv", x.shape)
        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
            # print("_middle_dw", x.shape)
        x = self._proj_conv(x)
        # print("_proj_conv", x.shape)
        if self._use_layer_scale:
            x = self.layer_scale(x)
        if self.use_res_connect:
            x = x + shortcut
        return x

class MultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(
        self,
        inp: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        query_h_strides: int,
        query_w_strides: int,
        kv_strides: int,
        dw_kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        """Multi-Query Attention (MQA) with spatial downsampling on Q / K,V."""
        super().__init__()
        assert key_dim % num_heads == 0, 'key_dim must be divisible by num_heads'
        self.num_heads = num_heads
        self.key_dim = key_dim                # total key dim
        self.value_dim = value_dim            # total value dim (shared for all heads)
        self.head_dim = key_dim // num_heads  # per-head dim

        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides

        # Q path
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            self._query_downsampling_norm = nn.BatchNorm2d(inp)
        # project Q to [B, heads*key_dim, Hq, Wq]
        self._query_proj = conv_2d(inp, key_dim, 1, 1, norm=False, act=False)

        # K / V path (shared across heads)
        self.key = nn.Sequential()
        self.value = nn.Sequential()
        if self.kv_strides > 1:
            self.key.add_module(
                '_key_dw_conv',
                conv_2d(inp, inp, dw_kernel_size, self.kv_strides, groups=inp, norm=True, act=False),
            )
            self.value.add_module(
                '_value_dw_conv',
                conv_2d(inp, inp, dw_kernel_size, self.kv_strides, groups=inp, norm=True, act=False),
            )
        # IMPORTANT: K projects to head_dim (per-head), V projects to value_dim (shared)
        self.key.add_module('_key_proj', conv_2d(inp, self.head_dim, 1, 1, norm=False, act=False))
        self.value.add_module('_value_proj', conv_2d(inp, value_dim, 1, 1, norm=False, act=False))

        # final output projection: concat heads*value_dim -> C_in
        self._output_proj = conv_2d(num_heads * value_dim, inp, 1, 1, norm=False, act=False)

        self.attn_drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, H, W]
        returns: [B, C_in, H//qh, W//qw]
        """
        B, C, H, W = x.shape

        # ---- Q ----
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q_in = F.avg_pool2d(
                x,
                kernel_size=(self.query_h_strides, self.query_w_strides),
                stride=(self.query_h_strides, self.query_w_strides),
            )
            q_in = self._query_downsampling_norm(q_in)
        else:
            q_in = x
        # [B, heads*key_dim, Hq, Wq]
        q = self._query_proj(q_in)
        Hq, Wq = q.shape[-2:]
        # [B, heads, Lq, head_dim]
        q = q.view(B, self.num_heads, self.head_dim, Hq * Wq).transpose(2, 3).contiguous()

        # ---- K, V ----
        # K: [B, head_dim, Hk, Wk]; V: [B, value_dim, Hk, Wk]
        k = self.key(x)
        v = self.value(x)
        Hk, Wk = k.shape[-2:]

        # reshape to sequence:
        # k: [B, 1, Lk, head_dim] ; v: [B, 1, Lk, value_dim]
        k = k.view(B, 1, self.head_dim, Hk * Wk).transpose(2, 3).contiguous()
        v = v.view(B, 1, self.value_dim, Hk * Wk).transpose(2, 3).contiguous()

        # ---- Attention ----
        # q @ k^T -> [B, heads, Lq, Lk]
        q = q * (self.head_dim ** -0.5)
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # out: [B, heads, Lq, value_dim]
        out = torch.matmul(attn, v)

        # ---- fold back ----
        # -> [B, heads*value_dim, Hq, Wq] -> project to C_in
        out = out.transpose(2, 3).contiguous().view(B, self.num_heads * self.value_dim, Hq, Wq)
        out = self._output_proj(out)
        return out

    
class MNV4LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        """LayerScale as introduced in CaiT: https://arxiv.org/abs/2103.17239
        Referenced from here https://github.com/tensorflow/models/blob/master/official/vision/modeling/layers/nn_blocks.py
       
        As used in MobileNetV4.

        Attributes:
            init_value (float): value to initialize the diagonal matrix of LayerScale.
        """
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x * gamma

class MultiHeadSelfAttentionBlock(nn.Module):
    """
    2D特征版的 MHSA Block（本实现仅支持 Multi-Query Attention 路径）。
    - 输入/输出均为 [B, C, H, W]
    - 可选：对 Query 做空间下采样（query_h_strides, query_w_strides）
    - 残差分支会用同样的步幅对 shortcut 做 AvgPool 以匹配空间尺寸
    """
    def __init__(
        self,
        inp: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        query_h_strides: int,
        query_w_strides: int,
        kv_strides: int,
        use_layer_scale: bool,
        use_multi_query: bool,
        use_residual: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):
        super().__init__()
        # 仅支持 2D MQA 路径（nn.MultiheadAttention 期望序列形状，这里不用）
        if not use_multi_query:
            raise NotImplementedError(
                "This 2D MHSA block supports only Multi-Query Attention (use_multi_query=True)."
            )

        self.use_layer_scale = use_layer_scale
        self.use_residual = use_residual
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides

        # 输入归一化
        self._input_norm = nn.BatchNorm2d(inp)

        # Multi-Query Attention（已修复的实现，需先定义好该类）
        self.multi_query_attention = MultiQueryAttentionLayerWithDownSampling(
            inp=inp,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            query_h_strides=query_h_strides,
            query_w_strides=query_w_strides,
            kv_strides=kv_strides,
        )

        # LayerScale（逐通道缩放）
        if self.use_layer_scale:
            self.layer_scale = MNV4LayerScale(inp, init_values=layer_scale_init_value)

        # 残差分支下采样（若 Q 做了下采样，则对 shortcut 做同样的下采样以对齐空间尺寸）
        if self.use_residual and (query_h_strides > 1 or query_w_strides > 1):
            self.shortcut_down = nn.AvgPool2d(
                kernel_size=(query_h_strides, query_w_strides),
                stride=(query_h_strides, query_w_strides),
            )
        else:
            self.shortcut_down = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        shortcut = x
        x = self._input_norm(x)

        # MQA 路径（输出通道为 C，空间可能变为 H//qh, W//qw）
        x = self.multi_query_attention(x)

        if self.use_layer_scale:
            x = self.layer_scale(x)

        if self.use_residual:
            shortcut = self.shortcut_down(shortcut)
            x = x + shortcut

        return x


def build_blocks(layer_spec):
    if not layer_spec.get('block_name'):
        return nn.Sequential()
    block_names = layer_spec['block_name']
    layers = nn.Sequential()
    if block_names == "convbn":
        schema_ = ['inp', 'oup', 'kernel_size', 'stride']
        for i in range(layer_spec['num_blocks']):
            specs_ = layer_spec['block_specs'][i]
            if isinstance(specs_, list):
                args = dict(zip(schema_, layer_spec['block_specs'][i]))
                layers.add_module(f"convbn_{i}", conv_2d(**args))
            elif specs_ == "AdaptiveAvgPool2d":
                layers.add_module(f"AdaptiveAvgPool2d", nn.AdaptiveAvgPool2d(1))
            else:
                raise NotImplementedError
    elif block_names == "uib":
        schema_ =  ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride', 'expand_ratio', 'use_layer_scale', 'mhsa']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            mhsa = args.pop("mhsa") if "mhsa" in args else 0
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
            if mhsa:
                mhsa_schema_ = [
                    "inp", "num_heads", "key_dim", "value_dim", "query_h_strides", "query_w_strides", "kv_strides", 
                    "use_layer_scale", "use_multi_query", "use_residual"
                ]
                args = dict(zip(mhsa_schema_, [args['oup']] + (mhsa)))
                layers.add_module(f"mhsa_{i}", MultiHeadSelfAttentionBlock(**args))
    elif block_names == "fused_ib":
        schema_ = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers




class DWASPP(nn.Module):
    """
    并联 DW-3x3(d=1/2/3) + BN + SiLU + 1x1 → (c2)
    标量门控残差：out = x + γ * (y - x)，约定 c1 == c2。
    """
    def __init__(self, c1, c2, dilations=(1, 2, 3), gamma_init: float = 0.0):
        super().__init__()
        assert c1 == c2, "此精简版假定 c1 == c2，若通道不等请恢复投影分支"
        self.dws = nn.ModuleList([
            nn.Conv2d(c1, c1, kernel_size=3, padding=d, dilation=d, groups=c1, bias=False)
            for d in dilations
        ])
        self.bn  = nn.BatchNorm2d(c1)
        self.act = nn.SiLU(inplace=True)
        # 1x1 可学习融合（无激活，通道保持 c1==c2）
        self.pw  = conv_2d(c1, c2, kernel_size=1, stride=1, bias=False, norm=True, act=True)
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, x):
        y = 0
        for dw in self.dws:
            y = y + dw(x)
        y = self.act(self.bn(y))
        y = self.pw(y)
        return x + self.gamma * (y - x)


class MobileNetV4(nn.Module):
    def __init__(self, model,use_p5_dwaspp: bool = False,
                 p5_dilations=(1,2,3), p5_gamma_init: float = 0.0):
        """
        精简为 backbone 用法：只构建到 layer4，删除 layer5（分类尾）。
        注意：forward 仅返回 (x1, x2, x3, x4)，索引分别对应 /4, /8, /16, /32。
        """
        super().__init__()
        assert model in MODEL_SPECS.keys(), f'Unknown model spec: {model}'
        self.model = model
        self.spec = MODEL_SPECS[self.model]
        self.use_p5_dwaspp = use_p5_dwaspp

        # 只构建到 layer4
        self.conv0  = build_blocks(self.spec['conv0'])
        self.layer1 = build_blocks(self.spec['layer1'])
        self.layer2 = build_blocks(self.spec['layer2'])
        self.layer3 = build_blocks(self.spec['layer3'])
        self.layer4 = build_blocks(self.spec['layer4'])
        # 不再构建 layer5（分类尾）
                # ===== 在这里添加：根据 layer4 的第一块推断 P5 通道，并按需实例化 DWASPP =====
        # 计算 P5 通道（layer4 第一块的 oup）
        if self.spec['layer4']['num_blocks'] > 0:
            first_l4 = self.spec['layer4']['block_specs'][0]
            p5_ch = int(first_l4[1])  # 各种 block 的第二个元素都是 oup
        else:
            # 兜底：若 layer4 为空，就用 layer3 最后一块的 oup（一般不会发生）
            last_l3 = self.spec['layer3']['block_specs'][-1]
            p5_ch = int(last_l3[1])

        if self.use_p5_dwaspp:
            self.p5_dwaspp = DWASPP(p5_ch, p5_ch,
                                    dilations=p5_dilations,
                                    gamma_init=p5_gamma_init)
        else:
            self.p5_dwaspp = None
        # 如果 MODEL_SPECS 里仍然包含 "layer5"，也会被忽略

    def forward(self, x):
        x0 = self.conv0(x)     # /2
        x1 = self.layer1(x0)   # /4
        x2 = self.layer2(x1)   # /8
        x3 = self.layer3(x2)   # /16
#         x4 = self.layer4(x3)   # /32
                # --- layer4 拆开 ---
        if len(self.layer4) == 0:
            x4 = x3
        else:
            # 先过第一个模块（通常 stride=2，下到 /32）
            x4 = self.layer4[0](x3)
            # 在 P5 分辨率的"开头"插入 DWASPP
            if self.p5_dwaspp is not None:
                x4 = self.p5_dwaspp(x4)
            # 再跑剩余的 layer4
            for m in list(self.layer4)[1:]:
                x4 = m(x4)

        return (x1, x2, x3, x4)
        # 不再计算 x5