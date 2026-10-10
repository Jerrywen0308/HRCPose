import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import warnings
from typing import Optional, Sequence, Tuple, Union

import torch
from mmengine.dist import get_dist_info
from mmengine.structures import PixelData
from torch import Tensor, nn

from mmpose.codecs.utils import get_simcc_normalized
from mmpose.evaluation.functional import simcc_pck_accuracy
from mmpose.models.utils.rtmcc_block import RTMCCBlock, ScaleNorm
from mmpose.models.utils.tta import flip_vectors
from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.tensor_utils import to_numpy
from mmpose.utils.typing import (ConfigType, InstanceList, OptConfigType,
                                 OptSampleList)
from ..base_head import BaseHead



try:
    # 常见导入路径1（flash-attn 2.x 可能提供此接口）
    from flash_attn import flash_attn_func  # type: ignore
    _FLASH_ATTN_OK = True
except Exception:
    try:
        # 常见导入路径2（有些版本暴露的是 unpadded 接口，做一个兼容层）
        from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked as _flash_attn_unpadded  # type: ignore

        def flash_attn_func(q, k, v):
            # q/k/v: (B, N, H, D) half/fp16
            # 拼 packed，调用 unpadded。这里给出一个最简兜底；不同版本签名可能有差异，必要时自行适配。
            import torch
            B, N, H, D = q.shape
            qkv = torch.stack([q, k, v], dim=2)  # (B, N, 3, H, D)
            # 下行只是示意，若报错请按你安装的 flash-attn 版本改造
            return _flash_attn_unpadded(qkv.contiguous(), None, 0.0, causal=False)
        _FLASH_ATTN_OK = True
    except Exception:
        _FLASH_ATTN_OK = False

# 对外暴露一个开关常量，AAttn 会用到
USE_FLASH_ATTN = _FLASH_ATTN_OK

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))
    
    
class DSConv(nn.Module):
    """The Basic Depthwise Separable Convolution."""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, d=1, bias=False):
        super().__init__()
        if p is None:
            p = (d * (k - 1)) // 2
        self.dw = nn.Conv2d(
            c_in, c_in, kernel_size=k, stride=s,
            padding=p, dilation=d, groups=c_in, bias=bias
        )
        self.pw = nn.Conv2d(c_in, c_out, 1, 1, 0, bias=bias)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        return self.act(self.bn(x))

    
class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))    
    

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


# class C3k2(C2f):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
#         """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.m = nn.ModuleList(
#             C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
#         )
class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, k=3, dw=False):
        super().__init__(c1, c2, n, shortcut, g, e)
        # hidden 通道
        c_hidden = self.c  # = int(c2 * e)
        g_eff = c_hidden if (c3k and dw) else g  # DW: groups = 通道数

        self.m = nn.ModuleList(
            C3k(c_hidden, c_hidden, 2, shortcut, g_eff, k=k) if c3k
            else Bottleneck(c_hidden, c_hidden, shortcut, g)
            for _ in range(n)
        )

        

import torch
import torch.nn as nn
from torch import Tensor

def autopad(k: int, d: int = 1) -> int:
    """Same padding under dilation d (keeps H/W)."""
    return ((k - 1) // 2) * d


class AxialDW(nn.Module):
    def __init__(self, ch: int, k: int = 11, d: int = 1):
        super().__init__()
        assert k >= 1 and k % 2 == 1, f'k must be odd >=1, got {k}'
        assert d >= 1, f'd must be >=1, got {d}'
        self.h = nn.Conv2d(
            ch, ch, kernel_size=(k, 1),
            padding=(autopad(k, d), 0),
            dilation=(d, 1), groups=ch, bias=False
        )
        self.bnh = nn.BatchNorm2d(ch)

        self.w = nn.Conv2d(
            ch, ch, kernel_size=(1, k),
            padding=(0, autopad(k, d)),
            dilation=(1, d), groups=ch, bias=False
        )
        self.bnw = nn.BatchNorm2d(ch)

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.bnh(self.h(x)))
        x = self.act(self.bnw(self.w(x)))
        return x


class AxialDWResBlock(nn.Module):
    def __init__(self, ch: int, axial_k: int = 11, axial_d: int = 1):
        super().__init__()
        self.axial = AxialDW(ch, k=axial_k, d=axial_d)
        self.dw3 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),  # 保持尺寸
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True)
        )


    def forward(self, x: Tensor) -> Tensor:
        y = self.axial(x)
        y = self.dw3(y)
        return x + y



class CoordAttnRes(nn.Module):
    """Coordinate Attention + residual gate γ (init=0 for stability)."""
    def __init__(self, ch: int, reduction: int = 32, act: nn.Module = None):
        super().__init__()
        if act is None:
            act = nn.SiLU(inplace=True)
        mip = max(8, ch // reduction)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(ch, mip, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = act
        self.conv_h = nn.Conv2d(mip, ch, 1, bias=False)
        self.conv_w = nn.Conv2d(mip, ch, 1, bias=False)


    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        if torch.onnx.is_in_onnx_export():
            
            x_h = x.mean(dim=3, keepdim=True)                     # [B, C, H, 1]
            x_w = x.mean(dim=2, keepdim=True).permute(            # [B, C, W, 1]
                0, 1, 3, 2)
        else:
            x_h = self.pool_h(x)                                  # [B, C, H, 1]
            x_w = self.pool_w(x).permute(0, 1, 3, 2)              # [B, C, W, 1]
        
        

        y = torch.cat([x_h, x_w], dim=2)             # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(x_h))                    # [B, C, H, 1]
        a_w = torch.sigmoid(self.conv_w(x_w)).permute(0, 1, 3, 2)  # [B, C, 1, W]
        A = a_h * a_w

        return x + x * A









class FinalDWkCoordProj(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 axial_k: int = 11, axial_d: int = 1,
                 ca_reduction: int = 8):
        super().__init__()
        self.pre  = AxialDWResBlock(in_ch, axial_k=axial_k, axial_d=axial_d)
        # 注意：CoordAttnRes 请在你的代码中已定义
        self.ca   = CoordAttnRes(in_ch, reduction=ca_reduction)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pre(x)   # 残差块（保持尺寸）
        x = self.ca(x)    # 坐标注意力（外部定义）
        x = self.proj(x)  # 投影到 K
        return x

OptIntSeq = Optional[Sequence[int]]




@MODELS.register_module()
class RTMCC_CoordAttnHead(BaseHead):
    """Top-down head introduced in RTMPose (2023). The head is composed of a
    local-context conv head (C3k2 + CoordAttnRes + 1x1) to produce K-channel
    logits, then a per-joint MLP + GAU to generate 1D SimCC representation.
    """

    def __init__(
        self,
        in_channels: Union[int, Sequence[int]],
        out_channels: int,
        input_size: Tuple[int, int],
        in_featuremap_size: Tuple[int, int],
        simcc_split_ratio: float = 2.0,
        # 原参数保留（不再使用 final_layer_kernel_size，但保持兼容）
        final_layer_kernel_size: int = 1,
        gau_cfg: ConfigType = dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='ReLU',
            use_rel_bias=False,
            pos_enc=False),
        loss: ConfigType = dict(type='KLDiscretLoss', use_target_weight=True),
        decoder: OptConfigType = None,
        init_cfg: OptConfigType = None,
        # 新增：final_layer 配置（可选）
        final_layer_cfg: OptConfigType = None,
        axial_d: int = 1,
    ):

        if init_cfg is None:
            init_cfg = self.default_init_cfg
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.in_featuremap_size = in_featuremap_size
        self.simcc_split_ratio = simcc_split_ratio

        self.loss_module = MODELS.build(loss)
        if decoder is not None:
            self.decoder = KEYPOINT_CODECS.build(decoder)
        else:
            self.decoder = None

        if isinstance(in_channels, (tuple, list)):
            raise ValueError(
                f'{self.__class__.__name__} does not support selecting '
                'multiple input features.')

        # ===== SimCC pathway settings =====
        flatten_dims = self.in_featuremap_size[0] * self.in_featuremap_size[1]

        # ===== New final_layer: C3k2 -> CoordAttnRes -> 1x1 -> [B,K,H,W] =====
        if final_layer_cfg is None:
            final_layer_cfg = dict(axial_k=11, axial_d=axial_d, ca_reduction=8)
        else:
            final_layer_cfg = dict(final_layer_cfg)  # 复制一份避免原地修改
            if 'axial_k' not in final_layer_cfg and 'k' in final_layer_cfg:
                final_layer_cfg['axial_k'] = final_layer_cfg.pop('k')
            # 若没在 cfg 里写 axial_d，就用 __init__ 形参 axial_d
            final_layer_cfg['axial_d'] = final_layer_cfg.get('axial_d', axial_d)
            # 只保留需要的键
            final_layer_cfg = {
                'axial_k': final_layer_cfg.get('axial_k', 11),
                'axial_d': final_layer_cfg.get('axial_d', axial_d),
                'ca_reduction': final_layer_cfg.get('ca_reduction', 8),
            }

        self.final_layer = FinalDWkCoordProj(
            in_ch=self.in_channels,
            out_ch=self.out_channels,
            axial_k=final_layer_cfg['axial_k'],
            axial_d=final_layer_cfg['axial_d'],
            ca_reduction=final_layer_cfg['ca_reduction'],
        )

        self.final_layer = FinalDWkCoordProj(
            in_ch=self.in_channels,
            out_ch=self.out_channels,
            **final_layer_cfg
        )

        # Per-joint tokenization (flatten H*W -> hidden)
        self.mlp = nn.Sequential(
            ScaleNorm(flatten_dims),
            nn.Linear(flatten_dims, gau_cfg['hidden_dims'], bias=False))

        # GAU over joint dimension (self-attn among K joints)
        W = int(self.input_size[0] * self.simcc_split_ratio)
        H = int(self.input_size[1] * self.simcc_split_ratio)

        self.gau = RTMCCBlock(
            self.out_channels,
            gau_cfg['hidden_dims'],
            gau_cfg['hidden_dims'],
            s=gau_cfg['s'],
            expansion_factor=gau_cfg['expansion_factor'],
            dropout_rate=gau_cfg['dropout_rate'],
            drop_path=gau_cfg['drop_path'],
            attn_type='self-attn',
            act_fn=gau_cfg['act_fn'],
            use_rel_bias=gau_cfg['use_rel_bias'],
            pos_enc=gau_cfg['pos_enc'])

        self.cls_x = nn.Linear(gau_cfg['hidden_dims'], W, bias=False)
        self.cls_y = nn.Linear(gau_cfg['hidden_dims'], H, bias=False)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the network to produce SimCC x/y logits."""
        feats = feats[-1]                       # use last stage feature (P3)
        feats = self.final_layer(feats)         # -> [B, K, H, W]

        # flatten per joint map
        feats = torch.flatten(feats, 2)         # -> [B, K, H*W]
        feats = self.mlp(feats)                 # -> [B, K, hidden]
        feats = self.gau(feats)                 # -> [B, K, hidden]

        pred_x = self.cls_x(feats)              # -> [B, K, Wx]
        pred_y = self.cls_y(feats)              # -> [B, K, Wy]
        return pred_x, pred_y

    def predict(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        test_cfg: OptConfigType = {},
    ) -> InstanceList:
        """Predict results from features."""
        if test_cfg.get('flip_test', False):
            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            _feats, _feats_flip = feats

            _px, _py = self.forward(_feats)
            _pxf, _pyf = self.forward(_feats_flip)
            _pxf, _pyf = flip_vectors(_pxf, _pyf, flip_indices=flip_indices)

            batch_pred_x = (_px + _pxf) * 0.5
            batch_pred_y = (_py + _pyf) * 0.5
        else:
            batch_pred_x, batch_pred_y = self.forward(feats)

        preds = self.decode((batch_pred_x, batch_pred_y))

        if test_cfg.get('output_heatmaps', False):
            rank, _ = get_dist_info()
            if rank == 0:
                warnings.warn('The predicted simcc values are normalized for '
                              'visualization. This may cause discrepancy '
                              'between the keypoint scores and the 1D heatmaps'
                              '.')

            # normalize the predicted 1d distribution
            batch_pred_x = get_simcc_normalized(batch_pred_x)
            batch_pred_y = get_simcc_normalized(batch_pred_y)

            B, K, _ = batch_pred_x.shape
            # B, K, Wx -> B, K, Wx, 1
            x = batch_pred_x.reshape(B, K, 1, -1)
            # B, K, Wy -> B, K, 1, Wy
            y = batch_pred_y.reshape(B, K, -1, 1)
            # B, K, Wx, Wy
            batch_heatmaps = torch.matmul(y, x)
            pred_fields = [
                PixelData(heatmaps=hm) for hm in batch_heatmaps.detach()
            ]

            for pred_instances, pred_x, pred_y in zip(
                    preds, to_numpy(batch_pred_x), to_numpy(batch_pred_y)):
                pred_instances.keypoint_x_labels = pred_x[None]
                pred_instances.keypoint_y_labels = pred_y[None]

            return preds, pred_fields
        else:
            return preds

    def loss(
        self,
        feats: Tuple[Tensor],
        batch_data_samples: OptSampleList,
        train_cfg: OptConfigType = {},
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        pred_x, pred_y = self.forward(feats)

        gt_x = torch.cat(
            [d.gt_instance_labels.keypoint_x_labels for d in batch_data_samples],
            dim=0)
        gt_y = torch.cat(
            [d.gt_instance_labels.keypoint_y_labels for d in batch_data_samples],
            dim=0)
        keypoint_weights = torch.cat(
            [d.gt_instance_labels.keypoint_weights for d in batch_data_samples],
            dim=0)

        pred_simcc = (pred_x, pred_y)
        gt_simcc = (gt_x, gt_y)

        losses = dict()
        loss = self.loss_module(pred_simcc, gt_simcc, keypoint_weights)
        losses.update(loss_kpt=loss)

        # accuracy
        _, avg_acc, _ = simcc_pck_accuracy(
            output=to_numpy(pred_simcc),
            target=to_numpy(gt_simcc),
            simcc_split_ratio=self.simcc_split_ratio,
            mask=to_numpy(keypoint_weights) > 0,
        )
        acc_pose = torch.tensor(avg_acc, device=gt_x.device)
        losses.update(acc_pose=acc_pose)
        return losses

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(type='Normal', layer=['Conv2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1),
            dict(type='Normal', layer=['Linear'], std=0.01, bias=0),
        ]
        return init_cfg