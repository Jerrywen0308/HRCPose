# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from mmengine.utils import digit_version
from mmengine.utils.dl_utils import TORCH_VERSION

# 依赖你现有工程里的实现
from .transformer import ScaleNorm

def rope(x, dim):
    """Rotary Position Embedding."""
    shape = x.shape
    if isinstance(dim, int):
        dim = [dim]
    spatial_shape = [shape[i] for i in dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
    position = torch.reshape(
        torch.arange(total_len, dtype=torch.int, device=x.device),
        spatial_shape)
    for i in range(dim[-1] + 1, len(shape) - 1, 1):
        position = torch.unsqueeze(position, dim=-1)
    half_size = shape[-1] // 2
    freq_seq = -torch.arange(
        half_size, dtype=torch.int, device=x.device) / float(half_size)
    inv_freq = 10000**-freq_seq
    sinusoid = position[..., None] * inv_freq[None, None, :]
    sin = torch.sin(sinusoid)
    cos = torch.cos(sinusoid)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Scale(nn.Module):
    """Element-wise learnable scale."""
    def __init__(self, dim, init_value=1., trainable=True):
        super().__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale





class RTMCC_GroupedGAU_Block(nn.Module):
    """
    Gated Attention Unit（GAU）的分组版本（类似多头），在每组内独立计算 qk、门控和聚合，然后在通道维 concat。

    - self-attn: 输入为 x:[B,K,C]；生成 u,v:[B,K,e] 和 q,k:[B,K,s]，按 groups 切分后在每组内计算。
    - cross-attn: 输入为 (x, k_in, v_in)，k/v 先线性到 s/e 再分组。

    Args:
        num_token (int): K
        in_token_dims (int): C
        out_token_dims (int): C_out
        expansion_factor (int): e = C * expansion_factor
        s (int): 注意力瓶颈维
        groups (int): 分组数（必须整除 e 和 s）
        eps (float): ScaleNorm eps
        dropout_rate (float)
        drop_path (float)
        attn_type (str): 'self-attn' or 'cross-attn'
        act_fn (str): 'SiLU' or 'ReLU'
        bias (bool)
        use_rel_bias (bool): self-attn 时启用相对位置偏置（共享到各组）
        pos_enc (bool): 是否对 q/k 使用 RoPE（self-/cross- 都可）
        rel_bias_mode (str): 'toeplitz' or 'none'（简化实现，默认 toeplitz）
    """

    def __init__(self,
                 num_token: int,
                 in_token_dims: int,
                 out_token_dims: int,
                 expansion_factor: int = 2,
                 s: int = 128,
                 groups: int = 2,
                 eps: float = 1e-5,
                 dropout_rate: float = 0.,
                 drop_path: float = 0.,
                 attn_type: str = 'self-attn',
                 act_fn: str = 'SiLU',
                 bias: bool = False,
                 use_rel_bias: bool = True,
                 pos_enc: bool = False,
                 rel_bias_mode: str = 'toeplitz'):

        super().__init__()
        assert groups >= 1
        self.s = s
        self.groups = groups
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.attn_type = attn_type
        self.pos_enc = pos_enc
        self.rel_bias_mode = rel_bias_mode

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.e = int(in_token_dims * expansion_factor)
        assert self.e % groups == 0, f"e={self.e} must be divisible by groups={groups}"
        assert self.s % groups == 0, f"s={self.s} must be divisible by groups={groups}"
        self.e_g = self.e // groups
        self.s_g = self.s // groups

        # 相对位置偏置（共享到各组）
        if use_rel_bias and attn_type == 'self-attn' and rel_bias_mode == 'toeplitz':
            self.w = nn.Parameter(torch.zeros(2 * num_token - 1, dtype=torch.float32))
        else:
            self.w = None  # 'none' 或 cross-attn

        self.o = nn.Linear(self.e, out_token_dims, bias=bias)

        if attn_type == 'self-attn':
            self.uv = nn.Linear(in_token_dims, 2 * self.e + self.s, bias=bias)
            self.gamma = nn.Parameter(torch.zeros((2, self.s), dtype=torch.float32))
            self.beta = nn.Parameter(torch.zeros((2, self.s), dtype=torch.float32))
        else:
            self.uv = nn.Linear(in_token_dims, self.e + self.s, bias=bias)
            self.k_fc = nn.Linear(in_token_dims, self.s, bias=bias)
            self.v_fc = nn.Linear(in_token_dims, self.e, bias=bias)
            nn.init.xavier_uniform_(self.k_fc.weight)
            nn.init.xavier_uniform_(self.v_fc.weight)

        self.ln = ScaleNorm(in_token_dims, eps=eps)
        nn.init.xavier_uniform_(self.uv.weight)

        if act_fn == 'SiLU' or act_fn == nn.SiLU:
            assert digit_version(TORCH_VERSION) >= digit_version('1.7.0'), \
                'SiLU activation requires PyTorch >= 1.7'
            self.act_fn = nn.SiLU(True)
        elif act_fn == 'ReLU' or act_fn == nn.ReLU:
            self.act_fn = nn.ReLU(True)
        else:
            raise NotImplementedError

        if in_token_dims == out_token_dims:
            self.shortcut = True
            self.res_scale = Scale(in_token_dims)
        else:
            self.shortcut = False

        self.sqrt_sg = math.sqrt(self.s_g)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.:
            self.dropout = nn.Dropout(dropout_rate)

    def _rel_pos_bias(self, q_len: int):
        """Toeplitz 相对位置偏置，形状 [1, K, K]，各组共享。"""
        if self.w is None:
            return None
        t = F.pad(self.w[:2 * q_len - 1], [0, q_len]).repeat(q_len)
        t = t[..., :-q_len].reshape(-1, q_len, 3 * q_len - 2)
        r = (2 * q_len - 1) // 2
        t = t[..., r:-r]  # [K, K]
        return t[None, ...]  # [1, K, K]

    def _forward(self, inputs):
        if self.attn_type == 'self-attn':
            x = inputs                              # [B, K, C]
        else:
            x, k_in, v_in = inputs                  # [B, K, C], [B, K, C], [B, K, C]

        x = self.ln(x)
        uv = self.uv(x)                             # -> [B, K, 2e+s] or [B, K, e+s]
        uv = self.act_fn(uv)

        if self.attn_type == 'self-attn':
            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)   # [B,K,e],[B,K,e],[B,K,s]
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta  # [B,K,2,s]
            if self.pos_enc:
                base = rope(base, dim=1)
            q, k = torch.unbind(base, dim=2)        # [B,K,s], [B,K,s]
        else:
            u, q = torch.split(uv, [self.e, self.s], dim=2)  # [B,K,e], [B,K,s]
            k = self.k_fc(k_in)                      # [B,K,s]
            v = self.v_fc(v_in)                      # [B,K,e]
            if self.pos_enc:
                q = rope(q, 1)
                k = rope(k, 1)

        # 分组 reshape
        B, K, _ = q.shape
        u = u.view(B, K, self.groups, self.e_g)      # [B,K,G,e_g]
        v = v.view(B, K, self.groups, self.e_g)      # [B,K,G,e_g]
        q = q.view(B, K, self.groups, self.s_g)      # [B,K,G,s_g]
        k = k.view(B, K, self.groups, self.s_g)      # [B,K,G,s_g]

        # 合并 batch 和 group 维做高效 bmm
        q = q.permute(0, 2, 1, 3).reshape(B * self.groups, K, self.s_g)      # [B*G, K, s_g]
        k = k.permute(0, 2, 1, 3).reshape(B * self.groups, K, self.s_g)      # [B*G, K, s_g]
        v = v.permute(0, 2, 1, 3).reshape(B * self.groups, K, self.e_g)      # [B*G, K, e_g]
        u = u.permute(0, 2, 1, 3).reshape(B * self.groups, K, self.e_g)      # [B*G, K, e_g]

#         qk = torch.bmm(q, k.permute(0, 2, 1))
        qk = torch.matmul(q, k.transpose(1, 2))
        # [B*G, K, K]
        if self.use_rel_bias and self.attn_type == 'self-attn':
            bias = self._rel_pos_bias(K)                                       # [1,K,K] or None
            if bias is not None:
                qk = qk + bias  # 广播到 [B*G,K,K]

        kernel = torch.square(F.relu(qk / self.sqrt_sg))                        # [B*G, K, K]
        if self.dropout_rate > 0.:
            kernel = self.dropout(kernel)

#         out = u * torch.bmm(kernel, v)
        out = u * torch.matmul(kernel, v)# [B*G, K, e_g]
#         out = out.reshape(B, self.groups, K, self.e_g).permute(0, 2, 1, 2)
        out = out.reshape(B, self.groups, K, self.e_g).permute(0, 2, 1, 3)  # 0,2,1,3
# [B,K,G,e_g]
        out = out.reshape(B, K, self.e)                                         # [B,K,e]
        out = self.o(out)                                                       # [B,K,out_token_dims]
        return out

    def forward(self, x):
        if self.shortcut:
            res_shortcut = x[0] if self.attn_type == 'cross-attn' else x
            main = self.drop_path(self._forward(x))
            return self.res_scale(res_shortcut) + main
        else:
            return self.drop_path(self._forward(x))
