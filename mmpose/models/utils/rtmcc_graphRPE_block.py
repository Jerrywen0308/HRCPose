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


class RTMCC_GraphRPE_Block(nn.Module):
    """
    Gated Attention Unit (GAU) with Graph-RPE for keypoint tokens.

    与原 RTMCCBlock 的差异：
      - self-attn 下用 learnable 的 [K,K] 相对偏置矩阵 w_full 代替 1D Toeplitz。
      - 新增稳定化选项：α 缩放、去中心、对角置零、对称化。
      - 其它逻辑（门控/激活/残差/DropPath）保持一致；cross-attn 分支不变。

    Args:
        num_token (int): K，token 数（关键点数量）
        in_token_dims (int): 输入 token 维度
        out_token_dims (int): 输出 token 维度
        expansion_factor (int): e 的扩张倍数（默认 2）
        s (int): 注意力瓶颈维（默认 128）
        eps (float): ScaleNorm eps
        dropout_rate (float): 注意力核 dropout
        drop_path (float): DropPath 概率
        attn_type (str): 'self-attn' 或 'cross-attn'
        act_fn (str): 'SiLU' 或 'ReLU'
        bias (bool): 线性层是否带 bias
        use_rel_bias (bool): 是否使用相对偏置（Graph-RPE/Toeplitz/RoPE外积）
        pos_enc (bool): 是否对 q/k 使用 RoPE
        graph_rpe (bool): 是否在 self-attn 下启用 Graph-RPE
        graph_rpe_init (Tensor|None): [K,K] 的初始化矩阵
        freeze_graph_rpe (bool): 是否冻结 Graph-RPE 参数
        graph_rpe_scale (float): α 缩放系数，默认 0.1
        graph_rpe_centering (bool): 去中心，默认 True
        graph_rpe_zero_diag (bool): 对角置零，默认 True
        graph_rpe_symmetrize (bool): 对称化，默认 True
    """

    def __init__(self,
                 num_token: int,
                 in_token_dims: int,
                 out_token_dims: int,
                 expansion_factor: int = 2,
                 s: int = 128,
                 eps: float = 1e-5,
                 dropout_rate: float = 0.,
                 drop_path: float = 0.,
                 attn_type: str = 'self-attn',
                 act_fn: str = 'SiLU',
                 bias: bool = False,
                 use_rel_bias: bool = True,
                 pos_enc: bool = False,
                 # --- Graph-RPE ---
                 graph_rpe: bool = True,
                 graph_rpe_init: torch.Tensor = None,
                 freeze_graph_rpe: bool = False,
                 # --- 稳定化选项 ---
                 graph_rpe_scale: float = 0.1,
                 graph_rpe_centering: bool = True,
                 graph_rpe_zero_diag: bool = True,
                 graph_rpe_symmetrize: bool = True):

        super().__init__()
        self.s = s
        self.num_token = num_token
        self.use_rel_bias = use_rel_bias
        self.attn_type = attn_type
        self.pos_enc = pos_enc
        self.graph_rpe = bool(graph_rpe and attn_type == 'self-attn')

        # 稳定化选项
        self.graph_rpe_scale = float(graph_rpe_scale)
        self.graph_rpe_centering = bool(graph_rpe_centering)
        self.graph_rpe_zero_diag = bool(graph_rpe_zero_diag)
        self.graph_rpe_symmetrize = bool(graph_rpe_symmetrize)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.e = int(in_token_dims * expansion_factor)

        # 相对位置偏置参数
        if use_rel_bias:
            if attn_type == 'self-attn':
                if self.graph_rpe:
                    # Graph-RPE: learnable full bias [K, K]
                    self.w_full = nn.Parameter(
                        torch.zeros(num_token, num_token, dtype=torch.float32),
                        requires_grad=not freeze_graph_rpe)
                    if graph_rpe_init is not None:
                        with torch.no_grad():
                            init_mat = graph_rpe_init.float()
                            assert init_mat.shape[:2] == (num_token, num_token), \
                                f'graph_rpe_init must be [{num_token},{num_token}], got {tuple(init_mat.shape)}'
                            self.w_full.copy_(init_mat)
                else:
                    # 退回原 1D Toeplitz 偏置
                    self.w = nn.Parameter(
                        torch.zeros(2 * num_token - 1, dtype=torch.float32),
                        requires_grad=True)
            else:
                # cross-attn 使用 RoPE 外积
                self.a = nn.Parameter(torch.zeros(1, s, dtype=torch.float32))
                self.b = nn.Parameter(torch.zeros(1, s, dtype=torch.float32))

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
                'SiLU activation requires PyTorch version >= 1.7'
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

        self.sqrt_s = math.sqrt(s)
        self.dropout_rate = dropout_rate
        if dropout_rate > 0.:
            self.dropout = nn.Dropout(dropout_rate)

    # ---- 相对位置偏置（Graph-RPE / Toeplitz / Cross）----
    def rel_pos_bias(self, q_len: int, k_len: int = None):
        if self.attn_type == 'self-attn':
            if self.graph_rpe:
                # 取子块（一般 q_len=k_len=K）
                k_len = k_len if k_len is not None else q_len
                bias = self.w_full[:q_len, :k_len]

                # 稳定化处理（仅在方阵情况下做对称化/去对角）
                if self.graph_rpe_symmetrize and q_len == k_len:
                    bias = 0.5 * (bias + bias.T)
                if self.graph_rpe_zero_diag and q_len == k_len:
                    bias = bias - torch.diag(torch.diag(bias))
                if self.graph_rpe_centering:
                    bias = bias - bias.mean()

                # α 缩放后返回 [1, q_len, k_len]
                return (self.graph_rpe_scale * bias)[None, ...]
            else:
                # 原 Toeplitz 偏置
                t = F.pad(self.w[:2 * q_len - 1], [0, q_len]).repeat(q_len)
                t = t[..., :-q_len].reshape(-1, q_len, 3 * q_len - 2)
                r = (2 * q_len - 1) // 2
                t = t[..., r:-r]
                return t[None, ...]
        else:
            # cross-attn：RoPE 后外积
            a = rope(self.a.repeat(q_len, 1), dim=0)
            b = rope(self.b.repeat(k_len, 1), dim=0)
            return torch.bmm(a, b.permute(0, 2, 1))  # [1, q_len, k_len]

    # ---- 主前向（与原版一致）----
    def _forward(self, inputs):
        if self.attn_type == 'self-attn':
            x = inputs
        else:
            x, k, v = inputs

        x = self.ln(x)

        # [B, K, in_token_dims] -> [B, K, e + e + s] 或 [B, K, e + s]
        uv = self.uv(x)
        uv = self.act_fn(uv)

        if self.attn_type == 'self-attn':
            # u, v, base
            u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
            base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
            if self.pos_enc:
                base = rope(base, dim=1)
            # [B, K, 2, s] -> q, k: [B, K, s]
            q, k = torch.unbind(base, dim=2)
        else:
            u, q = torch.split(uv, [self.e, self.s], dim=2)
            k = self.k_fc(k)  # -> [B, K, s]
            v = self.v_fc(v)  # -> [B, K, e]
            if self.pos_enc:
                q = rope(q, 1)
                k = rope(k, 1)

        # 注意力核 (非 softmax)：ReLU^2( qk / sqrt(s) )
        qk = torch.bmm(q, k.permute(0, 2, 1))  # [B, K, K]

        if self.use_rel_bias:
            bias = (self.rel_pos_bias(q.size(1), k.size(1))
                    if self.attn_type != 'self-attn'
                    else self.rel_pos_bias(q.size(1)))
            qk += bias[:, :q.size(1), :k.size(1)]

        kernel = torch.square(F.relu(qk / self.sqrt_s))
        if self.dropout_rate > 0.:
            kernel = self.dropout(kernel)

        # 门控聚合
        x = u * torch.bmm(kernel, v)           # [B, K, e]
        x = self.o(x)                          # [B, K, out_token_dims]
        return x

    def forward(self, x):
        if self.shortcut:
            if self.attn_type == 'cross-attn':
                res_shortcut = x[0]
            else:
                res_shortcut = x
            main_branch = self.drop_path(self._forward(x))
            return self.res_scale(res_shortcut) + main_branch
        else:
            return self.drop_path(self._forward(x))