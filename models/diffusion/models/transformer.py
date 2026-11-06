import functools
import math

import torch
import torch.nn.functional as F
from torch import nn

from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max
import torch.utils.checkpoint as activation_checkpoint


from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 基础模块：RMSNorm / SwiGLU / MLPBlock
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, use_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.use_scale = use_scale
        if use_scale:
            self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.use_scale:
            norm_x = norm_x * self.scale
        return norm_x


class SwiGLU(nn.Module):
    """
    y = (SiLU(xW) * xV) W2
    """
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None, bias: bool = True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 3
        self.w = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.v = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, embed_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xw = self.w(x)
        xv = self.v(x)
        gates = F.silu(xw) * xv
        return self.w2(gates)


class MLPBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        out_embed_dim: Optional[int] = None,
        activation: str = "silu",
        bias: bool = True,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 4
        if out_embed_dim is None:
            out_embed_dim = embed_dim

        act = {
            "silu": F.silu,
            "swish": F.silu,
            "relu": F.relu,
            "gelu": F.gelu,
        }.get(activation, F.silu)

        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, out_embed_dim, bias=bias)
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# ============================================================
# GraphTransformerLayer：带 λA bias、qk_norm、softcap、sm_scale
# ============================================================

class GraphTransformerLayer(nn.Module):
    """
    Lightweight graph Transformer layer with structural bias:

        Attention(Q,K,V) = softmax( softcap( sm_scale * QK^T + bias ) ) V

    其中 bias 一般取 λ A（邻接矩阵或其它 attention bias）。
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        softcap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        qk_norm: bool = False,
        use_swiglu: bool = True,
        ffn_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.dropout = nn.Dropout(dropout)

        # 如果没指定，就用标准 1/sqrt(d_k)
        self.sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(self.d_head)
        self.softcap = softcap

        # Q, K, V 投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # qk_norm：对每个 head 的特征做 RMSNorm（不带 scale）
        self.q_norm = RMSNorm(self.d_head, use_scale=False) if qk_norm else None
        self.k_norm = RMSNorm(self.d_head, use_scale=False) if qk_norm else None

        # 残差 + Norm + FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if use_swiglu:
            self.ffn = SwiGLU(d_model, hidden_dim=ffn_hidden_dim)
        else:
            self.ffn = MLPBlock(d_model, hidden_dim=ffn_hidden_dim, activation="silu")

    def _attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        q,k,v: (B, H, N, d_head)
        bias:  (B, 1, N, N) 或 (B, N, N) 或 None
        """
        B, H, N, _ = q.shape

        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.sm_scale  # (B,H,N,N)

        if bias is not None:
            if bias.dim() == 3:
                bias = bias.unsqueeze(1)  # (B,1,N,N)
            logits = logits + bias.to(logits.dtype)

        if self.softcap is not None:
            logits = self.softcap * torch.tanh(logits / self.softcap)

        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B,H,N,d_head)
        return out

    def forward(self, h: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        h:    (B, N, d_model)
        bias: (B, N, N) or (B,1,N,N) or None  —— 通常是 λ A
        """
        B, N, _ = h.shape

        # 自注意力块
        residual = h
        h_norm = self.norm1(h)

        q = self.W_q(h_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(h_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(h_norm).view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn_out = self._attn(q, k, v, bias)          # (B,H,N,d_head)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        attn_out = self.W_o(attn_out)
        h = residual + self.dropout(attn_out)

        # FFN 块
        residual = h
        h = self.norm2(h)
        h = self.ffn(h)
        h = residual + self.dropout(h)

        return h


# ============================================================
# Transformer 堆栈：encoder / decoder 共用
# ============================================================

class GraphTransformerStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        num_heads: int,
        softcap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        qk_norm: bool = False,
        use_swiglu: bool = True,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                d_model=embed_dim,
                n_heads=num_heads,
                dropout=dropout,
                softcap=softcap,
                sm_scale=sm_scale,
                qk_norm=qk_norm,
                use_swiglu=use_swiglu,
                ffn_hidden_dim=ffn_hidden_dim,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:    (B, N, D)
        bias: (B, N, N) or None
        """
        for layer in self.layers:
            x = layer(x, bias=bias)
        return x


# ============================================================
# TSPTransformerModel：encode / decode / feature2logit / forward
# ============================================================

class TSPTransformerModel(nn.Module):
    """
    PyTorch 版 TSP Transformer 模型，对应 JAX 版 TSPModel：

        features = model.encode(raw_features)
        logits   = model.decode(features, timesteps, adj)
        log_probs = F.log_softmax(logits, dim=-1)

    同时支持：
      - λ A 结构 bias
      - qk_norm、attention softcap、sm_scale
      - SwiGLU / MLP FFN
      - final_norm + final_softcap + final_logit_scale + 对角 mask
    """

    def __init__(
        self,
        num_layers: Tuple[int, int] = (16, 6),
        embed_dim: Union[Tuple[int, int], int] = 256,
        hidden_dim: Optional[int] = None,
        num_heads: Union[Tuple[int, int], int] = 8,
        activation: str = "silu",
        use_swiglu: bool = True,
        *,
        encoder_input_dim: int = 2,
        softcap: Optional[float] = 30.0,
        qk_norm: bool = False,
        sm_scale: Optional[float] = None,
        final_norm: bool = True,
        final_norm_use_scale: bool = False,
        final_softcap: Optional[float] = 50.0,
        final_logit_scale: float = 1.0,
        lambda_adj: float = 1.0,
    ):
        super().__init__()

        if isinstance(embed_dim, int):
            embed_dim = (embed_dim, embed_dim)
        if isinstance(num_heads, int):
            num_heads = (num_heads, num_heads)

        enc_layers, dec_layers = num_layers
        enc_dim, dec_dim = embed_dim
        enc_heads, dec_heads = num_heads

        self.encoder_embed_dim = enc_dim
        self.decoder_embed_dim = dec_dim
        self.final_softcap = final_softcap
        self.final_logit_scale = final_logit_scale
        self.lambda_adj = lambda_adj

        # ---------- encoder 输入投影 ----------
        # 如果 raw_features 不是 (B,N,2)，把 encoder_input_dim 改成实际维度即可
        self.init_proj = nn.Linear(encoder_input_dim, enc_dim, bias=True)

        # ---------- encoder stack ----------
        self.encoder = GraphTransformerStack(
            num_layers=enc_layers,
            embed_dim=enc_dim,
            num_heads=enc_heads,
            softcap=softcap,
            sm_scale=sm_scale,
            qk_norm=qk_norm,
            use_swiglu=use_swiglu,
            ffn_hidden_dim=hidden_dim,
        )

        # ---------- encoder -> decoder 维度对齐 ----------
        self.mid_proj = nn.Linear(enc_dim, dec_dim, bias=True)

        # ---------- decoder stack ----------
        self.decoder = GraphTransformerStack(
            num_layers=dec_layers,
            embed_dim=dec_dim,
            num_heads=dec_heads,
            softcap=softcap,
            sm_scale=sm_scale,
            qk_norm=qk_norm,
            use_swiglu=use_swiglu,
            ffn_hidden_dim=hidden_dim,
        )

        # ---------- final proj ----------
        if use_swiglu:
            self.final_proj = SwiGLU(dec_dim, hidden_dim=hidden_dim)
        else:
            self.final_proj = MLPBlock(dec_dim, hidden_dim=hidden_dim, activation=activation)

        self.final_norm = RMSNorm(dec_dim, use_scale=final_norm_use_scale) if final_norm else None
        self.apply_final_norm = final_norm

    # ---------------- encode / decode / feature2logit ---------------- #

    def encode(self, raw_features: torch.Tensor) -> torch.Tensor:
        """
        raw_features: (B, N, encoder_input_dim)
        返回:         (B, N, encoder_embed_dim)
        """
        h = self.init_proj(raw_features)
        h = self.encoder(h, bias=None)
        return h

    def decode(
        self,
        features: torch.Tensor,            # (B, N, encoder_embed_dim)
        adjmat: Optional[torch.Tensor],    # (B, N, N)
        target: str = "logit",
        force_fp32_logit: bool = True,
    ) -> torch.Tensor:
        """
        features: encoder 输出
        timesteps: diffusion 时间步
        adjmat: 邻接矩阵，作为 attention bias：λ A
        """
        # encoder -> decoder 维度对齐
        h = self.mid_proj(features)  # (B,N,dec_dim)

        # λ A 结构 bias
        if adjmat is not None:
            bias = self.lambda_adj * adjmat.to(h.dtype)  # (B,N,N)
        else:
            bias = None

        h = self.decoder(h, bias=bias)
        h = self.final_proj(h)

        if target == "embed":
            return h
        elif target == "logit":
            return self.feature2logit(h, force_fp32_logit=force_fp32_logit)
        else:
            raise ValueError(f"Unknown target: {target}")

    def feature2logit(
        self,
        features: torch.Tensor,             # (B, N, D)
        *,
        mask_value: Optional[Union[float, str]] = "default",
        force_fp32_logit: bool = True,
    ) -> torch.Tensor:
        """
        PyTorch 版 feature2logit：
          - final_norm
          - logits = h h^T
          - final_logit_scale
          - final_softcap
          - 对角线 mask
        返回: (B, N, N)
        """
        assert features.dim() == 3

        h = features
        if self.apply_final_norm and self.final_norm is not None:
            h = self.final_norm(h)

        logits = torch.matmul(h, h.transpose(-2, -1))  # (B,N,N)

        if force_fp32_logit:
            logits = logits.to(torch.float32)

        if self.final_logit_scale != 1.0:
            logits = logits * self.final_logit_scale

        if self.final_softcap is not None:
            logits = self.final_softcap * torch.tanh(logits / self.final_softcap)

        if mask_value is not None:
            B, N, _ = logits.shape
            if mask_value == "default":
                mask_value = 0.5 * torch.finfo(logits.dtype).min
            eye = torch.eye(N, device=logits.device, dtype=torch.bool).unsqueeze(0)  # (1,N,N)
            logits = logits.masked_fill(eye, mask_value)

        return logits

    # ---------------- 一步到位的 forward 接口 ---------------- #

    def forward(
        self,
        raw_features: torch.Tensor,          # (B, N, encoder_input_dim)
        adjmat: Optional[torch.Tensor],      # (B, N, N)
        edge_index=None,                     # 为兼容旧接口，保留不用
        return_logits: bool = True,
    ) -> torch.Tensor:
        """
        等价于：
            features = self.encode(raw_features)
            logits   = self.decode(features, timesteps, adjmat)
            log_probs = F.log_softmax(logits, dim=-1)

        默认返回 log_probs；若 return_logits=True，则返回 logits。
        """
        del edge_index

        features = self.encode(raw_features)
        logits = self.decode(features, adjmat)

        if return_logits:
            return logits

        log_probs = F.log_softmax(logits, dim=-1)  # 对最后一维做 log_softmax
        return log_probs