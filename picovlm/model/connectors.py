"""视觉-语言连接器：把 ViT 的 patch 序列压成 LLM 能吃的视觉 token。

三种实现对应 S6 的三个 token 预算档：

| 连接器 | 输出 token | 思想 | 风险 |
|---|---|---|---|
| mlp          | N=576 | LLaVA-1.0 式线性投影，保全部细节 | 最贵 |
| resampler    | Q=64  | Perceiver 式 query 压缩（Qwen2-VL/OpenFlamingo 同源） | 压缩有损 |
| pixelshuffle | N/4=144 | 空间相邻 patch 先合并再投影（pixel-unshuffle） | 细粒度定位受损 |

`forward` 一律吃 (B, N, D) 的 patch 特征，返回 (B, N', D_out)。
"""

from __future__ import annotations

import math

import torch
from torch import nn

CONNECTORS = ("mlp", "resampler", "pixelshuffle")


class MLPConnector(nn.Module):
    """2 层 MLP + GELU。逐 token 独立，不混合空间信息。"""

    def __init__(self, d_in: int, d_out: int, hidden: int | None = None,
                 depth: int = 2) -> None:
        super().__init__()
        hidden = hidden or d_out
        layers: list[nn.Module] = []
        last = d_in
        for _ in range(depth):
            layers += [nn.Linear(last, hidden), nn.GELU()]
            last = hidden
        layers.append(nn.Linear(last, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _CrossBlock(nn.Module):
    def __init__(self, d: int, heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q2, _ = self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
                          need_weights=False)
        q = q + q2
        return q + self.ff(self.norm_ff(q))


class ResamplerConnector(nn.Module):
    """固定数量的 learnable query 从 patch 序列里"读"信息。"""

    def __init__(self, d_in: int, d_out: int, n_queries: int = 64,
                 heads: int = 8, depth: int = 2) -> None:
        super().__init__()
        self.n_queries = n_queries
        self.in_proj = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.queries = nn.Parameter(torch.randn(n_queries, d_out) * 0.02)
        self.blocks = nn.ModuleList([_CrossBlock(d_out, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kv = self.in_proj(x)
        q = self.queries.unsqueeze(0).expand(kv.size(0), -1, -1)
        for blk in self.blocks:
            q = blk(q, kv)
        return self.norm(q)


class PixelShuffleConnector(nn.Module):
    """f×f 相邻 patch 先拼成一个 token，再线性投影。"""

    def __init__(self, d_in: int, d_out: int, factor: int = 2) -> None:
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(d_in * factor * factor, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        f = self.factor
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError(f"patch 数 {n} 不是完全平方，无法像素重排")
        if side % f:
            raise ValueError(f"网格边长 {side} 不能被 {f} 整除")
        x = x.view(b, side // f, f, side // f, f, d)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, (side // f) ** 2, f * f * d)
        return self.proj(x)


def build_connector(name: str, d_in: int, d_out: int, **kw) -> nn.Module:
    if name == "mlp":
        return MLPConnector(d_in, d_out, **{k: v for k, v in kw.items()
                                            if k in ("hidden", "depth")})
    if name == "resampler":
        return ResamplerConnector(d_in, d_out, **{k: v for k, v in kw.items()
                                                  if k in ("n_queries", "heads", "depth")})
    if name == "pixelshuffle":
        return PixelShuffleConnector(d_in, d_out, **{k: v for k, v in kw.items()
                                                     if k in ("factor",)})
    raise KeyError(f"未知连接器 {name!r}；可选 {CONNECTORS}")
