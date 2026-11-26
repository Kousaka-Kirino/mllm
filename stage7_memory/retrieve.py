"""S7 · 检索与放大：从"整页"里定位一块，再放大喂模型。

两条策略，一条是学习的、一条是纯几何的：

- `visual_query`：用一块裁片当查询，在页面 patch 记忆里找最像的位置（需要 S7
  索引）；这是"用户指着一块问"的建模。
- `saliency`：不训练也能用的启发式——字多的地方（patch 方差大）最可能是重点。
  它是个**对照**：如果 saliency 和 visual_query 效果一样，说明"检索"这一步没
  有学到东西，只是找到了字。

放大方式统一：裁 bbox → 双三次放大到 384²（`crop_zoom`）。这一步是唯一能真正
提高模型侧字号的杠杆（见 S1 的 acuity 恒等式）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class Region:
    x0: int
    y0: int
    x1: int
    y1: int

    def size(self) -> tuple[int, int]:
        return self.x1 - self.x0, self.y1 - self.y0


def grid_bbox(i: int, side: int, size: int) -> tuple[int, int, int, int]:
    """第 i 个格子（行优先）在 size×size 图上的像素 bbox。"""
    cell = size / side
    r, c = divmod(i, side)
    return (int(c * cell), int(r * cell), int((c + 1) * cell), int((r + 1) * cell))


def region_from_grid(i: int, side: int, size: int, pad: float = 0.5) -> Region:
    x0, y0, x1, y1 = grid_bbox(i, side, size)
    w, h = x1 - x0, y1 - y0
    px, py = int(w * pad), int(h * pad)
    return Region(max(0, x0 - px), max(0, y0 - py),
                  min(size, x1 + px), min(size, y1 + py))


def saliency_scores(img: Image.Image, side: int = 16) -> np.ndarray:
    """每个格子的"字多不多"：灰度标准差。返回 side*side 的数组。"""
    g = np.asarray(img.convert("L").resize((side * 8, side * 8))).astype(np.float32)
    cell = 8
    out = np.zeros((side, side), dtype=np.float32)
    for r in range(side):
        for c in range(side):
            blk = g[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            out[r, c] = blk.std()
    return out


def best_salient(img: Image.Image, side: int = 16) -> Region:
    s = saliency_scores(img, side)
    return region_from_grid(int(np.argmax(s)), side, img.width)


def random_region(img: Image.Image, side: int = 16, seed: int = 0) -> Region:
    rng = np.random.default_rng(seed)
    return region_from_grid(int(rng.integers(0, side * side)), side, img.width)


def crop_zoom(img: Image.Image, region: Region, out: int = 384,
              mode: str = "bicubic") -> np.ndarray:
    """裁剪 + 放大到 out²，返回 CHW float32，取值 [-1,1]（与训练一致）。"""
    crop = img.crop((region.x0, region.y0, region.x1, region.y1))
    rs = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR,
          "nearest": Image.NEAREST}[mode]
    crop = crop.resize((out, out), rs)
    a = np.asarray(crop).astype(np.float32) / 255.0
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    return np.ascontiguousarray(a.transpose(2, 0, 1) * 2.0 - 1.0)


def full_image(img: Image.Image, out: int = 384) -> np.ndarray:
    a = np.asarray(img.convert("RGB").resize((out, out), Image.BICUBIC)).astype(np.float32)
    return np.ascontiguousarray((a / 255.0).transpose(2, 0, 1) * 2.0 - 1.0)


def score_region(feats: np.ndarray, region_idx: list[int]) -> float:
    """用一组格子的平均相似度给候选区域打分（feats 已 L2 归一化）。"""
    if not region_idx:
        return 0.0
    return float(np.mean(feats[region_idx]))
