"""冻结的视觉塔。默认 SigLIP-base-patch16-384（93M，576 patch）。

为什么冻它：
- 8GB 显存下，解冻 ViT 会让激活暴涨，而且小数据上解冻只会让它忘掉预训练；
- 冻结还有一个科研上的好处：**视觉侧的能力是常数**，acuity 的差异只能来自
  输入分辨率/裁剪与 LLM 侧的使用方式，因果链更干净。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn


class FrozenVisionTower(nn.Module):
    def __init__(self, name: str = "google/siglip-base-patch16-384",
                 image_size: int = 384, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        from transformers import SiglipVisionModel

        self.name = name
        self.image_size = image_size
        self.backbone = SiglipVisionModel.from_pretrained(name, torch_dtype=dtype)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.hidden_size = int(self.backbone.config.hidden_size)
        self.n_tokens = int((image_size // self.backbone.config.patch_size) ** 2)
        self.register_buffer("mean", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

    @torch.no_grad()
    def preprocess(self, images: Sequence[np.ndarray] | torch.Tensor) -> torch.Tensor:
        """uint8 HWC 列表或 (B,3,H,W) 张量 → 归一化到 [-1,1]。"""
        if isinstance(images, torch.Tensor):
            x = images.float()
            if x.max() > 2:
                x = x / 255.0
        else:
            arr = np.stack([np.asarray(im.convert("RGB").resize(
                (self.image_size, self.image_size), 3)) for im in images])
            x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        x = (x - self.mean) / self.std
        return x.to(next(self.backbone.parameters()).device)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=pixel_values)
        return out.last_hidden_state
