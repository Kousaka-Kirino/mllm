"""PicoVLM 组装：视觉塔 → 连接器 → Qwen2.5-0.5B。

布局用 **prefix**（LLaVA-1.0 式）：视觉 token 全部放在文本前面。

    [v1 … vNv] [t1 … tL]

为什么不用 in-place 占位符：Qwen 的 chat template 会插入特殊 token，
占位符替换要处理 tokenizer 的 special token 边界，容易出错；prefix 布局
没有这个问题，且对"图像是上下文"的语义更直白。代价是不能交错图文。

训练时 labels 的构造是**VLM 版 GRPO/SFT 最容易错的地方**：
- 视觉 token 位置：-100
- prompt 文本位置：-100
- 只有答案 token 参与 loss
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from picovlm.config import ModelConfig

from .connectors import build_connector


def assemble_inputs(vis: torch.Tensor, txt: torch.Tensor,
                    text_mask: torch.Tensor,
                    labels: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """把视觉/文本 embedding 拼起来，并同步 attention_mask 与 labels。

    纯函数，不依赖任何权重——所以 verify_stage2 在 CPU 上就能测它的正确性。
    """
    if vis.dim() != 3 or txt.dim() != 3:
        raise ValueError("vis/txt 都必须是 (B, N, D)")
    if vis.size(0) != txt.size(0) or vis.size(2) != txt.size(2):
        raise ValueError("batch 与隐藏维度必须一致")
    b, nv, _ = vis.shape
    inputs_embeds = torch.cat([vis, txt], dim=1)
    vis_mask = torch.ones(b, nv, dtype=text_mask.dtype, device=text_mask.device)
    attention_mask = torch.cat([vis_mask, text_mask], dim=1)
    if labels is None:
        return inputs_embeds, attention_mask, None
    vis_labels = torch.full((b, nv), -100, dtype=labels.dtype, device=labels.device)
    labels = torch.cat([vis_labels, labels], dim=1)
    labels = labels.masked_fill(attention_mask == 0, -100)
    return inputs_embeds, attention_mask, labels


@dataclass
class Batch:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None = None


class PicoVLM(nn.Module):
    """骨架版：默认**不加载权重**也能实例化（`load_weights=False`），便于形状测试。"""

    def __init__(self, cfg: ModelConfig | None = None, *,
                 load_weights: bool = False, d_vision: int = 768,
                 d_llm: int = 896, vocab: int = 151936) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.load_weights = load_weights
        d_llm = d_llm if not load_weights else 0

        if load_weights:
            from transformers import AutoModelForCausalLM

            from .vision import FrozenVisionTower

            self.vision: nn.Module = FrozenVisionTower(self.cfg.vision,
                                                       self.cfg.input_size)
            d_vision = self.vision.hidden_size
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.cfg.llm, torch_dtype=torch.bfloat16)
            d_llm = int(self.llm.config.hidden_size)
        else:
            self.vision = _DummyVision(d_vision, self.cfg.n_patches)
            self.llm = _DummyLLM(d_llm, vocab)

        self.connector = build_connector(
            self.cfg.connector, d_vision, d_llm,
            n_queries=self.cfg.n_queries, factor=self.cfg.shuffle_factor)
        self.n_visual_tokens = {"mlp": self.cfg.n_patches,
                                "resampler": self.cfg.n_queries,
                                "pixelshuffle": self.cfg.n_patches // (
                                    self.cfg.shuffle_factor ** 2)}[self.cfg.connector]

    # ------------------------------------------------------------------ 视觉
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.vision(pixel_values)          # (B, N, Dv)
        return self.connector(feats)               # (B, Nv, Dllm)

    # ------------------------------------------------------------------ 前向
    def forward(self, batch: Batch) -> dict:
        vis = self.encode_images(batch.pixel_values)
        txt = self.llm.get_input_embeddings()(batch.input_ids)
        embeds, mask, labels = assemble_inputs(vis, txt, batch.attention_mask,
                                               batch.labels)
        out = self.llm(inputs_embeds=embeds, attention_mask=mask, labels=labels,
                       return_dict=True)
        return {"loss": out.loss, "logits": out.logits}

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor, max_new_tokens: int = 32,
                 **kw) -> torch.Tensor:
        vis = self.encode_images(pixel_values)
        txt = self.llm.get_input_embeddings()(input_ids)
        embeds, mask, _ = assemble_inputs(vis, txt, attention_mask)
        return self.llm.generate(inputs_embeds=embeds, attention_mask=mask,
                                 max_new_tokens=max_new_tokens, **kw)


class _DummyVision(nn.Module):
    def __init__(self, d: int, n: int) -> None:
        super().__init__()
        self.d, self.n = d, n
        self.proj = nn.Linear(3, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        pooled = x.mean(dim=(2, 3))
        return self.proj(pooled).unsqueeze(1).expand(b, self.n, self.d)


class _DummyLLM(nn.Module):
    """只提供 embedding / forward / config 的最小替身，用于 CPU 形状测试。"""

    def __init__(self, d: int, vocab: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.config = type("C", (), {"hidden_size": d})()

    def get_input_embeddings(self) -> nn.Module:
        return self.emb

    def forward(self, inputs_embeds, attention_mask=None, labels=None,
                return_dict=True, **kw):
        logits = self.head(inputs_embeds)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1), ignore_index=-100)
        out = type("O", (), {"loss": loss, "logits": logits})()
        return out
