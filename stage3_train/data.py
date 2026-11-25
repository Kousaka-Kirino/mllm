"""S3 的数据侧：把 Sample 变成 (pixel_values, input_ids, labels)。

**这里只有一处容易错、且错了会静默降分**：labels 的构造。

    labels = [-100] * len(prompt) + answer_ids

- 视觉 token 的 -100 由 `picovlm.model.assemble_inputs` 补在**前面**；
- prompt 的 -100 在这里补；
- 只有 answer_ids 参与 loss。

如果用整句 labels（不掩 prompt），模型会去学"复述问题"，loss 看着降、
准确率不动——S3 最常见的假进步。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from picovlm.config import ROOT
from picovlm.schema import Sample

PROMPT = "Question: {q}\nAnswer:"


def _tok_ids(tok, text: str) -> list[int]:
    out = tok(text, add_special_tokens=False)
    return list(out["input_ids"] if hasattr(out, "keys") else out)


def build_example(sample: Sample, tok, image_size: int = 384,
                  root=None) -> dict:
    prompt = PROMPT.format(q=sample.question)
    p_ids = _tok_ids(tok, prompt)
    a_ids = _tok_ids(tok, " " + str(sample.answer))
    labels = [-100] * len(p_ids) + a_ids
    img = Image.open((root or ROOT) / sample.image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return {
        "pixel_values": torch.from_numpy(arr).permute(2, 0, 1),
        "input_ids": torch.tensor(p_ids + a_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "answer": str(sample.answer),
    }


class SynthSFT(Dataset):
    def __init__(self, samples: Sequence[Sample], tok, image_size: int = 384,
                 root=None) -> None:
        self.samples = list(samples)
        self.tok = tok
        self.image_size = image_size
        self.root = root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        return build_example(self.samples[i], self.tok, self.image_size, self.root)


def collate(batch: list[dict], pad_id: int = 0) -> dict:
    """右侧 padding；attention_mask 同步；labels 的 pad 位置 = -100。"""
    L = max(b["input_ids"].numel() for b in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].numel()
        ids[i, :n] = b["input_ids"]
        lab[i, :n] = b["labels"]
        mask[i, :n] = 1
    pix = torch.stack([b["pixel_values"] for b in batch])
    # SigLIP 归一化：mean=std=0.5
    pix = (pix - 0.5) / 0.5
    return {"pixel_values": pix, "input_ids": ids,
            "attention_mask": mask, "labels": lab}
