"""S3 数据集：把 `data/synth/*.jsonl` 变成 (pixel_values, input_ids, labels)。

三条规矩（和 `stage2_model.check_loss_mask` 一致）：
1. labels 的 `-100` 只留给"视觉 token + prompt + padding"；
2. 图像预处理**训练与评测必须同一套**（`resize → [-1,1]`），否则评测算的是别的分布；
3. 答案末尾一定补 EOS——不然模型学不会"什么时候停"。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from picovlm.config import DATA, ROOT
from picovlm.schema import Sample

PROMPT = "Question: {q}\nAnswer:"
EOS_SUFFIX = "\n"


def answer_with_eos(tok, answer: str) -> str:
    """答案后面接**真正的 EOS**，模型才学得会"说完就停"。

    早期版本用 `"\\n"` 当结尾，结果模型答出 "A" 之后继续写
    "\\nSolving for:ln("——评测算错，GRPO 的格式分也永远是 0。
    tokenizer 有 eos_token 就用它，没有才退回换行。
    """
    eos = getattr(tok, "eos_token", None)
    return answer + (eos if eos else EOS_SUFFIX)


def load_samples(split: str = "train", root: Path | None = None,
                 limit: int | None = None) -> list[Sample]:
    root = Path(root or DATA / "synth")
    path = root / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 不存在；先跑 `python -m stage1_data.build --profile pilot`")
    out: list[Sample] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(Sample(**d))
            if limit and len(out) >= limit:
                break
    return out


def load_image(rel_path: str, image_size: int = 384,
               root: Path | None = None) -> np.ndarray:
    root = Path(root or ROOT)
    with Image.open(root / rel_path) as im:
        im = im.convert("RGB").resize((image_size, image_size),
                                      Image.BICUBIC)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5                       # SigLIP 的 [-1,1]
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


class ChartSFT:
    """单条样本 → 可直接进 PicoVLM 的张量字典（不 pad，pad 在 collate 里做）。"""

    def __init__(self, samples: list[Sample], tokenizer, image_size: int = 384,
                 root: Path | None = None) -> None:
        self.samples = samples
        self.tok = tokenizer
        self.image_size = image_size
        self.root = Path(root or ROOT)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        prompt_ids = self.tok(PROMPT.format(q=s.question),
                              add_special_tokens=False)["input_ids"]
        answer_ids = self.tok(answer_with_eos(self.tok, s.answer),
                              add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + list(answer_ids)
        return {
            "sample_id": s.id,
            "pixel_values": load_image(s.image_path, self.image_size, self.root),
            "input_ids": input_ids,
            "labels": labels,
            "answer": s.answer,
        }


def collate(batch: list[dict], pad_id: int) -> dict:
    """右 padding；attention_mask 与 labels 同步 pad。"""
    n = max(len(b["input_ids"]) for b in batch)
    ids, lab, msk = [], [], []
    for b in batch:
        k = n - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * k)
        lab.append(b["labels"] + [-100] * k)
        msk.append([1] * len(b["input_ids"]) + [0] * k)
    return {
        "sample_id": [b["sample_id"] for b in batch],
        "answer": [b["answer"] for b in batch],
        "pixel_values": np.stack([b["pixel_values"] for b in batch]),
        "input_ids": np.array(ids, dtype=np.int64),
        "attention_mask": np.array(msk, dtype=np.int64),
        "labels": np.array(lab, dtype=np.int64),
    }
