"""S5 · RFT（rejection finetune）：把"采样到的好答案"当监督信号。

它和 GRPO 是**同一份 rollout 的两种用法**：

- GRPO：用组内相对好坏当 advantage（只用在线的排序信号）；
- RFT：只留正确答案，当成普通 SFT 目标（离线、离策略）。

这也是一个必须做的对照：如果 RFT 花更少的算力就达到 GRPO 的效果，
那"GRPO 更好"的结论就不成立——PandaVLA 在纯文本任务上就撞到了这一点
（同数据 SFT 逐位相同，84s vs 3313s）。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from picovlm.answers import is_correct
from picovlm.config import OUTPUTS
from picovlm.io_utils import ensure_dir, utc_stamp, write_jsonl
from picovlm.schema import Sample

from stage3_train.dataset import PROMPT, load_image
from stage5_intervene import grpo as G


def build(model, tok, samples: list[Sample], *, group_size: int = 4,
          device: str = "cuda", image_size: int = 384,
          out: Path | None = None) -> dict:
    kept, total = [], 0
    for s in samples:
        px = torch.from_numpy(load_image(s.image_path, image_size)[None]).to(device)
        ids = tok(PROMPT.format(q=s.question), return_tensors="pt",
                  add_special_tokens=False)["input_ids"].to(device)
        gen = G.rollout(model, tok, px, ids, group_size=group_size)
        total += len(gen)
        best = None
        for g in gen:
            text = tok.decode(g, skip_special_tokens=True).strip()
            if is_correct(text, s.answer, s.answer_kind) and G.format_ok(text):
                best = text
                break
        if best:
            kept.append({"sample_id": s.id, "answer": s.answer,
                         "pred": best, "family_id": s.family_id,
                         "qtype": s.qtype, "prior_type": s.prior_type})
    out = Path(out or (OUTPUTS / "rft" / f"rft_{utc_stamp()}.jsonl"))
    ensure_dir(out.parent)
    write_jsonl(out, kept)
    stats = {"samples": len(samples), "rollouts": total, "kept": len(kept),
             "keep_rate": len(kept) / max(len(samples), 1),
             "path": str(out)}
    (out.parent / "meta.json").write_text(json.dumps(stats, ensure_ascii=False,
                                                     indent=2), encoding="utf-8")
    return stats
