"""S4 · 预测器：把模型/规则包成统一的 `answer(sample, condition)`。

四种条件（E0 实验矩阵的前四行）是 RQ1 的全部原料：

| condition | 图 | 问题 | 用来测什么 |
|---|---|---|---|
| `correct_image` | 正确图 | 真实问题 | 完整能力 |
| `mismatched_image` | 别族的图 | 真实问题 | 不看图、靠先验能不能答对 |
| `prior_only` | 纯灰图 | 真实问题 | 纯先验（盲基线） |
| `image_only` | 正确图 | 泛化问句 | 只看图、不看问题 |

`mismatched_image` 是关键设计：如果模型在冲突格上仍然答"先验答案"，它在
mismatched 条件下的表现会**接近** correct_image——这就是"不看图"的直接证据。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from picovlm.answers import is_correct, normalize
from picovlm.config import OUTPUTS, ROOT, load_config, model_config_from
from picovlm.io_utils import utc_stamp
from picovlm.model.vlm import PicoVLM, Batch
from picovlm.schema import Prediction, Sample

from stage3_train.dataset import EOS_SUFFIX, PROMPT, load_image

CONDITIONS = ("correct_image", "mismatched_image", "prior_only", "image_only")
IMAGE_ONLY = "Describe the chart in one word."
PROMPT_IMAGE_ONLY = "Question: {q}\nAnswer:"


def gray_image(image_size: int = 384) -> np.ndarray:
    """纯灰图：给模型"有图但没信息"的输入，测纯先验。用 0.0 而不是 -1.0。"""
    return np.zeros((3, image_size, image_size), dtype=np.float32)


def load_pico(kind: str, *, device: str = "cuda", connector_ckpt: str | None = None,
              adapter: str | None = None) -> tuple[PicoVLM, object]:
    """kind ∈ {"align","sft"}；按需载入连接器 / LoRA 适配器。"""
    from transformers import AutoTokenizer

    cfg = load_config("model.yaml")
    mc = model_config_from(cfg)
    tok = AutoTokenizer.from_pretrained(mc.llm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = PicoVLM(mc, load_weights=True).to(device).to(torch.bfloat16).eval()
    if connector_ckpt:
        p = Path(connector_ckpt)
        if not p.is_absolute():
            p = OUTPUTS.parent / p
        model.connector.load_state_dict(torch.load(p, map_location="cpu"))
    if kind == "sft" and adapter:
        from peft import PeftModel
        p = Path(adapter)
        if not p.is_absolute():
            p = OUTPUTS.parent / p
        model.llm = PeftModel.from_pretrained(model.llm, p)
    return model, tok


class RulePrior:
    """盲基线：只用先验答案（`prior_answer`，没有就用 meta 里的多数答案）。"""

    def __init__(self, model_id: str = "rule:prior") -> None:
        self.model_id = model_id
        self.visual_tokens = 0

    def answer(self, sample: Sample, image: np.ndarray) -> str:
        if sample.prior_answer:
            return sample.prior_answer
        return str(sample.meta.get("majority_answer", ""))


class PicoPredictor:
    def __init__(self, model: PicoVLM, tok, device: str = "cuda",
                 model_id: str = "pico", max_new_tokens: int = 8) -> None:
        self.model = model
        self.tok = tok
        self.device = device
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.visual_tokens = int(model.n_visual_tokens)

    @torch.no_grad()
    def answer(self, sample: Sample, image: np.ndarray) -> str:
        px = torch.from_numpy(np.ascontiguousarray(image)[None]).to(self.device)
        text = sample.question if True else ""
        ids = self.tok(PROMPT.format(q=text),
                       return_tensors="pt", add_special_tokens=False)
        gen = self.model.generate(
            px, ids["input_ids"].to(self.device),
            ids["attention_mask"].to(self.device),
            max_new_tokens=self.max_new_tokens, do_sample=False,
            pad_token_id=self.tok.pad_token_id)
        return self.tok.decode(gen[0], skip_special_tokens=True).strip()

    def answer_image_only(self, sample: Sample, image: np.ndarray) -> str:
        """把问题换成一个与数值无关的泛化问句，测"不看问题"的纯视觉行为。"""
        return self.answer(Sample(**{**sample.__dict__, "question": IMAGE_ONLY}),
                           image)


def image_for(condition: str, sample: Sample, partner: Sample | None,
              image_size: int = 384, root: Path | None = None) -> np.ndarray:
    if condition == "prior_only":
        return gray_image(image_size)
    if condition == "mismatched_image":
        if partner is None:
            raise ValueError("mismatched_image 需要一条不同族的 partner 样本")
        return load_image(partner.image_path, image_size, root)
    return load_image(sample.image_path, image_size, root)


def predict_all(predictor, samples: list[Sample], *, run_id: str,
                conditions: tuple[str, ...] = ("correct_image",),
                image_size: int = 384, sleep: bool = False) -> list[Prediction]:
    """逐条跑；`mismatched_image` 用"索引 + 一半长度"取异族图（pilot 每族只有 1 条）。"""
    out: list[Prediction] = []
    for i, s in enumerate(samples):
        partner = samples[(i + len(samples) // 2) % len(samples)]
        for cond in conditions:
            t0 = time.time()
            if cond == "image_only" and hasattr(predictor, "answer_image_only"):
                raw = predictor.answer_image_only(s, load_image(s.image_path, image_size))
                img = load_image(s.image_path, image_size)
                _ = img
            else:
                img = image_for(cond, s, partner, image_size)
                raw = predictor.answer(s, img)
            ms = (time.time() - t0) * 1000
            out.append(Prediction(
                sample_id=s.id, model_id=predictor.model_id, run_id=run_id,
                raw=raw, norm=normalize(raw),
                correct_strict=is_correct(raw, s.answer, s.answer_kind, mode="strict"),
                correct_relaxed=is_correct(raw, s.answer, s.answer_kind, mode="relaxed"),
                latency_ms=round(ms, 1), visual_tokens=predictor.visual_tokens,
                condition=cond, family_id=s.family_id))
            if sleep:
                time.sleep(0.01)
    return out
