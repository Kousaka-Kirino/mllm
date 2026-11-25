"""评测的通用件：把任意 `predict_fn` 变成 Prediction，再聚合成 EvalRecord。

关键设计：**评测不关心模型是什么**。
policy（oracle/prior）、真 VLM、GRPO 后的模型都实现同一个 `predict_fn`，
所以 S3 的基线、S4 的诊断、S5 的干预对比、S6 的预算是同一套代码在跑。
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any, Callable, Iterable, Sequence

from . import answers
from .config import OUTPUTS
from .io_utils import ensure_dir, write_json, write_jsonl
from .schema import EvalRecord, Prediction, Sample
from .stats import summarize, wilson_ci

# predict_fn(sample) -> 答案字符串，或 (答案字符串, 额外信息 dict)
PredictFn = Callable[[Sample], "str | tuple[str, dict[str, Any]]"]


def predict(fn: PredictFn, samples: Sequence[Sample], model_id: str,
            run_id: str, budget: str = "fixed") -> list[Prediction]:
    out: list[Prediction] = []
    for s in samples:
        t0 = time.perf_counter()
        raw = fn(s)
        extra: dict[str, Any] = {}
        if isinstance(raw, tuple):
            raw, extra = raw[0], raw[1]
        ms = (time.perf_counter() - t0) * 1000.0
        sc = answers.score(str(raw), s.answer, s.answer_kind)
        out.append(Prediction(
            sample_id=s.id, model_id=model_id, run_id=run_id, raw=str(raw),
            norm=answers.normalize(str(raw)),
            correct_strict=sc["correct_strict"],
            correct_relaxed=sc["correct_relaxed"],
            latency_ms=round(ms, 3),
            visual_tokens=extra.get("visual_tokens"),
            budget=budget, extra=extra))
    return out


def _suite_of(s: Sample) -> str:
    return str(s.meta.get("suite") or s.source)


def records(predictions: Sequence[Prediction], samples: Sequence[Sample],
            run_id: str, model_id: str, tol: float = 0.05) -> list[EvalRecord]:
    by_id = {s.id: s for s in samples}
    groups: dict[str, list[tuple[Prediction, Sample]]] = defaultdict(list)
    for p in predictions:
        s = by_id.get(p.sample_id)
        if s is not None:
            groups[_suite_of(s)].append((p, s))

    out: list[EvalRecord] = []
    for suite, rows in sorted(groups.items()):
        n = len(rows)
        k = sum(1 for p, _ in rows if p.correct_relaxed)
        lo, hi = wilson_ci(k, n)
        by_q = _breakdown(rows, lambda s: s.qtype)
        by_p = _breakdown(rows, lambda s: s.prior_type)
        by_a = _breakdown(rows, lambda s: _acuity_bucket(s))
        toks = [p.visual_tokens for p, _ in rows if p.visual_tokens]
        ms = [p.latency_ms for p, _ in rows if p.latency_ms is not None]
        out.append(EvalRecord(
            run_id=run_id, suite=suite, model_id=model_id, n=n,
            accuracy=k / n if n else 0.0, ci_low=lo, ci_high=hi,
            tokens_mean=(sum(toks) / len(toks)) if toks else None,
            latency_ms_mean=(sum(ms) / len(ms)) if ms else None,
            by_qtype=by_q, by_prior=by_p, by_acuity=by_a,
            extra={"k": k, "tol": tol}))
    return out


def _acuity_bucket(s: Sample) -> str:
    if s.acuity is None:
        return "unknown"
    for name, px in (("easy", 12.0), ("mid", 8.0), ("hard", 5.0)):
        if abs(s.acuity - px) <= 0.6:
            return name
    return f"{s.acuity:.0f}px"


def _breakdown(rows: Iterable[tuple[Prediction, Sample]],
               key: Callable[[Sample], str]) -> dict[str, Any]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for p, s in rows:
        buckets[key(s)].append(bool(p.correct_relaxed))
    return {k: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for k, v in sorted(buckets.items())}


def save(predictions: Sequence[Prediction], recs: Sequence[EvalRecord],
         run_id: str) -> dict[str, str]:
    base = ensure_dir(OUTPUTS / "eval" / run_id)
    p_path = base / "predictions.jsonl"
    r_path = base / "records.json"
    write_jsonl(p_path, (p.to_dict() for p in predictions))
    write_json(r_path, [r.to_dict() for r in recs])
    return {"predictions": str(p_path), "records": str(r_path)}


def print_table(recs: Sequence[EvalRecord]) -> None:
    for r in recs:
        print(r.line())
