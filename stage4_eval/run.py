"""S4 · 跑评测：把预测落盘成 `predictions.jsonl` + `records.json`。

    python -m stage4_eval.run --model prior --suite dev_test \
        --conditions correct_image,mismatched_image,prior_only
    python -m stage4_eval.run --model sft --suite locked_test --limit 80
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from picovlm.config import DATA, OUTPUTS, load_config
from picovlm.io_utils import ensure_dir, utc_stamp, write_json, write_jsonl
from picovlm.schema import EvalRecord
from picovlm.stats import wilson_ci

from stage4_eval.predictor import (CONDITIONS, PicoPredictor, RulePrior,
                                   load_pico, predict_all)
from stage3_train.dataset import load_samples


def suite_of(s) -> str:
    """四格命名：`{acuity}_{role}`；role=consistent/conflict/neutral。"""
    role = "neutral" if s.prior_type == "none" else (
        "consistent" if s.prior_holds else "conflict")
    return f"{s.acuity_bin}_{role}"


def to_records(preds, samples, *, run_id: str, model_id: str) -> list[EvalRecord]:
    by_id = {s.id: s for s in samples}
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for p in preds:
        s = by_id[p.sample_id]
        groups[(p.condition, suite_of(s))].append((p, s))
    out: list[EvalRecord] = []
    for (cond, suite), items in sorted(groups.items()):
        n = len(items)
        k = sum(1 for p, _ in items if p.correct_relaxed)
        lo, hi = wilson_ci(k, n)
        by_qtype: dict[str, list[int]] = defaultdict(list)
        by_prior: dict[str, list[int]] = defaultdict(list)
        for p, s in items:
            by_qtype[s.qtype].append(int(bool(p.correct_relaxed)))
            by_prior[s.prior_type].append(int(bool(p.correct_relaxed)))
        out.append(EvalRecord(
            run_id=run_id, suite=suite, model_id=model_id, n=n,
            accuracy=k / n if n else 0.0, ci_low=lo, ci_high=hi,
            tokens_mean=sum(p.visual_tokens or 0 for p, _ in items) / n,
            latency_ms_mean=sum(p.latency_ms or 0 for p, _ in items) / n,
            by_qtype={k2: sum(v) / len(v) for k2, v in by_qtype.items()},
            by_prior={k2: sum(v) / len(v) for k2, v in by_prior.items()},
            condition=cond, n_families=len({s.family_id for _, s in items})))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="prior",
                    choices=["prior", "align", "sft"])
    ap.add_argument("--suite", default="dev_test",
                    choices=["dev_test", "locked_test", "val", "ood_style"],
                    help="ood_style = E5 用同批题换画风（stage1_data.ood 生成）")
    ap.add_argument("--conditions", default="correct_image")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--connector", default=None,
                    help="默认 align→outputs/ckpt/align/connector.pt, sft→outputs/ckpt/sft_pilot/connector.pt")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    conds = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        print(f"未知 condition：{bad}（可选 {CONDITIONS}）")
        return 2

    samples = load_samples(args.suite, limit=args.limit)
    run_id = args.run_id or f"{args.model}_{args.suite}_{utc_stamp()}"
    print(f"== S4 评测 ==\nmodel={args.model} suite={args.suite} n={len(samples)} "
          f"conditions={conds} run={run_id}")

    if args.model == "prior":
        pred = RulePrior()
    else:
        ckpt = args.connector or (
            "outputs/ckpt/align/connector.pt" if args.model == "align"
            else "outputs/ckpt/sft_pilot/connector.pt")
        adapter = args.adapter or (
            "outputs/ckpt/sft_pilot/lora" if args.model == "sft" else None)
        model, tok = load_pico(args.model, device=args.device,
                               connector_ckpt=ckpt, adapter=adapter)
        pred = PicoPredictor(model, tok, args.device, model_id=f"pico:{args.model}")

    preds = predict_all(pred, samples, run_id=run_id, conditions=conds)
    out_dir = ensure_dir(OUTPUTS / "eval" / run_id)
    write_jsonl(out_dir / "predictions.jsonl", (
        {"sample_id": p.sample_id, "model_id": p.model_id,
         "run_id": p.run_id, "raw": p.raw, "norm": p.norm,
         "correct_strict": p.correct_strict,
         "correct_relaxed": p.correct_relaxed,
         "latency_ms": p.latency_ms, "visual_tokens": p.visual_tokens,
         "condition": p.condition, "family_id": p.family_id}
        for p in preds))
    records = to_records(preds, samples, run_id=run_id, model_id=pred.model_id)
    write_json(out_dir / "records.json", [r.__dict__ for r in records])
    acc = sum(1 for p in preds if p.correct_relaxed) / max(len(preds), 1)
    print(f"predictions {len(preds)}  overall {acc:.3f}")
    for r in records:
        print(f"  {r.condition:<17} {r.suite:<20} n={r.n:<4} "
              f"acc={r.accuracy:.3f} [{r.ci_low:.2f},{r.ci_high:.2f}]")
    print(f"saved {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
