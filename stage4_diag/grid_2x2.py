"""S4 诊断：把一次评测拆成 2×2（acuity × prior），并算 VDR 与两个主效应。

这张表是整个项目的核心图：

                先验一致        先验冲突
    acuity 易    A               B          ← B 掉多少 = "会不会看图"
    acuity 难    C               D

- 先验效应  = A - B   （图看得清时，先验把答案拉走多少）
- acuity 效应 = A - C （先验友好时，看不清损失多少）
- VDR       = 冲突格里"答对"的比例（越高越看图）
- 交互      = (A - B) - (C - D)

用法：
    python -m stage4_diag.grid_2x2 --run-id prior-test
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from picovlm import io_utils as io
from picovlm.config import DATA, OUTPUTS, REPORTS
from picovlm.report import Skip, require, run
from picovlm.schema import Prediction, Sample
from picovlm.stats import summarize

CELLS = (("easy", "consistent"), ("easy", "conflict"),
         ("hard", "consistent"), ("hard", "conflict"))
LEVEL_PX = {"easy": 12.0, "hard": 5.0}


def latest_run() -> str:
    base = OUTPUTS / "eval"
    if not base.exists():
        raise Skip("还没有任何评测 run")
    runs = sorted([p for p in base.iterdir() if (p / "predictions.jsonl").exists()],
                  key=lambda p: p.stat().st_mtime)
    if not runs:
        raise Skip("还没有任何评测 run")
    return runs[-1].name


def load_run(run_id: str) -> tuple[dict[str, Sample], list[Prediction], dict]:
    base = OUTPUTS / "eval" / run_id
    if not (base / "predictions.jsonl").exists():
        raise Skip(f"{run_id} 不存在")
    samples: dict[str, Sample] = {}
    for split in ("train", "dev", "test"):
        path = DATA / "synth" / f"{split}.jsonl"
        if path.exists():
            for r in io.read_jsonl(path):
                s = Sample.from_dict(r)
                samples[s.id] = s
    preds = [Prediction.from_dict(r) for r in io.read_jsonl(base / "predictions.jsonl")]
    grid = io.read_json(base / "grid.json") if (base / "grid.json").exists() else {}
    return samples, preds, grid


def cell_of(s: Sample) -> tuple[str, str] | None:
    if s.acuity is None:
        return None
    level = "easy" if s.acuity >= 10 else "hard" if s.acuity <= 6 else None
    if level is None:
        return None
    suite = str(s.meta.get("suite", ""))
    if suite.endswith("_consistent"):
        prior = "consistent"
    elif suite.endswith("_conflict"):
        prior = "conflict"
    else:
        prior = "consistent" if str(s.prior_answer) == str(s.answer) else "conflict"
    return level, prior


def analyse(samples: dict[str, Sample], preds: list[Prediction]) -> dict:
    cells: dict[str, list[bool]] = defaultdict(list)
    vdr: dict[str, list[bool]] = defaultdict(list)
    for p in preds:
        s = samples.get(p.sample_id)
        if s is None:
            continue
        key = cell_of(s)
        if key:
            cells[f"{key[0]}_{key[1]}"].append(bool(p.correct_relaxed))
        if str(s.meta.get("pair_role")) == "conflict":
            vdr[s.prior_type].append(bool(p.correct_relaxed))

    stats = {k: summarize(v) for k, v in cells.items()}
    table = {k: round(stats[k]["accuracy"], 4) if k in stats else None for k in
             (f"{a}_{b}" for a, b in CELLS)}

    def acc(key: str) -> float | None:
        return table.get(key)

    prior_effect = ((acc("easy_consistent") - acc("easy_conflict"))
                    if None not in (acc("easy_consistent"), acc("easy_conflict")) else None)
    acuity_effect = ((acc("easy_consistent") - acc("hard_consistent"))
                     if None not in (acc("easy_consistent"), acc("hard_consistent")) else None)
    interaction = None
    if None not in (acc("easy_consistent"), acc("easy_conflict"),
                    acc("hard_consistent"), acc("hard_conflict")):
        interaction = ((acc("easy_consistent") - acc("easy_conflict"))
                       - (acc("hard_consistent") - acc("hard_conflict")))

    return {
        "run_id": preds[0].run_id if preds else "",
        "model_id": preds[0].model_id if preds else "",
        "table": table,
        "cells": stats,
        "prior_effect": prior_effect,
        "acuity_effect": acuity_effect,
        "interaction": interaction,
        "vdr": {k: summarize(v) for k, v in vdr.items()},
        "n": len(preds),
    }


def to_markdown(res: dict) -> str:
    t = res["table"]
    fmt = lambda v: "—" if v is None else f"{v:.1%}"  # noqa: E731
    lines = [
        f"# 2×2 诊断 —— `{res['run_id']}`（{res['model_id']}，n={res['n']}）",
        "",
        "| | 先验一致 | 先验冲突 |",
        "|---|---|---|",
        f"| **acuity 易（≈12px）** | {fmt(t.get('easy_consistent'))} | {fmt(t.get('easy_conflict'))} |",
        f"| **acuity 难（≈5px）** | {fmt(t.get('hard_consistent'))} | {fmt(t.get('hard_conflict'))} |",
        "",
        f"- 先验效应（易格掉点）：{fmt(res['prior_effect'])}",
        f"- acuity 效应（一致格掉点）：{fmt(res['acuity_effect'])}",
        f"- 交互：{fmt(res['interaction'])}",
        "",
        "## VDR（冲突格里看图的比例）",
        "",
        "| 先验类型 | n | VDR | 95% CI |",
        "|---|---|---|---|",
    ]
    for k, s in sorted(res["vdr"].items()):
        lines.append(f"| {k} | {s['n']} | {s['accuracy']:.1%} | "
                     f"[{s['ci_low']:.1%}, {s['ci_high']:.1%}] |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="")
    ap.add_argument("--verify", action="store_true", help="只做自检，不写报告")
    args = ap.parse_args(argv)

    samples, preds, _ = load_run(args.run_id or latest_run())
    res = analyse(samples, preds)

    if args.verify:
        def has_cells() -> str:
            require(all(res["table"][k] is not None for k in
                        ("easy_consistent", "easy_conflict",
                         "hard_consistent", "hard_conflict")),
                    f"四格不全：{res['table']}")
            return "四格齐全"

        def monotone() -> str:
            a, b, c = (res["table"]["easy_consistent"], res["table"]["easy_conflict"],
                       res["table"]["hard_consistent"])
            require(a >= b - 1e-9, "先验冲突格不应高于一致格")
            require(a >= c - 1e-9, "acuity 难格不应高于易格")
            return "方向符合预期"

        return run([("four-cells", has_cells), ("direction", monotone)],
                   title="stage4_diag")

    md = to_markdown(res)
    io.write_json(OUTPUTS / "eval" / res["run_id"] / "grid.json", res)
    (REPORTS / f"grid_{res['run_id']}.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
