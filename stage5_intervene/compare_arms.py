"""S5 · 干预臂对比：同一评测集、同一判分、逐格配对。

    python -m stage5_intervene.compare_arms --a sft_dev --b curr_dev
"""

from __future__ import annotations

import argparse

from picovlm.config import OUTPUTS, REPORTS
from picovlm.io_utils import ensure_dir, write_json

from stage3_train.dataset import load_samples
from stage4_eval.analyse import cell_accuracy, load_run, mcnemar_vs


def compare(a: str, b: str, suite: str = "dev_test",
            condition: str = "correct_image") -> dict:
    pa, pb = load_run(a), load_run(b)
    samples = {s.id: s for s in load_samples(suite)}
    ca, cb = cell_accuracy(pa, samples, condition), cell_accuracy(pb, samples, condition)
    mc = mcnemar_vs(pa, pb, condition)
    rows = {}
    for c in ca:
        rows[c] = {"a": ca[c]["acc"], "b": cb[c]["acc"],
                   "delta": ca[c]["acc"] - cb[c]["acc"],
                   "n": ca[c]["n"]}
    return {"run_a": a, "run_b": b, "cells": rows, "mcnemar": mc}


def to_markdown(res: dict) -> str:
    lines = [f"### `{res['run_a']}` vs `{res['run_b']}`", "",
             "| 格 | A | B | Δ(A−B) | n |", "|---|---:|---:|---:|---:|"]
    for c, v in res["cells"].items():
        lines.append(f"| {c} | {v['a']:.3f} | {v['b']:.3f} | {v['delta']:+.3f} | {v['n']} |")
    mc = res["mcnemar"]
    lines += ["", f"- McNemar：n={mc['n']}，仅 A 对 {mc['a_only']}，仅 B 对 "
                  f"{mc['b_only']}，delta={mc['delta']:+.3f}，p={mc['p_value']:.4g}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--suite", default="dev_test")
    args = ap.parse_args(argv)
    res = compare(args.a, args.b, args.suite)
    md = to_markdown(res)
    print(md)
    out = ensure_dir(REPORTS) / f"compare_{args.a}_vs_{args.b}.md"
    out.write_text(md + "\n", encoding="utf-8")
    write_json(REPORTS / f"compare_{args.a}_vs_{args.b}.json", res)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
