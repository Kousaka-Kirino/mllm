"""S5：干预臂的配对比较。三臂共用同一批样本，只比"同一道题谁答对"。

    python -m stage5_intervene.compare_arms --a sft-test --b grpo-test
"""

from __future__ import annotations

import argparse
import json
import sys

from picovlm import io_utils as io
from picovlm.config import OUTPUTS, REPORTS
from picovlm.report import Skip, require, run
from picovlm.schema import Prediction
from picovlm.stats import paired_compare, summarize


def load_preds(run_id: str) -> dict[str, bool]:
    path = OUTPUTS / "eval" / run_id / "predictions.jsonl"
    if not path.exists():
        raise Skip(f"{path} 不存在")
    return {p.sample_id: bool(p.correct_relaxed)
            for p in (Prediction.from_dict(r) for r in io.read_jsonl(path))}


def compare(a: dict[str, bool], b: dict[str, bool]) -> dict:
    ids = sorted(set(a) & set(b))
    require(bool(ids), "两个 run 没有共同样本")
    return {"n": len(ids), "a": summarize([a[i] for i in ids]),
            "b": summarize([b[i] for i in ids]),
            "paired": paired_compare([a[i] for i in ids], [b[i] for i in ids])}


def to_markdown(name_a: str, name_b: str, res: dict) -> str:
    p = res["paired"]
    return (
        f"# 配对比较：`{name_a}` vs `{name_b}`\n\n"
        f"| | 准确率 | 95% CI |\n|---|---|---|\n"
        f"| {name_a} | {res['a']['accuracy']:.1%} | "
        f"[{res['a']['ci_low']:.1%}, {res['a']['ci_high']:.1%}] |\n"
        f"| {name_b} | {res['b']['accuracy']:.1%} | "
        f"[{res['b']['ci_low']:.1%}, {res['b']['ci_high']:.1%}] |\n\n"
        f"配对（n={p['n']}）：**{p['wins']} 胜 / {p['losses']} 负 / {p['ties']} 平**，"
        f"净胜率 {p['delta']:+.1%}，McNemar p={p['p_value']:.3g}\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    a, b = load_preds(args.a), load_preds(args.b)
    res = compare(a, b)

    if args.verify:
        def sane() -> str:
            p = res["paired"]
            require(p["wins"] + p["losses"] + p["ties"] == res["n"], "计数对不上")
            require(0.0 <= p["p_value"] <= 1.0, "p 值越界")
            return f"{p['wins']}/{p['losses']}/{p['ties']} p={p['p_value']:.3g}"
        return run([("paired-sane", sane)], title="compare_arms")

    md = to_markdown(args.a, args.b, res)
    (REPORTS / f"compare_{args.a}_vs_{args.b}.md").write_text(md, encoding="utf-8")
    io.write_json(OUTPUTS / "eval" / f"compare_{args.a}_vs_{args.b}.json", res)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
