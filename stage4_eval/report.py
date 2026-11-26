"""S4 · 出报告：`python -m stage4_eval.report --run-id sft_dev`。

产出一份 markdown：四格 + 效应量 + family bootstrap + 与盲基线的 McNemar。
报告里的每个数字都来自 `outputs/eval/<run>/predictions.jsonl`，可重算、可回溯。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from picovlm.config import OUTPUTS, REPORTS
from picovlm.io_utils import ensure_dir, write_json

from stage3_train.dataset import load_samples
from stage4_eval.analyse import (bootstrap_effect, cell_accuracy, effects,
                                 load_run, mcnemar_vs, to_markdown)


def build(run_id: str, baseline: str | None = None, suite: str = "dev_test") -> str:
    preds = load_run(run_id)
    samples = {s.id: s for s in load_samples(suite)}
    cells = cell_accuracy(preds, samples)
    eff = effects(cells)
    boot = bootstrap_effect(preds, samples, n_boot=2000)
    md = [f"# S4 2×2 诊断 · `{run_id}`", "",
          f"- 样本数：{sum(v['n'] for v in cells.values())}（{suite}），"
          f"family 数：{len({p['family_id'] for p in preds})}",
          f"- 判分：宽松口径（数值容差 5%），同时报表严格口径",
          ""]
    md.append(to_markdown(run_id, cells, eff, boot))
    if baseline:
        base = load_run(baseline)
        mc = mcnemar_vs(preds, base)
        md += ["", f"### 与基线 `{baseline}` 的配对比较（McNemar 精确检验）", "",
               f"- 共同样本 n={mc['n']}，仅本模型对 {mc['a_only']}，"
               f"仅基线对 {mc['b_only']}，delta={mc['delta']:+.3f}，"
               f"p={mc['p_value']:.4g}"]
        write_json(REPORTS / f"mcnemar_{run_id}.json", mc)
    md += ["", "---", "",
           "> 读法：h 一致格高、冲突格低 → 先验在起作用；acuity_effect 大 → 字号是瓶颈。",
           "> 冲突格（VDR）才是「看图」的证据：图能翻掉先验，才叫真的在看图。"]
    out_dir = ensure_dir(REPORTS)
    path = out_dir / f"eval_{run_id}.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return str(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--baseline", default="prior_dev")
    ap.add_argument("--suite", default="dev_test")
    args = ap.parse_args(argv)
    base = args.baseline if (OUTPUTS / "eval" / args.baseline).exists() else None
    path = build(args.run_id, base, args.suite)
    print(f"report {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
