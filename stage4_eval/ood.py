"""S4 · E5 迁移：同批题、换画风之后掉多少。

两个 run 是**同实例配对**（sample_id 相同），所以可以：

- 逐格比较 in-domain 与 OOD 的准确率；
- 对"整体"和"每格"做 McNemar 精确检验；
- 再看 prior_only（灰图）是否也被画风拖垮——如果灰图掉得比换画风还多，
  说明模型对"图长什么样"本来就没多敏感。

用法：

    python -m stage4_eval.ood --in-domain sft_dev --ood sft_ood
"""

from __future__ import annotations

import argparse
import json

from picovlm.config import OUTPUTS, REPORTS
from picovlm.io_utils import ensure_dir, write_json
from picovlm.stats import mcnemar_exact, wilson_ci

from stage3_train.dataset import load_samples
from stage4_eval.analyse import CELLS, cell_accuracy, load_run

LABEL = {"easy_consistent": "易·一致", "easy_conflict": "易·冲突",
         "hard_consistent": "难·一致", "hard_conflict": "难·冲突"}


def _by_id(preds: list[dict], condition: str) -> dict[str, dict]:
    return {p["sample_id"]: p for p in preds if p["condition"] == condition}


def compare(run_in: str, run_ood: str, condition: str = "correct_image",
            suite: str = "dev_test") -> dict:
    a_preds, b_preds = load_run(run_in), load_run(run_ood)
    a = _by_id(a_preds, condition)
    b = _by_id(b_preds, condition)
    ids = sorted(set(a) & set(b))
    ok_a = {i: bool(a[i]["correct_relaxed"]) for i in ids}
    ok_b = {i: bool(b[i]["correct_relaxed"]) for i in ids}
    a_only = sum(1 for i in ids if ok_a[i] and not ok_b[i])
    b_only = sum(1 for i in ids if ok_b[i] and not ok_a[i])
    n = len(ids)
    acc_a = sum(ok_a.values()) / n if n else 0.0
    acc_b = sum(ok_b.values()) / n if n else 0.0

    samples = {s.id: s for s in load_samples(suite)}
    cells_a = cell_accuracy(a_preds, samples)
    cells_b = cell_accuracy(b_preds, samples)

    # 空回答率：换画风后模型可能**根本不出话**，这跟"答错"是两回事，必须分开报
    empty_a = sum(1 for i in ids if not (a[i].get("raw") or "").strip())
    empty_b = sum(1 for i in ids if not (b[i].get("raw") or "").strip())

    per_cell = {}
    for c in CELLS:
        ids_c = [i for i in ids if i in samples and samples[i].family_id
                 and cell_of(samples[i]) == c]
        if not ids_c:
            continue
        x = sum(1 for i in ids_c if ok_a.get(i) and not ok_b.get(i))
        y = sum(1 for i in ids_c if ok_b.get(i) and not ok_a.get(i))
        per_cell[c] = {"n": len(ids_c), "in": cells_a[c]["acc"],
                       "ood": cells_b[c]["acc"],
                       "delta": cells_b[c]["acc"] - cells_a[c]["acc"],
                       "p_value": mcnemar_exact(x, y)}

    return {
        "run_in": run_in, "run_ood": run_ood, "condition": condition,
        "n_pairs": n,
        "acc_in": acc_a, "acc_ood": acc_b, "delta": acc_b - acc_a,
        "ci_in": wilson_ci(sum(ok_a.values()), n) if n else (0.0, 0.0),
        "ci_ood": wilson_ci(sum(ok_b.values()), n) if n else (0.0, 0.0),
        "flip_in_only": a_only, "flip_ood_only": b_only,
        "p_value": mcnemar_exact(a_only, b_only),
        "empty_in": empty_a, "empty_ood": empty_b,
        "empty_rate_in": empty_a / n if n else 0.0,
        "empty_rate_ood": empty_b / n if n else 0.0,
        "per_cell": per_cell,
        "cells_in": cells_a, "cells_ood": cells_b,
    }


def cell_of(sample) -> str:
    """从 Sample 的 acuity_bin + prior 角色拼出 2×2 格名（与 S4 一致）。"""
    role = sample.meta.get("prior_role") or (
        "conflict" if sample.prior_holds is False else "consistent")
    bin_ = sample.acuity_bin
    if bin_ in ("easy", "mid"):
        acu = "easy"
    elif bin_ in ("hard", "extreme"):
        acu = "hard"
    else:
        acu = "easy"
    if role not in ("consistent", "conflict"):
        return "neutral"
    return f"{acu}_{role}"


def to_markdown(res: dict) -> str:
    n = res["n_pairs"]
    md = [f"# E5 · 画风迁移（`{res['run_in']}` → `{res['run_ood']}`）", "",
          f"- 同实例配对 n={n}，条件 `{res['condition']}`",
          f"- in-domain **{res['acc_in']:.3f}** [{res['ci_in'][0]:.2f}, {res['ci_in'][1]:.2f}]  "
          f"vs OOD **{res['acc_ood']:.3f}** [{res['ci_ood'][0]:.2f}, {res['ci_ood'][1]:.2f}]",
          f"- 差 **{res['delta']:+.3f}**，翻转：只有 in-domain 对 {res['flip_in_only']}，"
          f"只有 OOD 对 {res['flip_ood_only']}，McNemar p={res['p_value']:.4g}",
          f"- **空回答**：in-domain {res['empty_in']}/{n}（{res['empty_rate_in']:.1%}）"
          f" vs OOD {res['empty_ood']}/{n}（{res['empty_rate_ood']:.1%}）",
          "", "| 格 | n | in-domain | OOD | Δ | p |", "|---|---:|---:|---:|---:|---:|"]
    for c in CELLS:
        v = res["per_cell"].get(c)
        if not v:
            continue
        md.append(f"| {LABEL[c]} | {v['n']} | {v['in']:.3f} | {v['ood']:.3f} | "
                  f"{v['delta']:+.3f} | {v['p_value']:.3g} |")
    md += ["", "> 读法：换画风掉点是正常的（分布偏移）；掉到接近灰图基线，说明模型"
           "对像素细节本就不敏感，靠的是先验 + 版面位置。", ""]
    return "\n".join(md)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-domain", default="sft_dev")
    ap.add_argument("--ood", default="sft_ood")
    ap.add_argument("--condition", default="correct_image")
    ap.add_argument("--suite", default="dev_test")
    args = ap.parse_args(argv)
    res = compare(args.in_domain, args.ood, args.condition, args.suite)
    out = ensure_dir(REPORTS) / f"ood_{args.ood}.md"
    out.write_text(to_markdown(res), encoding="utf-8")
    write_json(OUTPUTS / "eval" / args.ood / "ood.json", res)
    print(f"E5 {res['acc_in']:.3f} → {res['acc_ood']:.3f} "
          f"({res['delta']:+.3f}), p={res['p_value']:.4g} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
