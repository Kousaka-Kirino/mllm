"""S4 · 2×2 诊断：把 `predictions.jsonl` 变成"看到了什么"。

核心量（RQ1 的四格）：

```
                 prior 一致        prior 冲突
    acuity 易    A (easy_cons)     B (easy_conf)
    acuity 难    C (hard_cons)     D (hard_conf)
```

- `prior_effect`  = A − B   （同样的图，只把先验从一致翻成冲突，掉多少）
- `acuity_effect` = A − C   （先验一致时，把字变小，掉多少）
- `interaction`   = (A−B) − (C−D)  （两个因素是否相乘）
- **VDR**（visual dependence rate，冲突格答对率）= B、D 的准确率。
  VDR 高 = 图能压过先验；VDR ≈ 0 = 模型完全被先验支配。

统计口径（2.0 §6）：Wilson CI 给单格；**按 family 聚类 bootstrap** 给效应量的 CI；
同实例比较用 McNemar 精确检验并报不一致对数。
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from picovlm.config import OUTPUTS
from picovlm.stats import mcnemar_exact, wilson_ci

CELLS = ("easy_consistent", "easy_conflict", "hard_consistent", "hard_conflict")


def load_run(run_id: str) -> list[dict]:
    p = OUTPUTS / "eval" / run_id / "predictions.jsonl"
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cell_of(rec: dict) -> str | None:
    """从 prediction 里拿不到 acuity，这里用 suite 的写法反推：需配合样本表。"""
    return rec.get("suite") or None


def cell_accuracy(preds: list[dict], samples: dict[str, object],
                  condition: str = "correct_image") -> dict[str, dict]:
    from stage4_eval.run import suite_of

    cells: dict[str, list[dict]] = {c: [] for c in CELLS}
    for p in preds:
        if p["condition"] != condition:
            continue
        s = samples.get(p["sample_id"])
        if s is None:
            continue
        key = suite_of(s)
        if key in cells:
            cells[key].append(p)
    out = {}
    for c, items in cells.items():
        n = len(items)
        k = sum(1 for p in items if p["correct_relaxed"])
        lo, hi = wilson_ci(k, n) if n else (0.0, 1.0)
        out[c] = {"n": n, "k": k, "acc": k / n if n else 0.0,
                  "ci": [lo, hi],
                  "families": len({p["family_id"] for p in items})}
    return out


def effects(cells: dict[str, dict]) -> dict:
    a = cells["easy_consistent"]["acc"]
    b = cells["easy_conflict"]["acc"]
    c = cells["hard_consistent"]["acc"]
    d = cells["hard_conflict"]["acc"]
    return {"prior_effect": a - b, "acuity_effect": a - c,
            "interaction": (a - b) - (c - d),
            "vdr_easy": b, "vdr_hard": d}


def bootstrap_effect(preds: list[dict], samples: dict[str, object],
                     fn=effects, n_boot: int = 2000, seed: int = 7) -> dict:
    """按 family 整簇重采样（同一族的样本永远一起抽）。"""
    from stage4_eval.run import suite_of

    by_family: dict[str, list[dict]] = {}
    for p in preds:
        if p["condition"] != "correct_image":
            continue
        s = samples.get(p["sample_id"])
        if s is None or suite_of(s) not in CELLS:
            continue
        by_family.setdefault(p["family_id"], []).append(p)
    fams = list(by_family)
    rng = random.Random(seed)
    stats: dict[str, list[float]] = {}
    for _ in range(n_boot):
        pick = [fams[rng.randrange(len(fams))] for _ in fams]
        flat = [p for f in pick for p in by_family[f]]
        e = fn(cell_accuracy(flat, samples))
        for k, v in e.items():
            stats.setdefault(k, []).append(v)
    out = {}
    for k, vals in stats.items():
        vals.sort()
        out[k] = {"point": fn(cell_accuracy(preds, samples))[k],
                  "lo": vals[int(0.025 * len(vals))],
                  "hi": vals[int(0.975 * len(vals)) - 1]}
    return out


def mcnemar_vs(preds_a: list[dict], preds_b: list[dict],
               condition: str = "correct_image") -> dict:
    """两条预测流在**同实例**上的配对比较（只比两边都有的样本）。"""
    a = {(p["sample_id"], p["condition"]): bool(p["correct_relaxed"])
         for p in preds_a if p["condition"] == condition}
    b = {(p["sample_id"], p["condition"]): bool(p["correct_relaxed"])
         for p in preds_b if p["condition"] == condition}
    keys = set(a) & set(b)
    a_only = sum(1 for k in keys if a[k] and not b[k])
    b_only = sum(1 for k in keys if b[k] and not a[k])
    p = mcnemar_exact(a_only, b_only)
    return {"n": len(keys), "a_only": a_only, "b_only": b_only,
            "delta": (sum(a[k] for k in keys) - sum(b[k] for k in keys)) / max(len(keys), 1),
            "p_value": p}


def mcnemar_conditions(preds: list[dict], cond_a: str, cond_b: str) -> dict:
    """同一批样本、同一条预测流里比较两个**条件**（correct_image vs mismatched_image）。

    和 `mcnemar_vs` 的区别：那个比两个模型，这个比同一个模型的两种输入。
    "换图不掉点"这条结论就是靠它给的 p 值，不是靠肉眼看两张表差不多。
    """
    a = {p["sample_id"]: bool(p["correct_relaxed"])
         for p in preds if p["condition"] == cond_a}
    b = {p["sample_id"]: bool(p["correct_relaxed"])
         for p in preds if p["condition"] == cond_b}
    keys = set(a) & set(b)
    a_only = sum(1 for k in keys if a[k] and not b[k])
    b_only = sum(1 for k in keys if b[k] and not a[k])
    return {"cond_a": cond_a, "cond_b": cond_b, "n": len(keys),
            f"{cond_a}_only": a_only, f"{cond_b}_only": b_only,
            "acc_a": sum(a[k] for k in keys) / max(len(keys), 1),
            "acc_b": sum(b[k] for k in keys) / max(len(keys), 1),
            "p_value": mcnemar_exact(a_only, b_only)}


def to_markdown(name: str, cells: dict, eff: dict, boot: dict | None = None) -> str:
    lines = [f"### {name}", "", "| 格 | n | 族 | acc | 95% Wilson |",
             "|---|---:|---:|---:|---|"]
    label = {"easy_consistent": "易·一致", "easy_conflict": "易·冲突",
             "hard_consistent": "难·一致", "hard_conflict": "难·冲突"}
    for c in CELLS:
        v = cells[c]
        lines.append(f"| {label[c]} | {v['n']} | {v['families']} | {v['acc']:.3f} | "
                     f"[{v['ci'][0]:.2f}, {v['ci'][1]:.2f}] |")
    lines += ["", f"- prior_effect（一致−冲突，易）= **{eff['prior_effect']:+.3f}**",
              f"- acuity_effect（易−难，一致）= **{eff['acuity_effect']:+.3f}**",
              f"- interaction = {eff['interaction']:+.3f}",
              f"- VDR（冲突格答对率）= 易 {eff['vdr_easy']:.3f} / 难 {eff['vdr_hard']:.3f}"]
    if boot:
        lines += ["", "| 效应 | 点估计 | family bootstrap 95% |", "|---|---:|---|"]
        for k, v in boot.items():
            lines.append(f"| {k} | {v['point']:+.3f} | [{v['lo']:+.3f}, {v['hi']:+.3f}] |")
    return "\n".join(lines)
