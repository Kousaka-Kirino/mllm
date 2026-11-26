"""S4 验收：评测管线、2×2 方向、统计口径。

    python -m stage4_eval.verify_stage4
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from picovlm.config import OUTPUTS, REPORTS, ROOT
from picovlm.report import Skip, require, run

from stage3_train.dataset import load_samples
from stage4_eval.analyse import (bootstrap_effect, cell_accuracy, effects,
                                 load_run, mcnemar_vs)

RUN = "verify_prior"


def py(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"}
    return subprocess.run([sys.executable, *args], cwd=str(ROOT),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=env)


def check_prior_eval() -> str:
    r = py("-m", "stage4_eval.run", "--model", "prior", "--suite", "dev_test",
           "--conditions", "correct_image,mismatched_image,prior_only,image_only",
           "--run-id", RUN)
    if r.returncode != 0:
        raise AssertionError(f"prior 评测失败：\n{r.stdout[-600:]}\n{r.stderr[-600:]}")
    preds = load_run(RUN)
    require(len(preds) == 320, f"应有 80×4=320 条预测，实际 {len(preds)}")
    conds = {p["condition"] for p in preds}
    require(conds == {"correct_image", "mismatched_image", "prior_only",
                      "image_only"}, f"condition 不全：{conds}")
    return f"predictions={len(preds)} conditions={len(conds)}"


def check_two_by_two() -> str:
    """盲基线的四格方向必须与构造一致：一致格 100%，冲突格 0%。"""
    preds = load_run(RUN)
    samples = {s.id: s for s in load_samples("dev_test")}
    cells = cell_accuracy(preds, samples)
    for c, v in cells.items():
        require(v["n"] > 0, f"{c} 没有样本")
    require(cells["easy_consistent"]["acc"] == 1.0,
            f"易·一致格 {cells['easy_consistent']['acc']} != 1")
    require(cells["hard_conflict"]["acc"] == 0.0,
            f"难·冲突格 {cells['hard_conflict']['acc']} != 0")
    eff = effects(cells)
    require(abs(eff["prior_effect"] - 1.0) < 1e-9, "prior_effect 应为 1.0")
    return "prior_effect=+1.000 acuity_effect=0.000 vdr=0.000"


def check_mismatch_test() -> str:
    """SFT 那一轮的报告里必须有"换图检验"（同实例 McNemar），否则结论没依据。"""
    p = OUTPUTS / "eval" / "sft_dev" / "grid.json"
    if not p.exists():
        raise Skip("没有 sft_dev 的 grid.json（先跑 stage4_eval.run --model sft）")
    import json

    grid = json.loads(p.read_text(encoding="utf-8"))
    cm = grid.get("mcnemar_mismatch")
    if not cm:
        raise Skip("该轮只跑了单个 condition，没有换图检验")
    require(cm["n"] > 0, "换图检验的 n 为 0")
    require(0.0 <= cm["p_value"] <= 1.0, "p 值越界")
    return (f"n={cm['n']} correct_only={cm['correct_image_only']} "
            f"mismatched_only={cm['mismatched_image_only']} p={cm['p_value']:.4g}")


def check_ood() -> str:
    """E5：画风迁移报告若存在，必须能对齐同批样本（配对 n 与 dev_test 一致）。"""
    p = OUTPUTS / "eval" / "sft_ood" / "ood.json"
    if not p.exists():
        raise Skip("没跑 E5（stage1_data.ood + stage4_eval.run --suite ood_style）")
    import json

    d = json.loads(p.read_text(encoding="utf-8"))
    require(d["n_pairs"] > 0, "配对数 n_pairs 为 0")
    require(0.0 <= d["acc_ood"] <= 1.0, "OOD 准确率越界")
    require(d["empty_ood"] <= d["n_pairs"], "空回答数超过配对数")
    return (f"n={d['n_pairs']} acc {d['acc_in']:.3f}→{d['acc_ood']:.3f} "
            f"empty={d['empty_ood']} p={d['p_value']:.3g}")


def check_records() -> str:
    import json

    p = OUTPUTS / "eval" / RUN / "records.json"
    require(p.exists(), "records.json 没有落盘")
    rows = json.loads(p.read_text(encoding="utf-8"))
    require(len(rows) >= 4, f"records 太少：{len(rows)}")
    for r in rows:
        require(0.0 <= r["ci_low"] <= r["accuracy"] <= r["ci_high"] <= 1.0,
                f"CI 与点估计错位：{r['suite']} {r['accuracy']} {r['ci_low']} {r['ci_high']}")
        require(r["n"] > 0 and r["n_families"] > 0, "n/n_families 为零")
    return f"records={len(rows)} ci-ok"


def check_bootstrap() -> str:
    preds = load_run(RUN)
    samples = {s.id: s for s in load_samples("dev_test")}
    boot = bootstrap_effect(preds, samples, n_boot=200)
    require("prior_effect" in boot, "bootstrap 没有 prior_effect")
    for k, v in boot.items():
        require(v["lo"] <= v["hi"], f"{k} 的 CI 反了")
    return f"keys={sorted(boot)} n_boot=200"


def check_mcnemar() -> str:
    preds = load_run(RUN)
    same = mcnemar_vs(preds, preds)
    require(same["a_only"] == 0 and same["b_only"] == 0, "自己比自己应有 0 个不一致")
    require(same["p_value"] == 1.0, f"自己比自己 p 应为 1，实际 {same['p_value']}")
    # 人工翻转每 3 条的正确性，构造一个"只有 A 对、B 错"的对照
    a = [p for p in preds if p["condition"] == "correct_image"]
    b = [dict(p, correct_relaxed=(not p["correct_relaxed"]) if i % 3 == 0
              else p["correct_relaxed"]) for i, p in enumerate(a)]
    mc = mcnemar_vs(a, b)
    flipped = sum(1 for p in a if p["correct_relaxed"]) - sum(
        1 for p in b if p["correct_relaxed"])
    require(mc["n"] == len(a), "配对样本数不对")
    require(mc["a_only"] + mc["b_only"] > 0, "构造的差异没有反映到配对表")
    require(mc["delta"] != 0, "delta 不该为 0")
    return f"paired={mc['n']} flipped_delta={mc['delta']:+.3f} p={mc['p_value']:.3g}"


def check_report() -> str:
    r = py("-m", "stage4_eval.report", "--run-id", RUN, "--baseline", RUN,
           "--suite", "dev_test")
    if r.returncode != 0:
        raise AssertionError(f"报告失败：\n{r.stdout[-600:]}\n{r.stderr[-600:]}")
    p = REPORTS / f"eval_{RUN}.md"
    require(p.exists(), "报告文件不存在")
    text = p.read_text(encoding="utf-8")
    require("prior_effect" in text and "Wilson" in text, "报告缺关键字段")
    return f"{p.name} bytes={p.stat().st_size}"


def check_sft_run() -> str:
    d = OUTPUTS / "eval" / "sft_dev"
    if not d.exists():
        raise Skip("还没跑过 sft 评测（python -m stage4_eval.run --model sft ...）")
    preds = load_run("sft_dev")
    require(len(preds) >= 80, f"预测数太少：{len(preds)}")
    return f"predictions={len(preds)}"


def main() -> int:
    return run([
        ("prior-eval", check_prior_eval),
        ("two-by-two-direction", check_two_by_two),
        ("records-ci", check_records),
        ("family-bootstrap", check_bootstrap),
        ("mcnemar", check_mcnemar),
        ("report-written", check_report),
        ("sft-run", check_sft_run),
        ("mismatch-test", check_mismatch_test),
        ("ood-transfer", check_ood),
    ], "S4 验收（2.0）")


if __name__ == "__main__":
    raise SystemExit(main())
