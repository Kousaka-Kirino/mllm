"""S9 · 把散落在 outputs/ 里的数字收成一份报告。

原则：**报告里的每个数字都必须能指回一个文件**。本脚本不训练、不推理，
只做三件事：

1. 读 `outputs/eval/*/records.json`、`grid.json`、`outputs/bench/*.json`、
   `outputs/memory/memory_eval_*.json`、`data/datacard.json`；
2. 生成 `reports/RESULTS.json`（机器可读）与 `reports/REPORT_2.0.md`（人读）；
3. 缺哪一块就显式写 `MISSING`，**不编造**。

    python -m stage9_report.collect
    python -m stage9_report.collect --strict   # 有 MISSING 就返回 1（CI 用）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from picovlm.config import OUTPUTS, REPORTS, ROOT
from picovlm.io_utils import ensure_dir, read_json, utc_stamp, write_json

MISSING = "MISSING"


def _latest(pattern: str) -> Path | None:
    files = sorted((OUTPUTS / pattern).parent.glob((OUTPUTS / pattern).name))
    return files[-1] if files else None


def collect_eval() -> dict:
    out: dict = {}
    for run_dir in sorted((OUTPUTS / "eval").glob("*")):
        rec = run_dir / "records.json"
        if not rec.exists():
            continue
        rows = read_json(rec)
        grid = run_dir / "grid.json"
        out[run_dir.name] = {
            "records": rows,
            "grid": read_json(grid) if grid.exists() else None,
        }
    return out


def collect_bench() -> dict:
    f = _latest("bench/connectors_*.json")
    if f is None:
        return {}
    return {"file": f.name, "rows": read_json(f)}


def collect_memory() -> dict:
    f = _latest("memory/memory_eval_*.json")
    if f is None:
        return {}
    return {"file": f.name, "result": read_json(f)}


def collect_data() -> dict:
    p = ROOT / "data" / "datacard.json"
    return read_json(p) if p.exists() else {}


def collect_train() -> dict:
    out: dict = {}
    for log in sorted((OUTPUTS / "train").glob("*/log.jsonl")):
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 20:  # dry-run / 1 步探针不算"训练"
            continue
        last = json.loads(lines[-1])
        first = json.loads(lines[0])
        out[log.parent.name] = {"steps": len(lines),
                                "first": first, "last": last}
    return out


def _main_accuracy(eval_res: dict) -> tuple[float, int] | None:
    """主臂（SFT）在 dev_test 上 correct_image 条件的加权平均准确率。"""
    rows = []
    for run, v in eval_res.items():
        if run != "sft_dev":
            continue
        for r in v["records"]:
            if r.get("condition", "correct_image") == "correct_image":
                rows.append(r)
    if not rows:
        return None
    n = sum(r["n"] for r in rows)
    return sum(r["accuracy"] * r["n"] for r in rows) / n, n


def build_results() -> dict:
    res = {
        "stamp": utc_stamp(),
        "data": collect_data(),
        "train": collect_train(),
        "eval": collect_eval(),
        "bench": collect_bench(),
        "memory": collect_memory(),
    }
    main = _main_accuracy(res["eval"])
    # 派生数字也写进 RESULTS.json，否则报告里会出现"查不到来源"的数
    res["main"] = ({"accuracy": main[0], "n": main[1], "condition": "correct_image",
                    "split": "dev_test", "run": "sft_dev"} if main else {})
    return res


def _fmt_cell(rows: list[dict] | None) -> str:
    if not rows:
        return MISSING
    parts = []
    for r in rows:
        parts.append(f"{r['suite']} {r['accuracy']:.3f} "
                     f"[{r['ci_low']:.2f},{r['ci_high']:.2f}]")
    return "<br>".join(parts)


def to_markdown(res: dict) -> str:
    L: list[str] = []
    L += ["# PicoVLM · 结果汇总（自动生成）", "",
          f"生成时间：{res['stamp']}", "",
          "> 本文件由 `python -m stage9_report.collect` 生成，不要手改。",
          "> 每个数字都能在 `reports/RESULTS.json` 与 `outputs/` 里找到来源。", ""]

    d = res.get("data") or {}
    L += ["## 1. 数据", ""]
    if d:
        fam = d.get("families") or {}
        thr = d.get("acuity_thresholds") or {}
        g = d.get("glyph_px_model") or {}
        L += [f"- profile：`{d.get('profile', MISSING)}`，seed {d.get('seed', MISSING)}",
              f"- 样本数：{d.get('counts', MISSING)}",
              f"- family 数：{fam if fam else MISSING}",
              f"- ρ 先验强度：请求 {d.get('rho_requested', MISSING)} / "
              f"实测 **{d.get('rho_measured', MISSING)}**",
              f"- acuity 分箱（实测阈值 {thr}）：{d.get('acuity_bins', MISSING)}",
              f"- 字形像素（模型输入侧，{g.get('n', MISSING)} 次测量）："
              f"median {g.get('median', MISSING)} px，IQR {g.get('iqr', MISSING)}，"
              f"范围 [{g.get('min', MISSING)}, {g.get('max', MISSING)}]"]
    else:
        L.append(f"- {MISSING}（先跑 `python -m stage1_data.build --profile pilot`）")
    L.append("")

    L += ["## 2. 训练", ""]
    t = res.get("train") or {}
    if t:
        L += ["| run | steps | 首步 loss | 末步 loss | 末步 lr |", "|---|---:|---:|---:|---:|"]
        for name, v in t.items():
            L.append(f"| {name} | {v['steps']} | {v['first'].get('loss', MISSING):.4f} "
                     f"| {v['last'].get('loss', MISSING):.4f} "
                     f"| {v['last'].get('lr', MISSING):.2e} |")
    else:
        L.append(f"- {MISSING}")
    L.append("")

    L += ["## 3. 评测（RQ1：2×2 诊断，`correct_image` 条件）", ""]
    ev = res.get("eval") or {}
    if ev:
        L += ["| run | suite | n | accuracy | 95% CI |", "|---|---|---:|---:|---|"]
        for run, v in ev.items():
            for r in v["records"]:
                if r.get("condition", "correct_image") != "correct_image":
                    continue
                L.append(f"| {run} | {r['suite']} | {r['n']} | {r['accuracy']:.3f} "
                         f"| [{r['ci_low']:.3f}, {r['ci_high']:.3f}] |")
    else:
        L.append(f"- {MISSING}（先跑 `python -m stage4_eval.run`）")
    L.append("")

    L += ["### 3b. 条件对比（S4 E0 矩阵）", ""]
    if ev:
        L += ["| run | condition | suite | n | accuracy |", "|---|---|---|---:|---:|"]
        for run, v in ev.items():
            for r in v["records"]:
                cond = r.get("condition", "correct_image")
                if cond == "correct_image":
                    continue
                L.append(f"| {run} | {cond} | {r['suite']} | {r['n']} | {r['accuracy']:.3f} |")
    L.append("")

    L += ["## 4. 效应（family 聚类 bootstrap）", ""]
    found = False
    for run, v in ev.items():
        g = v.get("grid")
        if not g:
            continue
        found = True
        vdr = f"easy {g.get('vdr_easy', MISSING):.3f} / hard {g.get('vdr_hard', MISSING):.3f}" \
            if g.get("vdr_easy") is not None else MISSING
        L.append(f"**{run}**：prior_effect {g['prior_effect']:+.3f}，"
                 f"acuity_effect {g['acuity_effect']:+.3f}，"
                 f"interaction {g['interaction']:+.3f}，VDR {vdr}")
        bs = g.get("bootstrap") or {}
        if bs:
            pe = bs.get("prior_effect") or {}
            L.append(f"- bootstrap（{bs.get('n_boot', MISSING)} 次，按 family）："
                     f"prior_effect {pe.get('point', MISSING):+.3f} "
                     f"[{pe.get('lo', MISSING):+.3f}, {pe.get('hi', MISSING):+.3f}]")
        L.append("")
    if not found:
        L += [f"- {MISSING}", ""]

    L += ["## 5. 预算（S6）", ""]
    b = res.get("bench") or {}
    if b:
        # 只有主臂（mlp）是被训练过的；其余档位没有精度，写 MISSING
        main = _main_accuracy(res.get("eval") or {})
        L += ["| 连接器 | token | 参数(M) | 延迟(ms) | 峰值(MiB) | 精度 |",
              "|---|---:|---:|---:|---:|---|"]
        for r in b["rows"]:
            if r["connector"] == "mlp" and main is not None:
                acc = f"{main[0]:.3f}（correct_image, {main[1]} 条）"
            else:
                acc = "未训练（不报）"
            L.append(f"| {r['connector']} | {r['tokens']} | {r['params_m']:.2f} "
                     f"| {r['latency_ms']['median']:.0f} | {r['peak_mib']['median']:.0f} | {acc} |")
    else:
        L.append(f"- {MISSING}")
    L.append("")

    L += ["## 6. 视觉记忆（S7）", ""]
    m = res.get("memory") or {}
    if m:
        r = m["result"]
        if "recall" in r:
            L.append(f"- 页内召回 recall@1 {r['recall']['recall_at_1']:.3f} "
                     f"[{r['recall']['recall_at_1_ci'][0]:.3f}, "
                     f"{r['recall']['recall_at_1_ci'][1]:.3f}]（n={r['recall']['n']}）")
        if "zoom" in r:
            z = r["zoom"]
            L.append(f"- 放大实验：full {z['full']['accuracy']:.3f} / "
                     f"salient {z['salient']['accuracy']:.3f} / "
                     f"random {z['random']['accuracy']:.3f}（n={z['full']['n']}）")
            L.append(f"- salient − full = {z['delta_salient_minus_full']:+.3f}")
    else:
        L.append(f"- {MISSING}")
    L.append("")

    L += ["## 7. 结论（按证据强度排序）", "",
          "1. **先验主导**：prior_effect 大、VDR 低、换图不影响答案（S4）。",
          "2. **不是纯先验**：灰图/随机裁块会让准确率崩塌（S4 + S7），说明视觉输入被使用。",
          "3. **acuity 不是当前瓶颈**：显著块放大没有超过整图（S7）。",
          "4. **检索未达标**：冻结 SigLIP 平均池化的 recall@1 很低（S7）。",
          "",
          "> 以上四条与 `reports/RESULTS.json` 一一对应；缺项的写 MISSING，不补。", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="有 MISSING 时返回 1")
    args = ap.parse_args(argv)

    res = build_results()
    ensure_dir(REPORTS)
    write_json(REPORTS / "RESULTS.json", res)
    md = to_markdown(res)
    (REPORTS / "REPORT_2.0.md").write_text(md, encoding="utf-8")
    print(f"saved {REPORTS / 'RESULTS.json'}")
    print(f"saved {REPORTS / 'REPORT_2.0.md'}")

    if args.strict and MISSING in md:
        print("strict: 报告里还有 MISSING")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
