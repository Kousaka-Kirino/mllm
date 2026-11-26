"""S8：一键验收矩阵。

    python -m stage8_system.run_all            # 跑全部阶段
    python -m stage8_system.run_all --only 1 4 # 只跑 S1、S4

把每个阶段 verify 的输出压成一张表，落 `outputs/acceptance.md`。
任何一个 FAIL，退出码为 1 —— CI（或复试时当着老师的面）可以直接跑。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time

from picovlm.config import OUTPUTS
from picovlm.io_utils import write_json

STAGES: list[tuple[str, str, str]] = [
    ("S0", "stage0_env.verify_stage0", "环境与契约"),
    ("S1", "stage1_data.verify_stage1", "数据引擎"),
    ("S2", "stage2_model.verify_stage2", "模型骨架"),
    ("S3", "stage3_train.verify_stage3", "两阶段训练"),
    ("S4", "stage4_eval.verify_stage4", "2×2 诊断"),
    ("S5", "stage5_intervene.verify_stage5", "课程/反事实/GRPO"),
    ("S6", "stage6_ablation.verify_stage6", "token 预算"),
    ("S7", "stage7_memory.verify_stage7", "视觉记忆"),
    ("S8", "stage8_system.verify_stage8", "系统与交付"),
    ("S9", "stage9_report.verify_stage9", "结果收集与报告"),
]

COUNT_RE = re.compile(r"-- PASS (\d+) / FAIL (\d+) / SKIP (\d+) --")


def run_stage(module: str) -> tuple[int, int, int, str]:
    proc = subprocess.run([sys.executable, "-m", module], capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    m = COUNT_RE.search(out)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3)), out
    return 0, 1 if proc.returncode else 0, 0, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只跑这些阶段号，如 1 4")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    wanted = {f"S{x}" for x in args.only} if args.only else {s[0] for s in STAGES}
    rows, t0 = [], time.time()
    for stage, module, title in STAGES:
        if stage not in wanted:
            continue
        p, f, s, out = run_stage(module)
        rows.append({"stage": stage, "module": module, "title": title,
                     "pass": p, "fail": f, "skip": s})
        print(f"{stage} {title:<12} PASS {p:2d} / FAIL {f} / SKIP {s}")
        if f and not args.quiet:
            tail = "\n".join(out.strip().splitlines()[-12:])
            print(f"  ↳ {module}\n" + "\n".join("    " + l for l in tail.splitlines()))

    total = {k: sum(r[k] for r in rows) for k in ("pass", "fail", "skip")}
    md = ["# 验收矩阵", "",
          f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}，耗时 {time.time() - t0:.1f}s", "",
          "| 阶段 | 内容 | PASS | FAIL | SKIP |", "|---|---|---|---|---|"]
    md += [f"| {r['stage']} | {r['title']} | {r['pass']} | {r['fail']} | {r['skip']} |"
           for r in rows]
    md += ["", f"**合计：{total['pass']} PASS / {total['fail']} FAIL / "
               f"{total['skip']} SKIP**", ""]

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "acceptance.md").write_text("\n".join(md), encoding="utf-8")
    write_json(OUTPUTS / "acceptance.json", {"rows": rows, "total": total})
    print("\n".join(md[-2:]))
    print(f"-> {OUTPUTS / 'acceptance.md'}")
    return 1 if total["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
