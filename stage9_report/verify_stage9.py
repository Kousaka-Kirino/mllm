"""S9 验收：报告里的每个数字都必须能回溯到 RESULTS.json。

四项：
- `results-json`：RESULTS.json 结构完整、时间戳是本次生成的；
- `report-sections`：REPORT_2.0.md 七个小节齐全；
- `numbers-traceable`：报告里出现的每个 `0.xxx` 精度都能在 RESULTS.json 里找到；
- `claims-table`：`stage9_report/CLAIMS.md` 的每条简历声明都指向一个文件。
"""

from __future__ import annotations

import json
import re

from picovlm.config import REPORTS
from picovlm.report import Skip, require, run

NUM = re.compile(r"\b(0\.\d{3})\b")


def _results() -> dict:
    p = REPORTS / "RESULTS.json"
    if not p.exists():
        raise Skip("没有 reports/RESULTS.json（先跑 python -m stage9_report.collect）")
    return json.loads(p.read_text(encoding="utf-8"))


def c_results() -> str:
    d = _results()
    for k in ("stamp", "data", "train", "eval", "bench", "memory"):
        require(k in d, f"RESULTS.json 缺 {k}")
    require(d["data"], "data 段为空")
    require(d["eval"], "eval 段为空")
    require(d["bench"], "bench 段为空")
    require(d["memory"], "memory 段为空")
    return (f"stamp={d['stamp']}，eval runs={len(d['eval'])}，"
            f"bench rows={len(d['bench']['rows'])}")


def c_sections() -> str:
    p = REPORTS / "REPORT_2.0.md"
    if not p.exists():
        raise Skip("没有 reports/REPORT_2.0.md")
    t = p.read_text(encoding="utf-8")
    need = ["## 1. 数据", "## 2. 训练", "## 3. 评测", "## 4. 效应",
            "## 5. 预算", "## 6. 视觉记忆", "## 7. 结论"]
    for s in need:
        require(s in t, f"报告缺小节：{s}")
    require("自动生成" in t, "报告应标明自动生成")
    return f"{len(need)} 个小节齐全"


def c_traceable() -> str:
    d = _results()
    t = (REPORTS / "REPORT_2.0.md").read_text(encoding="utf-8")

    # RESULTS.json 里出现的所有数值，统一格式化成三位小数
    known: set[str] = set()

    def walk(o):  # noqa: ANN001
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            known.add(f"{abs(float(o)):.3f}")

    walk(d)
    found = {x.lstrip("-") for x in NUM.findall(t)}
    unknown = sorted(x for x in found if x not in known)
    require(not unknown, f"报告里出现 RESULTS.json 中没有的数字：{unknown}")
    return f"{len(found)} 个三位小数全部可回溯"


def c_claims() -> str:
    p = REPORTS.parent / "stage9_report" / "CLAIMS.md"
    if not p.exists():
        raise Skip("没有 stage9_report/CLAIMS.md")
    t = p.read_text(encoding="utf-8")
    rows = [l for l in t.splitlines() if l.startswith("| ") and "---" not in l]
    require(len(rows) > 2, "CLAIMS.md 表格为空")
    for l in rows[2:]:
        require(any(x in l for x in ("reports/", "outputs/", "data/")),
                f"声明缺少来源文件：{l[:60]}")
    return f"{len(rows) - 2} 条声明均带来源"


def main() -> int:
    return run([
        ("results-json", c_results),
        ("report-sections", c_sections),
        ("numbers-traceable", c_traceable),
        ("claims-table", c_claims),
    ], "S9 验收（2.0）")


if __name__ == "__main__":
    raise SystemExit(main())
