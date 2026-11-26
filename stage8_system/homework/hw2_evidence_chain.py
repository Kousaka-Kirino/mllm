"""HW2：通用"证据链"检查。

背景：简历上写"模型在一致格 95%、冲突格 0%"，这个 95% 必须能追到
`outputs/eval/<run>/records.json` 里的某个字段。S9 的 `numbers-traceable`
只覆盖了 `RESULTS.json`；这个作业要求你把检查**通用化**。

输入：
  markdown 文件（报告或简历条目）+ 一组 json 文件（证据）

规则：markdown 里每个形如 `0.xxx` 的数，必须能在证据 json 里
找到"格式化成三位小数后相等"的值，否则算不可追溯。

要做的事：
  1. 实现 `load_numbers(paths)`：递归把 json 里所有数值收进集合（三位小数）。
  2. 实现 `trace(md_text, known)`：返回 (可追溯集合, 不可追溯集合)。
  3. 实现 `check_report(md_path, evidence_paths)`：打印不可追溯的数，
     并返回是否通过（不可追溯为空才算通过）。

验收：
  python homework/hw2_evidence_chain.py
  → 用 reports/REPORT_2.0.md + reports/RESULTS.json 跑一遍，应打印 PASS

提示见 HW8_提示.md。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from picovlm.config import REPORTS, ROOT

NUM = re.compile(r"\b(\d+\.\d{3})\b")


def load_numbers(paths) -> set[str]:
    """把 json 里所有数值格式化成三位小数，收进一个集合。

    TODO:
      - 支持任意嵌套（dict/list）
      - 跳过 bool（True 也是 int！）
      - 用 abs() 后格式化，这样 -0.042 与 0.042 都能匹配（报告里带负号）
    """
    raise NotImplementedError


def trace(md_text: str, known: set[str]) -> tuple[set[str], set[str]]:
    """返回 (可追溯, 不可追溯)。记得去掉前导负号再比对。"""
    raise NotImplementedError


def check_report(md_path: Path, evidence_paths) -> bool:
    """打印结果并返回是否通过。"""
    raise NotImplementedError


def main() -> int:
    md = REPORTS / "REPORT_2.0.md"
    ev = [REPORTS / "RESULTS.json"]
    if not md.exists():
        print("先跑 stage9_report.collect 生成报告")
        return 2
    try:
        ok = check_report(md, ev)
    except NotImplementedError:
        print("TODO: 还没实现")
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
