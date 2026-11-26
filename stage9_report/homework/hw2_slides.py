"""HW2：从 RESULTS.json 生成 6 页幻灯片大纲（纯 markdown）。

为什么是 markdown 而不是 pptx：大纲是"讲什么"的决策，pptx 是"长什么样"的
执行。先定大纲，再交给模板，顺序反了会为了排版删掉本该讲的负结果。

要做的事：
  1. `outline(res)` 返回 list[dict]，每项 {"title": str, "bullets": [str, ...]}
     共 6 页：问题 / 方法 / 主结果(2×2) / 干预 / 负结果 / 结论
  2. `to_markdown(pages)` 渲染成 `## slide` + `- bullet`
  3. 数字必须从 res 里取，**不允许硬编码**（和 S9 的可追溯性一致）

验收：
  python homework/hw2_slides.py
  → 输出 6 个 `## ` 标题，且第 3 页含 "0.950" 或 "0.95"

提示见 HW9_提示.md。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from picovlm.config import REPORTS


def outline(res: dict) -> list[dict]:
    """6 页大纲。

    TODO:
      - 主结果页从 res["eval"] 里找 sft 的四个格（easy/hard × consistent/conflict）
      - 负结果页用 res["memory"]["recall_at_1"] 与 res.get("zoom", {})
      - 结论页最多 3 条，且必须有一条是"我们没做到什么"
    """
    raise NotImplementedError


def to_markdown(pages: list[dict]) -> str:
    """TODO: `## title` + `- bullet` 拼接。"""
    raise NotImplementedError


def main() -> int:
    p = REPORTS / "RESULTS.json"
    if not p.exists():
        print("先跑 stage9_report.collect")
        return 2
    res = json.loads(p.read_text(encoding="utf-8"))
    try:
        md = to_markdown(outline(res))
    except NotImplementedError:
        print("TODO: 还没实现")
        return 1
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
