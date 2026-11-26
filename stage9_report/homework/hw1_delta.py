"""HW1：两次实验结果对比（Δaccuracy + 显著性粗判）。

为什么需要：你改了 prompt / 数据 / 学习率之后，最想知道的是"这 3 个点
是真的提升还是噪声"。单看两个 accuracy 无法回答——必须看置信区间。

要做的事：
  1. `load_runs(path)`：从 RESULTS.json 里抽出 {(run, suite): {"accuracy", "ci_low", "ci_high", "n"}}
  2. `delta_table(prev, cur)`：对两边都有的 key 算 Δ，并判断
     `|Δ| > (cur 半宽 + prev 半宽)` 才算"超出噪声"（保守判据）
  3. `render(rows)`：输出 markdown 表

验收：
  python homework/hw1_delta.py --prev reports/RESULTS.json --cur reports/RESULTS.json
  → 所有 Δ = 0，且没有一项被标"显著"（同一个文件跟自己比）

提示见 HW9_提示.md。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from picovlm.config import REPORTS


def load_runs(path: Path) -> dict:
    """返回 {(run, suite): {...}}。

    TODO:
      - RESULTS.json 里 eval 可能是 list[dict] 或 dict[str, dict]，两种都兼容
      - 只保留有 accuracy 与 ci_low/ci_high 的项
    """
    raise NotImplementedError


def delta_table(prev: dict, cur: dict) -> list[dict]:
    """对共同 key 算 Δ 与 significant 标记。

    TODO:
      - half_width = (ci_high - ci_low) / 2
      - significant = abs(delta) > prev_hw + cur_hw
      - 按 |delta| 降序排
    """
    raise NotImplementedError


def render(rows: list[dict]) -> str:
    """markdown 表：| run | suite | prev | cur | Δ | 显著 |"""
    raise NotImplementedError


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", default=str(REPORTS / "RESULTS.json"))
    ap.add_argument("--cur", default=str(REPORTS / "RESULTS.json"))
    args = ap.parse_args()
    try:
        rows = delta_table(load_runs(Path(args.prev)), load_runs(Path(args.cur)))
    except NotImplementedError:
        print("TODO: 还没实现")
        return 1
    print(render(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
