"""HW1：给 run_all 加"只跑指定阶段"。

为什么需要：改完共享包 `picovlm/` 之后，你只想确认 S3/S4 没被你改坏，
不想等 S0 的联网检查和 S7 的视觉塔加载（合起来要几分钟）。

要做的事：
  1. 在 `stage8_system/run_all.py` 的 main 里加 `--only S3,S4`（逗号分隔）；
  2. 解析时不存在的阶段名要报错退出（不要静默忽略）；
  3. 过滤后若为空，报错退出。
  4. 汇总的 `total` 只统计实际跑了的阶段。

验收：
  python -m stage8_system.run_all --only S4
  → 只看到 S4 一行，退出码 = S4 的退出码

提示见 HW8_提示.md。
"""

from __future__ import annotations

import sys

from picovlm.config import ROOT
from picovlm.report import require


def filter_stages(stages: list[tuple], only: str | None) -> list[tuple]:
    """返回过滤后的阶段列表。

    TODO:
      - only 为 None 时原样返回
      - 否则按逗号切分，去空白，大小写不敏感（s3 → S3）
      - 保留 STAGES 原有顺序（不要按用户输入顺序）
      - 出现未知阶段名 → ValueError，消息里带上未知名字
    """
    raise NotImplementedError


def check() -> None:
    stages = [("S0", "stage0_env.verify_stage0", "环境"),
              ("S3", "stage3_train.verify_stage3", "训练"),
              ("S4", "stage4_eval.verify_stage4", "评测")]
    require([s[0] for s in filter_stages(stages, None)] == ["S0", "S3", "S4"], "None 应原样返回")
    require([s[0] for s in filter_stages(stages, "s3")] == ["S3"], "大小写不敏感")
    require([s[0] for s in filter_stages(stages, "S4,S0")] == ["S0", "S4"], "应保留原顺序")
    try:
        filter_stages(stages, "S9")
    except ValueError as e:
        require("S9" in str(e), "错误消息里应包含未知阶段名")
    else:
        raise AssertionError("未知阶段应抛 ValueError")
    print("PASS: filter_stages 行为正确")


if __name__ == "__main__":
    try:
        check()
    except NotImplementedError:
        print("TODO: 还没实现 filter_stages")
        sys.exit(1)
