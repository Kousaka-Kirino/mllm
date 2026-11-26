"""HW5-1 · 手推组内优势。

目标：把 `[0,0,0,1]` 的优势算成 `[-0.577,-0.577,-0.577,1.732]`。

跑：python -m stage5_intervene.homework.hw1_advantage
"""

from __future__ import annotations

import math


def advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    """TODO: A_i = (r_i - mean) / (std + eps)；整组同分时返回全 0。

    注意 std 用的是**总体标准差**（除以 n，不是 n-1）——GRPO 原文如此，
    而且组是"整组已知"的，不是从总体里抽样。
    """
    raise NotImplementedError("HW5-1")


def main() -> int:
    got = advantages([0.0, 0.0, 0.0, 1.0])
    want = [-0.577, -0.577, -0.577, 1.732]
    ok = len(got) == 4 and all(abs(round(g, 3) - w) < 1e-3
                               for g, w in zip(got, want))
    print([round(g, 3) for g in got])
    print("hw5-1 PASS" if ok else "hw5-1 FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
