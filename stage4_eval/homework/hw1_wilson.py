"""HW4-1 · 自己实现 Wilson 区间。"""

from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """返回 (lo, hi)。

    TODO: 用 README 里的公式实现；n == 0 时返回 (0.0, 1.0)；
          最后把 lo/hi 夹到 [0,1]。
    """
    raise NotImplementedError("HW4-1")


def main() -> int:
    from picovlm.stats import wilson_ci

    cases = [(0, 20), (1, 20), (10, 20), (20, 20), (0, 0)]
    for k, n in cases:
        got, want = wilson(k, n), wilson_ci(k, n, 1.96)
        if abs(got[0] - want[0]) > 1e-9 or abs(got[1] - want[1]) > 1e-9:
            print(f"hw4-1 FAIL k={k} n={n}: {got} != {want}")
            return 1
    print("hw4-1 PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
