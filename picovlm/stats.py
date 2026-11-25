"""统计口径：Wilson 区间、bootstrap 区间、McNemar 精确检验、配对比较。

沿用 PandaVLA 的规矩：**每个比例都带 95% 区间**，每个"变好了"都带配对胜负平。
样本量小的时候 Wilson 比正态近似稳；配对比较用精确二项检验（不用卡方近似）。
"""

from __future__ import annotations

import math
import random
from typing import Iterable, Sequence


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """二项比例的 Wilson 得分区间。n=0 时返回 (0, 1)。"""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denom
    lo = 0.0 if centre - half < 0 else centre - half
    hi = 1.0 if centre + half > 1 else centre + half
    if k <= 0:
        lo = 0.0
    if k >= n:
        hi = 1.0
    return lo, hi


def bootstrap_ci(values: Sequence[float], stat: str = "mean", n_boot: int = 2000,
                 seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """对任意标量统计量做 percentile bootstrap。"""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = list(values)
    n = len(vals)
    fn = (lambda xs: sum(xs) / len(xs)) if stat == "mean" else (lambda xs: sorted(xs)[len(xs) // 2])
    boots = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(fn(sample))
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return lo, hi


def mcnemar_exact(n_a_only: int, n_b_only: int) -> float:
    """McNemar 精确检验的双侧 p 值（只用不一致对）。"""
    n = n_a_only + n_b_only
    if n == 0:
        return 1.0
    k = min(n_a_only, n_b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_compare(a: Iterable[bool], b: Iterable[bool]) -> dict:
    """两个模型在同一批样本上的配对比较。

    返回 wins（a 对 b 错）/ losses（a 错 b 对）/ ties / p 值 / 净胜率差。
    """
    wins = losses = ties = 0
    for x, y in zip(a, b):
        if x and not y:
            wins += 1
        elif y and not x:
            losses += 1
        else:
            ties += 1
    total = wins + losses + ties
    return {
        "n": total,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "p_value": mcnemar_exact(wins, losses),
        "delta": (wins - losses) / total if total else 0.0,
    }


def summarize(correct: Sequence[bool]) -> dict:
    n = len(correct)
    k = sum(1 for c in correct if c)
    lo, hi = wilson_ci(k, n)
    return {"n": n, "k": k, "accuracy": (k / n if n else 0.0),
            "ci_low": lo, "ci_high": hi}
