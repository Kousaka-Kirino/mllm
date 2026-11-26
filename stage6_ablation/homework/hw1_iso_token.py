"""HW1 · iso-token 与边际收益。

目标：把"加 token 值不值"变成一个能算的数，而不是一句感觉。

必做：
1. iso_token —— 在固定预算下找出可比的点（同算力对比的操作化定义）
2. marginal   —— 每 1000 token 买到多少精度
3. verdict    —— 给定预算下的建议，判据写进 docstring

提示：`stage6_ablation/pareto.py` 里有 `Point` 与 `iso_token` 的参考实现思路，
但**别直接抄**——先自己想"预算落在两个采样点之间时该怎么办"。

跑 `python -m stage6_ablation.homework.hw1_iso_token` 会跑内置自测。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Point:
    name: str
    tokens: int
    accuracy: float


POINTS = [Point("resampler", 64, 0.20), Point("pixelshuffle", 144, 0.30),
          Point("mlp", 576, 0.38)]


def iso_token(points: list[Point], budget: int, tol: float = 0.05) -> list[Point]:
    """返回 token 数在 [budget*(1-tol), budget*(1+tol)] 内的点，按 token 升序。

    边界：tol=0 时是精确匹配；结果可能为空——空不是错误，是"这个预算没测过"。
    """
    raise NotImplementedError("HW1.1")


def marginal(points: list[Point]) -> list[tuple[str, str, float]]:
    """相邻两点的边际收益：[(from, to, acc_per_1k_tokens), ...]，按 token 升序。

    `Δtokens=0` 的两点不合法——遇到就抛 ValueError。
    """
    raise NotImplementedError("HW1.2")


def verdict(points: list[Point], budget: int, min_gain: float = 0.05) -> str:
    """在给定预算下给建议：'keep' / 'add' / 'no-data'。

    判据（自己定，但要写清楚）：比如"预算升到下一档，若每 1000 token 至少换来
    min_gain 点精度，就 add"。
    """
    raise NotImplementedError("HW1.3")


def _selftest() -> None:
    pts = POINTS
    assert [p.name for p in iso_token(pts, 144, tol=0.0)] == ["pixelshuffle"]
    assert [p.name for p in iso_token(pts, 150, tol=0.05)] == ["pixelshuffle"]
    assert iso_token(pts, 1000, tol=0.05) == []
    m = marginal(pts)
    assert len(m) == 2, m
    assert m[0][2] > 0 and m[1][2] > 0, "精度应随 token 单调上升"
    assert m[0][2] > m[1][2], "边际收益应递减（若不然，说明你的表不合理）"
    try:
        marginal([Point("a", 10, 0.1), Point("b", 10, 0.2)])
    except ValueError:
        pass
    else:
        raise AssertionError("Δtoken=0 应抛 ValueError")
    assert verdict(pts, 576) in {"keep", "add", "no-data"}
    print("HW1 selftest OK")


if __name__ == "__main__":
    _selftest()
