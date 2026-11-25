"""S6：token 预算的 Pareto 分析。

三个问题，一张表回答：
1. 给定预算 B，哪种连接器最准？（iso-token）
2. 精度-预算的拐点在哪？（Pareto 前沿）
3. 自适应策略（简单图少 token、难图多 token）比固定预算强多少？

本文件只做**表格与前沿计算**，具体数字由 `stage6_budget/run_budget.py`
用真模型或 policy 填进来。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Point:
    name: str
    tokens: int
    accuracy: float


def pareto_front(points: Iterable[Point]) -> list[Point]:
    """token 越少越好、精度越高越好；去掉被支配的点。按 token 升序返回。"""
    pts = sorted(points, key=lambda p: (p.tokens, -p.accuracy))
    front: list[Point] = []
    best = -1.0
    for p in pts:
        if p.accuracy > best:
            front.append(p)
            best = p.accuracy
    return front


def iso_token(points: Sequence[Point], budget: int, tol: float = 0.0) -> Point | None:
    """预算内精度最高的点（token <= budget*(1+tol)）。"""
    cands = [p for p in points if p.tokens <= budget * (1 + tol)]
    return max(cands, key=lambda p: p.accuracy) if cands else None


def gains(points: Sequence[Point]) -> dict:
    """相对"最贵档"的省 token 比例，以及相对"最便宜档"的精度提升。"""
    if not points:
        return {}
    by_tok = sorted(points, key=lambda p: p.tokens)
    cheap, dear = by_tok[0], by_tok[-1]
    return {
        "cheapest": cheap.name,
        "most_expensive": dear.name,
        "token_saving": 1 - cheap.tokens / dear.tokens if dear.tokens else 0.0,
        "accuracy_drop": dear.accuracy - cheap.accuracy,
    }


def to_markdown(rows: Sequence[Point], front: Sequence[Point]) -> str:
    lines = ["# S6 视觉 token 预算", "",
             "| 配置 | 视觉 token | 准确率 | Pareto |",
             "|---|---|---|---|"]
    front_names = {p.name for p in front}
    for p in sorted(rows, key=lambda p: p.tokens):
        lines.append(f"| {p.name} | {p.tokens} | {p.accuracy:.1%} | "
                     f"{'✓' if p.name in front_names else ''} |")
    lines += ["", "## 前沿", ""]
    for p in front:
        lines.append(f"- {p.name}: {p.tokens} token, {p.accuracy:.1%}")
    return "\n".join(lines) + "\n"
