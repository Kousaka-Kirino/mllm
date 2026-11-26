"""HW1 · recall@k 与 MRR。

跑 `python -m stage7_memory.homework.hw1_recall_mrr` 会跑内置自测。
"""

from __future__ import annotations


def recall_at_k(ranked: list[str], gold: str, k: int) -> float:
    """gold 在 ranked 的前 k 名里 → 1.0，否则 0.0。

    边界：k<=0 抛 ValueError；空列表返回 0.0；重复 id 只算首次出现的位置。
    """
    raise NotImplementedError("HW1.1")


def mrr(ranked: list[str], gold: str) -> float:
    """1 / 首次命中名次（从 1 开始）；未命中 0.0；空列表 0.0。"""
    raise NotImplementedError("HW1.2")


def average_recall(runs: list[list[str]], golds: list[str], k: int) -> float:
    """一批查询的平均 recall@k；空输入 0.0；长度不等抛 ValueError。"""
    raise NotImplementedError("HW1.3")


def _selftest() -> None:
    ranked = ["p3", "p1", "p3", "p9"]
    assert recall_at_k(ranked, "p3", 3) == 1.0
    assert recall_at_k(ranked, "p1", 1) == 0.0
    try:
        recall_at_k(ranked, "p3", 0)
    except ValueError:
        pass
    else:
        raise AssertionError("k<=0 应抛 ValueError")
    assert recall_at_k([], "p3", 5) == 0.0
    assert mrr(ranked, "p3") == 1.0
    assert mrr(ranked, "p1") == 0.5
    assert mrr(ranked, "p9") == 0.25
    assert mrr([], "p3") == 0.0
    assert average_recall([["a", "b"], ["b", "c"]], ["a", "c"], 1) == 0.5
    assert average_recall([], [], 1) == 0.0
    try:
        average_recall([["a"]], [], 1)
    except ValueError:
        pass
    else:
        raise AssertionError("长度不等应抛 ValueError")
    print("HW1 selftest OK")


if __name__ == "__main__":
    _selftest()
