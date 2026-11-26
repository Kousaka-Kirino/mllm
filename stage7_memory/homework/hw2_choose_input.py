"""HW2 · 选输入策略：整图 / 显著块 / 随机块，该用哪个？

本实验的真实情况（n=24，同一批 hard 题）：
    full 0.583, salient 0.542, random 0.292
最大差（salient vs full）= -0.042，**不显著** → 正确结论是 "tie"，不是 "full 更好"。

这正是作业的核心：**不要用两个独立 Wilson 区间是否重叠来判断**（那太保守），
也不要用点估计比大小（那太激进），要用配对比较。

跑 `python -m stage7_memory.homework.hw2_choose_input` 会跑内置自测。
"""

from __future__ import annotations

PAIRS = {"full_vs_salient": (14, 13), "full_vs_random": (14, 7), "salient_vs_random": (13, 7)}
N = 24


def choose_input(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int],
                 n: int, alpha: float = 0.05) -> str:
    """a/b/c = (正确数, 总数)，分别对应 full / salient / random。

    返回 "full" | "salient" | "random" | "tie"。

    要求：
    - 用配对 McNemar（`picovlm.stats.mcnemar_exact`）比较"最好的两个"；
    - 若不显著 → "tie"；
    - 输入校验：三项 n 必须相同，否则抛 ValueError。
    """
    raise NotImplementedError("HW2")


def _selftest() -> None:
    assert choose_input((14, 24), (13, 24), (7, 24), 24) == "tie"
    assert choose_input((20, 24), (8, 24), (7, 24), 24) == "full"
    assert choose_input((7, 24), (8, 24), (20, 24), 24) == "random"
    try:
        choose_input((1, 10), (1, 20), (1, 10), 10)
    except ValueError:
        pass
    else:
        raise AssertionError("n 不一致应抛 ValueError")
    print("HW2 selftest OK")


if __name__ == "__main__":
    _selftest()
