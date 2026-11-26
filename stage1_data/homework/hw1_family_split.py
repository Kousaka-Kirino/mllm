"""HW1：把一个 family 列表按比例切到四个 split，并保证不泄漏。

为什么作业是这个：数据划分只有几十行，但它决定了"测试成绩"有没有意义。
自己写一遍，你就知道 `sklearn.train_test_split` 为什么不能直接用（它会
把同一张表的变体切到两边）。
"""

from __future__ import annotations

from typing import Sequence


def split_by_family(families: Sequence[str], ratios: Sequence[float],
                    seed: int = 17) -> dict[str, list[str]]:
    """把 family 按 ratios 切分。

    参数
    ----
    families : 唯一 family id 的序列（调用方保证唯一）
    ratios   : 四个比例，和为 1，顺序对应 train/val/dev_test/locked_test
    seed     : 随机种子，同 seed 必须同结果

    返回
    ----
    {"train": [...], "val": [...], "dev_test": [...], "locked_test": [...]}

    要求
    ----
    1. 每个 family 恰好出现一次；
    2. 各 split 的实际比例与目标比例误差 ≤ 0.02；
    3. 同 seed 可复现；不修改入参。
    """
    raise NotImplementedError("TODO: 用 default_rng(seed) 洗牌后按最大余数法切分")


if __name__ == "__main__":
    fams = [f"fam-{i:04d}" for i in range(120)]
    out = split_by_family(fams, (0.8, 0.1, 0.05, 0.05), seed=17)
    for k, v in out.items():
        print(f"{k:12s} {len(v):3d}  {len(v) / len(fams):.3f}")
