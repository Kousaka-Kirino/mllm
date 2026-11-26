"""HW4-2 · 逐条 bootstrap vs 整簇 bootstrap。"""

from __future__ import annotations

import random


def make_clustered(n_families: int = 40, per_family: int = 2):
    """每族两条，正确性相同：同一族的样本高度相关。"""
    fams = []
    rng = random.Random(0)
    for _ in range(n_families):
        correct = rng.random() < 0.5
        fams.append([correct] * per_family)
    return fams


def _ci(samples, n_boot=2000, seed=7):
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        pick = [samples[rng.randrange(len(samples))] for _ in samples]
        means.append(sum(pick) / len(pick))
    means.sort()
    return means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]


def compare(n_families: int = 40, per_family: int = 2) -> tuple[float, float]:
    """返回 (逐条 CI 宽度, 整簇 CI 宽度)。

    TODO: 逐条 —— 把所有 0/1 摊平成 n_families*per_family 个样本再重采样；
          整簇 —— 每次抽 n_families 个 family，把族内样本展开计算均值。
    """
    raise NotImplementedError("HW4-2")


def main() -> int:
    per_item, cluster = compare()
    print(f"per-item width {per_item:.3f}   cluster width {cluster:.3f}")
    print(f"cluster wider? {cluster > per_item}")
    return 0 if cluster > per_item else 1


if __name__ == "__main__":
    raise SystemExit(main())
