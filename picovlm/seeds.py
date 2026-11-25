"""随机种子。三条规矩：

1. 每个阶段用 `rng_for(stage, seed)` 拿独立的 numpy Generator，
   不要用全局 `np.random`——否则改一个阶段的调用顺序会污染别的阶段。
2. 数据生成、bootstrap、dataloader 的种子分开，报告里写清楚。
3. `seed_everything` 只在训练入口调用一次。
"""

from __future__ import annotations

import os
import random

import numpy as np

# 冻结的种子区间（写进 ENVIRONMENT.md，改区间 = 改了实验）
SEED_BANDS = {
    "data": 10000,
    "split": 20000,
    "bootstrap": 30000,
    "train": 40000,
    "grpo": 50000,
    "eval": 60000,
}


def rng_for(stage: str, seed: int) -> np.random.Generator:
    if stage not in SEED_BANDS:
        raise KeyError(f"未知阶段 {stage}；可选 {sorted(SEED_BANDS)}")
    return np.random.default_rng(SEED_BANDS[stage] + int(seed))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
