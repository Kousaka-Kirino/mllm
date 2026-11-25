"""PicoVLM：小算力多模态模型的可复现实验骨架。

这个包只放**跨阶段共享**的东西：
- `schema`   冻结的数据契约（Sample / Prediction / EvalRecord）
- `answers`  判分（严格 / 宽松两套口径）
- `stats`    置信区间与配对比较
- `policies` 零显存基线（oracle / prior / constant / random）
- `config`   路径与默认超参

约定：包顶层**不 import torch**。这样 `python -m stage1_data.build`
在没装 GPU 依赖的机器上也能跑。
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
