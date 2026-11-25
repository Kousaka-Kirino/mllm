"""路径与默认超参。所有脚本都从这里取路径，不写死盘符。

想换数据目录：设环境变量 `PICOVLM_DATA`，不用改代码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("PICOVLM_DATA", ROOT / "data"))
IMAGES = DATA / "images"
OUTPUTS = Path(os.environ.get("PICOVLM_OUT", ROOT / "outputs"))
REPORTS = ROOT / "reports"


@dataclass(frozen=True)
class ModelConfig:
    """S2 组装的默认规格。改这里不会破坏已训好的 checkpoint（训练脚本会把它存进 config.json）。"""

    vision: str = "google/siglip-base-patch16-384"
    llm: str = "Qwen/Qwen2.5-0.5B-Instruct"
    input_size: int = 384
    patch: int = 16
    connector: str = "mlp"          # mlp | resampler | pixelshuffle
    n_queries: int = 64
    shuffle_factor: int = 2
    freeze_vision: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_text_tokens: int = 128

    @property
    def n_patches(self) -> int:
        return (self.input_size // self.patch) ** 2

    def visual_tokens(self) -> int:
        from .budget import count_tokens
        return count_tokens(self.connector, self.n_patches, self.n_queries,
                            self.shuffle_factor)


@dataclass(frozen=True)
class DataConfig:
    """合成数据的旋钮。**这些数字直接决定实验能不能成立**，不要随手改。"""

    seed: int = 20261125
    render_size: int = 1024        # 渲染边长；模型输入恒为 384，靠 font_scale 控 acuity
    n_train: int = 20000
    n_dev: int = 1000
    n_test: int = 2000
    prior_strength: float = 0.7    # 训练集里"先验答案正确"的比例（0.5 = 无偏）
    font_scales: tuple[float, ...] = (0.55, 0.7, 0.85, 1.0, 1.2, 1.45)
    n_categories: tuple[int, ...] = (4, 6, 8, 10)
    n_series: tuple[int, ...] = (1, 2, 3)


@dataclass(frozen=True)
class TrainConfig:
    lr_align: float = 1e-3
    lr_sft: float = 2e-4
    batch_size: int = 2
    grad_accum: int = 8
    epochs_align: int = 1
    epochs_sft: int = 2
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    bf16: bool = True
    grad_checkpoint: bool = True
    log_every: int = 20
    save_every: int = 500


@dataclass(frozen=True)
class EvalConfig:
    tol: float = 0.05
    n_boot: int = 2000
    seed: int = 7
    suites: tuple[str, ...] = field(default_factory=lambda: (
        "easy_consistent", "easy_conflict", "hard_consistent", "hard_conflict",
        "chartqa", "textvqa", "ood_style",
    ))
