"""冻结的数据契约：Sample / Prediction / EvalRecord。

规则（v1.0 起冻结，改字段要写进规划文档 §17 变更记录）：

1. 字段名不做同义替换；新增字段必须带默认值，旧 jsonl 仍可读。
2. `Sample.id` 全局唯一、可由参数重建，**不含随机数**（便于按 id 复查）。
3. `Prediction` 只记录"模型说了什么"；判分在 `picovlm.answers` 里做。
   两者分离，才能对同一批 prediction 换判分口径重算。
4. 所有 *_path 都是相对项目根的 POSIX 风格路径（`data/images/...`）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SPLITS = ("train", "dev", "test")
ANSWER_KINDS = ("number", "text", "bool", "label")
PRIOR_TYPES = ("none", "position", "marginal", "numeric", "order")


@dataclass(frozen=True)
class Sample:
    """一道题。图像不进 jsonl，只存路径。"""

    id: str
    source: str                 # synth | flickr30k | chartqa | textvqa | plotqa
    split: str                  # train | dev | test
    image_path: str
    image_size: int             # 渲染边长 S（px）；真实数据集填原始短边
    question: str
    answer: str                 # 规范答案（判分前会再归一化一次）
    answer_kind: str = "text"
    qtype: str = "unknown"      # max | min | value | count | compare | sum | order
    prior_type: str = "none"
    prior_answer: str | None = None   # 只靠先验会给出的答案（冲突对里 != answer）
    acuity: float | None = None       # 模型侧字形高度 px（合成数据可解析计算）
    pair_id: str | None = None        # 冲突对 / 顺序对的组 id
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Sample.id 不能为空")
        if self.split not in SPLITS:
            raise ValueError(f"split 必须是 {SPLITS}，收到 {self.split!r}")
        if self.answer_kind not in ANSWER_KINDS:
            raise ValueError(f"answer_kind 必须是 {ANSWER_KINDS}")
        if self.prior_type not in PRIOR_TYPES:
            raise ValueError(f"prior_type 必须是 {PRIOR_TYPES}")
        if self.image_size <= 0:
            raise ValueError("image_size 必须为正")
        if not self.question.strip():
            raise ValueError("question 不能为空")
        if not str(self.answer).strip():
            raise ValueError("answer 不能为空")
        if self.prior_type != "none" and not self.prior_answer:
            raise ValueError("带先验的样本必须有 prior_answer")

    # ---------------------------------------------------------------- 序列化
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Sample":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Prediction:
    """一次推理的结果。不做判分，只留原文与成本。"""

    sample_id: str
    model_id: str
    run_id: str
    raw: str
    norm: str = ""
    correct_strict: bool | None = None
    correct_relaxed: bool | None = None
    latency_ms: float | None = None
    visual_tokens: int | None = None
    budget: str = "fixed"       # fixed | oracle | learned | random | heuristic
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prediction":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class EvalRecord:
    """一个 suite 在一个模型上的汇总。所有区间都基于二项分布。"""

    run_id: str
    suite: str
    model_id: str
    n: int
    accuracy: float
    ci_low: float
    ci_high: float
    tokens_mean: float | None = None
    latency_ms_mean: float | None = None
    by_qtype: dict[str, Any] = field(default_factory=dict)
    by_prior: dict[str, Any] = field(default_factory=dict)
    by_acuity: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def line(self) -> str:
        return (f"{self.suite:<22} {self.model_id:<28} n={self.n:<5} "
                f"acc={self.accuracy:6.1%} [{self.ci_low:.1%},{self.ci_high:.1%}]"
                + (f" tokens={self.tokens_mean:.0f}" if self.tokens_mean else ""))


def group_key(sample: Sample) -> tuple[str, str, str]:
    """评测分层用的三元组：(qtype, prior_type, acuity 桶)。"""
    bucket = "unknown" if sample.acuity is None else f"{sample.acuity:.0f}px"
    return sample.qtype, sample.prior_type, bucket
