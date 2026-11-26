"""冻结的数据契约：Sample / Prediction / EvalRecord。

规则（2.0 起冻结，改字段要写进规划文档的变更记录）：

1. 字段名不做同义替换；新增字段必须带默认值，旧 jsonl 仍可读。
2. `Sample.id` 全局唯一、可由参数重建，**不含随机数**（便于按 id 复查）。
3. `Sample.family_id` 是**划分单位**：同一个 family 的样本只能落在一个 split 里，
   否则"同一条底层表的题干变体"会横跨训练与测试，泄漏无法挽回（2.0 §4.4）。
4. `Prediction` 只记录"模型说了什么"；判分在 `picovlm.answers` 里做。
   两者分离，才能对同一批 prediction 换判分口径重算。
5. 所有 *_path 都是相对项目根的 POSIX 风格路径（`data/images/...`）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SPLITS = ("train", "val", "dev_test", "locked_test")
ANSWER_KINDS = ("number", "text", "bool", "label")
PRIOR_TYPES = ("none", "position", "marginal", "numeric", "order")
ACUITY_BINS = ("easy", "mid", "hard", "extreme")
CONDITIONS = ("prior_only", "image_only", "correct_image", "mismatched_image")


@dataclass(frozen=True)
class Sample:
    """一道题。图像不进 jsonl，只存路径。"""

    id: str
    source: str                 # synth | chartqa | textvqa | plotqa | ood_style
    split: str                  # train | val | dev_test | locked_test
    image_path: str
    image_size: int             # 渲染边长 S（px）；真实数据集填原始短边
    question: str
    answer: str                 # 规范答案（判分前会再归一化一次）
    answer_kind: str = "text"
    qtype: str = "unknown"      # max | min | value | count | compare | sum | order
    prior_type: str = "none"
    prior_answer: str | None = None   # 只靠先验会给出的答案（冲突对里 != answer）
    family_id: str = ""               # 划分单位：同族只能同 split
    glyph_px_render: float | None = None  # 实测：渲染尺度上数字的着墨高度（px）
    glyph_px_model: float | None = None   # 实测：换算到模型输入尺度后的高度（px）
    acuity_bin: str | None = None         # 由实测 px 按冻结阈值分箱
    pixel_stats: dict[str, Any] = field(default_factory=dict)  # 中位数/IQR/最小/最大
    pair_id: str | None = None            # 反事实对 / 顺序对的组 id
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
        if self.acuity_bin is not None and self.acuity_bin not in ACUITY_BINS:
            raise ValueError(f"acuity_bin 必须是 {ACUITY_BINS}")
        if self.image_size <= 0:
            raise ValueError("image_size 必须为正")
        if not self.question.strip():
            raise ValueError("question 不能为空")
        if not str(self.answer).strip():
            raise ValueError("answer 不能为空")
        if self.prior_type != "none" and not self.prior_answer:
            raise ValueError("带先验的样本必须有 prior_answer")
        if (self.glyph_px_model is None) != (self.glyph_px_render is None):
            raise ValueError("glyph_px_render 与 glyph_px_model 必须同时有或同时没有")

    # ---------------------------------------------------------------- 派生量
    @property
    def prior_holds(self) -> bool:
        """先验答案是否正好等于金标。冲突对里为 False。"""
        return self.prior_answer is not None and str(self.prior_answer) == str(self.answer)

    @property
    def is_conflict(self) -> bool:
        return self.prior_type != "none" and not self.prior_holds

    @property
    def acuity(self) -> float | None:
        """兼容 v1 字段名：等价于实测的模型侧字形高度。"""
        return self.glyph_px_model

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
    condition: str = "correct_image"   # E0 的四种呈现方式
    family_id: str = ""         # 从 Sample 复制，便于按族聚类
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Prediction":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class EvalRecord:
    """一个 suite 的汇总。由 `eval_utils.records()` 生成，可整段重算。"""

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
    condition: str = "correct_image"
    n_families: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def line(self) -> str:
        return (f"{self.suite:<18} n={self.n:<4} acc={self.accuracy:.3f} "
                f"[{self.ci_low:.3f}, {self.ci_high:.3f}] cond={self.condition}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def group_key(sample: Sample) -> str:
    """统计分组用：acuity 档 × 先验是否成立。四格分析就靠它。"""
    acu = sample.acuity_bin or "unknown"
    prior = "conflict" if sample.is_conflict else (
        "consistent" if sample.prior_type != "none" else "neutral")
    return f"{acu}/{prior}"
