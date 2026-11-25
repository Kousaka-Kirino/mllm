"""题目生成器：**先构造数值表，再问问题**。

这是整个项目"答案可信"的根基：
- 答案永远由表算出（`values[i, j]`），不从图上猜、不由模型生成；
- 先验答案也是显式构造的（"如果模型不看图、只按训练集偏置猜，会说什么"）；
- 因此 `answer` 与 `prior_answer` 的关系是**可审计**的，S1 的 verify 会逐条重算。

四种先验的构造方式（详见规划文档 §7.3）：
| prior_type | 怎么制造 | 先验答案 |
|---|---|---|
| position | 把最大值挪到第 0 列 | 最左边的类别 "A" |
| marginal | 是/否题的训练集 70% 为 yes | "yes" |
| numeric  | 计数题的训练集众数是 2 | "2" |
| order    | 同一张图，问法里先提谁 | 问题里第一个选项 |
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .render import ChartSpec, RenderResult, render

QTYPES = ("value", "compare", "count", "maxcat", "order", "maxseries")
NUMERIC_PRIOR = "2"


def fmt_num(v: float) -> str:
    """数字答案的规范写法：42.0 -> '42'，42.5 -> '42.5'。"""
    s = f"{float(v):.1f}".rstrip("0").rstrip(".")
    return s if s else "0"


@dataclass
class Question:
    text: str
    answer: str
    kind: str                 # number | bool | label
    qtype: str
    options: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def candidates(self) -> list[str]:
        return list(self.options) or [self.answer]


@dataclass
class Item:
    """一张图 + 一道题 + 先验标注。尚未分配 split / id。"""

    spec: ChartSpec
    values: np.ndarray
    question: Question
    prior_type: str = "none"
    prior_answer: str | None = None
    pair_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def render(self) -> RenderResult:
        return render(self.spec, self.values)

    def to_payload(self) -> dict[str, Any]:
        q = self.question
        return {
            "question": q.text,
            "answer": q.answer,
            "answer_kind": q.kind,
            "qtype": q.qtype,
            "prior_type": self.prior_type,
            "prior_answer": self.prior_answer,
            "pair_id": self.pair_id,
            "meta": {**self.meta, "options": q.candidates,
                     "table": self.values.round(4).tolist(),
                     "spec": _spec_dict(self.spec)},
        }


def _spec_dict(spec: ChartSpec) -> dict[str, Any]:
    return {k: getattr(spec, k) for k in spec.__dataclass_fields__}


def draw_values(seed: int, n_series: int, n_categories: int,
                noise: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(seed * 7919 + 13)
    base = rng.uniform(5, 95, size=(n_series, n_categories))
    if noise:
        base = base + rng.normal(0, noise * 100, size=base.shape)
    return np.clip(base, 1, 99).round(1)


def _base_spec(seed: int, n_categories: int, n_series: int, font_scale: float,
               size: int, distractors: int = 0) -> ChartSpec:
    kind = "bar" if n_series == 1 else "grouped_bar"
    return ChartSpec(seed=seed, kind=kind, n_categories=n_categories,
                     n_series=n_series, font_scale=font_scale, size=size,
                     distractors=distractors,
                     annotate=bool(seed % 3 == 0), noise=0.004)


# --------------------------------------------------------------------- 生成器
def gen_value(seed: int, n_categories: int = 6, font_scale: float = 1.0,
              size: int = 1152, distractors: int = 0) -> Item:
    """直接读一个柱子的数值。无先验，是最"干净"的能力题。"""
    values = draw_values(seed, 1, n_categories)
    spec = _base_spec(seed, n_categories, 1, font_scale, size, distractors)
    j = int(np.random.default_rng(seed + 1).integers(0, n_categories))
    label = chr(ord("A") + j)
    q = Question(f"What is the value for category {label}?", fmt_num(values[0, j]),
                 "number", "value",
                 options=tuple(fmt_num(v) for v in values[0]))
    return Item(spec, values, q, meta={"col": j})


def gen_compare(seed: int, prior_hit: bool, n_categories: int = 6,
                font_scale: float = 1.0, size: int = 1152) -> Item:
    """是/否题。prior_hit=True 时让答案为 yes（训练集 marginal 偏置）。"""
    rng = np.random.default_rng(seed + 2)
    values = draw_values(seed, 1, n_categories)
    a, b = rng.choice(n_categories, size=2, replace=False)
    if prior_hit and values[0, a] < values[0, b]:
        values[[0], [a, b]] = values[[0], [b, a]]
    if not prior_hit and values[0, a] > values[0, b]:
        values[[0], [a, b]] = values[[0], [b, a]]
    la, lb = chr(ord("A") + int(a)), chr(ord("A") + int(b))
    yes = bool(values[0, a] > values[0, b])
    q = Question(f"Is the value for category {la} greater than the value for category {lb}?",
                 "yes" if yes else "no", "bool", "compare", options=("yes", "no"))
    spec = _base_spec(seed, n_categories, 1, font_scale, size)
    return Item(spec, values, q, prior_type="marginal", prior_answer="yes",
                meta={"a": int(a), "b": int(b)})


def gen_count(seed: int, prior_hit: bool, n_categories: int = 8,
              threshold: float = 50.0, font_scale: float = 1.0,
              size: int = 1152) -> Item:
    """计数题。prior_hit=True 时让真实计数等于 2（训练集众数）。"""
    rng = np.random.default_rng(seed + 3)
    k = 2 if prior_hit else int(rng.choice([0, 1, 3, 4, 5]))
    k = min(k, n_categories)
    values = np.empty((1, n_categories))
    idx = rng.permutation(n_categories)
    for pos, col in enumerate(idx):
        if pos < k:
            values[0, col] = rng.uniform(threshold + 5, 95)
        else:
            values[0, col] = rng.uniform(5, threshold - 5)
    values = values.round(1)
    q = Question(f"How many categories have a value above {fmt_num(threshold)}?",
                 fmt_num(k), "number", "count", options=tuple(str(i) for i in range(n_categories + 1)))
    spec = _base_spec(seed, n_categories, 1, font_scale, size)
    return Item(spec, values, q, prior_type="numeric", prior_answer=NUMERIC_PRIOR,
                meta={"threshold": threshold, "count": k})


def gen_maxcat(seed: int, prior_hit: bool, n_categories: int = 6,
               font_scale: float = 1.0, size: int = 1152) -> Item:
    """位置题。prior_hit=True 时最大值在第 0 列（最左），先验答案 = 'A'。"""
    values = draw_values(seed, 1, n_categories)
    target = 0 if prior_hit else int(np.random.default_rng(seed + 4).integers(1, n_categories))
    arg = int(np.argmax(values[0]))
    if arg != target:
        values[0, [arg, target]] = values[0, [target, arg]]
    ans = chr(ord("A") + int(np.argmax(values[0])))
    q = Question("Which category has the highest value?", ans, "label", "maxcat",
                 options=tuple(chr(ord("A") + i) for i in range(n_categories)))
    spec = _base_spec(seed, n_categories, 1, font_scale, size)
    return Item(spec, values, q, prior_type="position", prior_answer="A",
                meta={"argmax_col": int(np.argmax(values[0]))})


def gen_order_pair(seed: int, n_categories: int = 6, font_scale: float = 1.0,
                   size: int = 1152) -> list[Item]:
    """顺序题：同一张图、两种问法。先验答案 = 问法里第一个被提到的类别。

    两条样本共享图像（同一 spec/values），只有题干顺序不同。
    这样"看图"与"看题干顺序"的答案就被拆开了。
    """
    rng = np.random.default_rng(seed + 5)
    values = draw_values(seed, 1, n_categories)
    picks = rng.choice(n_categories, size=2, replace=False)
    a, b = int(picks[0]), int(picks[1])
    while abs(values[0, a] - values[0, b]) < 5:
        b = int(rng.integers(0, n_categories))
        if b == a:
            continue
    la, lb = chr(ord("A") + a), chr(ord("A") + b)
    true_label = la if values[0, a] > values[0, b] else lb
    spec = _base_spec(seed, n_categories, 1, font_scale, size)
    pair_id = f"order-{seed:06d}"
    items = []
    for first, second in ((la, lb), (lb, la)):
        q = Question(f"Which category has the higher value, {first} or {second}?",
                     true_label, "label", "order", options=(first, second))
        items.append(Item(spec, values.copy(), q, prior_type="order",
                          prior_answer=first, pair_id=pair_id,
                          meta={"true_label": true_label}))
    return items


def gen_maxseries(seed: int, n_categories: int = 6, n_series: int = 3,
                  font_scale: float = 1.0, size: int = 1152) -> Item:
    """多系列题：在固定类别上，哪个系列最高。考"跨系列比较"。"""
    values = draw_values(seed, n_series, n_categories)
    rng = np.random.default_rng(seed + 6)
    j = int(rng.integers(0, n_categories))
    i = int(np.argmax(values[:, j]))
    q = Question(f"Which series has the highest value for category {chr(ord('A') + j)}?",
                 f"S{i + 1}", "label", "maxseries",
                 options=tuple(f"S{k + 1}" for k in range(n_series)))
    spec = ChartSpec(seed=seed, kind="grouped_bar", n_categories=n_categories,
                     n_series=n_series, font_scale=font_scale, size=size,
                     annotate=False, noise=0.004)
    return Item(spec, values, q, meta={"col": j, "argmax_series": i})


def gen_acuity_ladder(seed: int, target_px: tuple[float, ...],
                      n_categories: int = 8, size: int = 1152) -> list[tuple[float, Item]]:
    """同一张表 + 同一道题，只改字号 → acuity 阶梯。"""
    from .render import glyph_px_modelled

    base_values = draw_values(seed, 1, n_categories)
    rng = np.random.default_rng(seed + 7)
    j = int(rng.integers(0, n_categories))
    label = chr(ord("A") + j)
    out = []
    for px in target_px:
        probe = _base_spec(seed, n_categories, 1, 1.0, size)
        factor = px / glyph_px_modelled(probe)
        spec = _base_spec(seed, n_categories, 1, max(0.15, factor), size)
        q = Question(f"What is the value for category {label}?", fmt_num(base_values[0, j]),
                     "number", "value",
                     options=tuple(fmt_num(v) for v in base_values[0]))
        out.append((px, Item(spec, base_values.copy(), q, meta={"col": j})))
    return out
