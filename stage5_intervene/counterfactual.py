"""S5 · 反事实数据：同一张图，两种改动。

1. **不变性变体**（wording）：只改写问句措辞，金标不变。模型如果对措辞敏感，
   说明它在"背题面"而不是"看图"。
2. **响应变体**（numeric）：改表里的数值，让金标真的变。模型如果仍然答旧答案，
   说明它没重新读图。

两者都保留 `family_id`（同一族），这样划分时整族进同一个 split（2.0 §4.4）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from picovlm.config import DATA, ROOT
from picovlm.schema import Sample

WORDING = {
    "Which category has the highest value?":
        "Which category is the largest?",
    "Which category has the lowest value?":
        "Which category is the smallest?",
    "How many categories are above 50?": "Count the categories above 50.",
    "Is category A greater than category B?": "Does A exceed B?",
    "Which category is mentioned first in the question?":
        "Which category comes first in the question?",
}

ANSWER_KINDS = {"maxcat": "label", "mincat": "label", "compare": "bool",
                "count": "number", "order": "label", "value": "number"}


def wording_variant(s: Sample) -> Sample | None:
    """改写问句、金标与图都不变的版本。找不到映射时返回 None。"""
    new_q = WORDING.get(s.question)
    if not new_q:
        return None
    return Sample(**{**s.__dict__, "question": new_q,
                     "meta": {**s.meta, "variant": "wording",
                              "base_id": s.id}})


def numeric_variant(s: Sample, delta: float = 25.0) -> Sample | None:
    """改表里"当前最大值"那一列，让 argmax 换人（只支持 maxcat）。

    其他题型也能做，但需要按题型重算金标；为了不引入"我重算错了"的风险，
    这里只做我能证明正确的那一种，其余返回 None。
    """
    if s.qtype not in {"maxcat"}:
        return None
    from stage1_data.make import fmt_num
    from stage1_data.render import ChartSpec, render

    table = [row[:] for row in s.meta["table"]]
    spec = ChartSpec(**s.meta["spec"])
    n = len(table[0])
    col_max = [max(t[i] for t in table) for i in range(n)]
    cur = max(range(n), key=lambda i: col_max[i])
    nxt = max((i for i in range(n) if i != cur), key=lambda i: col_max[i])
    for t in table:
        t[nxt] = t[cur] + delta
    # 重新渲染（新图）并写盘
    res = render(spec, values=table)
    img_path = Path(s.image_path)
    root = ROOT
    digest = hashlib.sha1(res.image.tobytes()).hexdigest()[:16]
    out = root / "data" / "synth" / "images" / f"{digest}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        res.image.save(out)
    new_answer = chr(ord("A") + nxt)
    return Sample(**{**s.__dict__,
                     "id": f"{s.id}-cf-num",
                     "image_path": str(out.relative_to(root)).replace("\\", "/"),
                     "answer": new_answer,
                     "answer_kind": "label",
                     "prior_answer": s.prior_answer,
                     "meta": {**s.meta, "variant": "numeric",
                              "base_id": s.id, "table": table,
                              "argmax_col": nxt,
                              "delta": delta,
                              "old_answer": s.answer}})


def numeric_answer_changed(base: Sample, cf: Sample) -> bool:
    return base.answer != cf.answer
