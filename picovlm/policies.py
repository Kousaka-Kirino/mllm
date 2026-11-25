"""零显存基线：oracle / prior / constant / random。

它们的作用不是"凑数"，而是**定义下界和上界**：
- `oracle`  = 100%，任何模型低于它说明还有空间；也是"上限校准"。
- `prior`   = 只靠先验能拿多少分。**这是本项目最重要的基线**：
  如果 VLM 的分数 ≈ prior，说明它根本没在图上看东西。
- `constant`= 训练集里最常见的答案（最笨的基线）。
- `random`  = 在候选答案里均匀猜。

所有 policy 都是 `(Sample, rng) -> str`，和真模型同一个接口，
所以 `stage3_train/eval_suite.py` 不需要为它们写特殊分支。
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .schema import Sample

Policy = Callable[[Sample, random.Random], str]


@dataclass(frozen=True)
class PolicySpec:
    name: str
    fn: Policy
    needs: str = ""     # 需要在训练集上统计什么："" | "majority" | "candidates"


def oracle(sample: Sample, rng: random.Random) -> str:
    return str(sample.answer)


def prior(sample: Sample, rng: random.Random) -> str:
    """只看问题与训练分布，不看图。没有先验答案时退化为训练集多数答案。"""
    if sample.prior_answer:
        return str(sample.prior_answer)
    return str(sample.meta.get("majority_answer", sample.answer))


def constant(sample: Sample, rng: random.Random) -> str:
    return str(sample.meta.get("majority_answer", ""))


def uniform(sample: Sample, rng: random.Random) -> str:
    cands = sample.meta.get("candidates") or [sample.answer]
    return str(rng.choice(list(cands)))


def first_option(sample: Sample, rng: random.Random) -> str:
    """"永远答第一个提到的选项"——专测 order 先验。"""
    opts = sample.meta.get("options") or [sample.answer]
    return str(opts[0])


POLICIES: dict[str, PolicySpec] = {
    "oracle": PolicySpec("oracle", oracle),
    "prior": PolicySpec("prior", prior, needs="majority"),
    "constant": PolicySpec("constant", constant, needs="majority"),
    "random": PolicySpec("random", uniform, needs="candidates"),
    "first_option": PolicySpec("first_option", first_option, needs="candidates"),
}


def get_policy(name: str) -> Policy:
    if name not in POLICIES:
        raise KeyError(f"未知 policy {name!r}；可选 {sorted(POLICIES)}")
    return POLICIES[name].fn


def fit_train_stats(samples: Iterable[Sample]) -> dict[str, Counter]:
    """从训练集统计多数答案、各题型的候选答案集合。供 constant / random 基线使用。"""
    overall: Counter = Counter()
    by_qtype: dict[str, Counter] = {}
    for s in samples:
        overall[str(s.answer)] += 1
        by_qtype.setdefault(s.qtype, Counter())[str(s.answer)] += 1
    return {"overall": overall, "by_qtype": by_qtype}


def attach_stats(samples: Sequence[Sample], stats: dict[str, Counter]) -> list[Sample]:
    """把统计结果写进 meta，便于 jsonl 落盘后复盘。"""
    out = []
    for s in samples:
        meta = dict(s.meta)
        counter = stats["by_qtype"].get(s.qtype) or stats["overall"]
        if counter:
            meta["majority_answer"] = counter.most_common(1)[0][0]
            meta["candidates"] = [a for a, _ in counter.most_common(8)]
        out.append(Sample.from_dict({**s.to_dict(), "meta": meta}))
    return out
