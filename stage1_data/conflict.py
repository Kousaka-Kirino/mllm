"""冲突对：**同一道题，先验答案与视觉答案相反**。

四种先验各有一个 `*_pair(seed)`，返回 `(consistent, conflict)`：
- consistent：答案 == 先验答案（不看图也能答对）
- conflict  ：答案 != 先验答案（不看图必错）

一致性对用来算"先验能拿多少分"，冲突对用来算 VDR：
    VDR = 冲突对上答对的比例 = 1 − 跟着先验走
"""

from __future__ import annotations

from .make import (NUMERIC_PRIOR, Item, gen_compare, gen_count, gen_maxcat,
                   gen_order_pair)

# 字号 → 模型侧像素：glyph = 9pt × fs × 384/432 = 8×fs
# 所以 fs = px/8 是精确反解，不是拟合。
PX_PER_FS = 8.0


def fs_for_px(px: float) -> float:
    return round(px / PX_PER_FS, 4)


def px_for_fs(fs: float) -> float:
    return round(fs * PX_PER_FS, 3)


def _tag(items: list[Item], prefix: str) -> list[Item]:
    for k, it in enumerate(items):
        it.pair_id = f"{prefix}-{it.spec.seed:06d}"
        it.meta["pair_role"] = "consistent" if k == 0 else "conflict"
    return items


def position_pair(seed: int, font_scale: float = 1.0, **kw) -> list[Item]:
    """同 seed 同表，只把最大值挪到第 0 列 / 第 k 列。"""
    ok = gen_maxcat(seed, prior_hit=True, font_scale=font_scale, **kw)
    bad = gen_maxcat(seed, prior_hit=False, font_scale=font_scale, **kw)
    return _tag([ok, bad], "pos")


def marginal_pair(seed: int, font_scale: float = 1.0, **kw) -> list[Item]:
    """是/否题：真值为 yes / no，先验恒为 yes。"""
    ok = gen_compare(seed, prior_hit=True, font_scale=font_scale, **kw)
    bad = gen_compare(seed, prior_hit=False, font_scale=font_scale, **kw)
    return _tag([ok, bad], "mar")


def numeric_pair(seed: int, font_scale: float = 1.0, **kw) -> list[Item]:
    """计数题：真值 = 2 / != 2，先验恒为 2。"""
    ok = gen_count(seed, prior_hit=True, font_scale=font_scale, **kw)
    bad = gen_count(seed, prior_hit=False, font_scale=font_scale, **kw)
    return _tag([ok, bad], "num")


def order_pair(seed: int, font_scale: float = 1.0, **kw) -> list[Item]:
    """顺序题：同一张图两种问法，按"先验是否成立"排序。"""
    a, b = gen_order_pair(seed, font_scale=font_scale, **kw)
    ok = a if a.question.answer == a.prior_answer else b
    bad = b if ok is a else a
    return _tag([ok, bad], "ord")


PAIR_MAKERS = {
    "position": position_pair,
    "marginal": marginal_pair,
    "numeric": numeric_pair,
    "order": order_pair,
}


def make_pair(prior_type: str, seed: int, **kw) -> list[Item]:
    if prior_type not in PAIR_MAKERS:
        raise KeyError(f"未知 prior_type {prior_type!r}；可选 {sorted(PAIR_MAKERS)}")
    return PAIR_MAKERS[prior_type](seed, **kw)


def prior_holds(item: Item) -> bool:
    """这条样本上，先验答案是否就是正确答案。"""
    return str(item.prior_answer) == str(item.question.answer)


__all__ = ["PAIR_MAKERS", "make_pair", "prior_holds", "fs_for_px", "px_for_fs",
           "position_pair", "marginal_pair", "numeric_pair", "order_pair",
           "NUMERIC_PRIOR"]
