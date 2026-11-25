"""Acuity 阶梯：同一道题，只改字号。

关键事实（务必不要在报告里说错）：
- 模型输入恒为 384×384，所以**渲染分辨率本身不改变 acuity**；
- acuity 由"逻辑字号 × 内容密度"决定，公式 `glyph_px = 8 × font_scale`；
- 让模型"看清"的两条路：调大 font_scale（训练侧），或者裁剪放大（推理侧，S6）。

"难度"三档在本项目里的定义：
    easy = 12 px，mid = 8 px，hard = 5 px
8 px 恰好是 font_scale=1.0 的默认值，也是 384 输入下 9pt 字体的真实像素高度。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from .conflict import fs_for_px, px_for_fs
from .make import Item

LEVELS: dict[str, float] = {"easy": 12.0, "mid": 8.0, "hard": 5.0}
LADDER: tuple[float, ...] = (4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0)


def at_px(item: Item, px: float) -> Item:
    """复制一条样本，把字号调到目标像素高度（其余不变）。"""
    from .render import glyph_px_modelled

    probe = replace(item.spec, font_scale=1.0)
    factor = px / glyph_px_modelled(probe)
    return Item(replace(item.spec, font_scale=round(factor, 4)), item.values,
                item.question, item.prior_type, item.prior_answer, item.pair_id,
                {**item.meta, "acuity_target": px})


def ladder(item_factory: Callable[[], Item], px_list: tuple[float, ...] = LADDER
           ) -> list[Item]:
    """对同一个生成器产出整条阶梯。item_factory 必须返回同一张表。"""
    base = item_factory()
    return [at_px(base, px) for px in px_list]


def level(item_factory: Callable[[], Item], name: str) -> Item:
    if name not in LEVELS:
        raise KeyError(f"未知难度 {name!r}；可选 {sorted(LEVELS)}")
    return at_px(item_factory(), LEVELS[name])


def describe(item: Item) -> dict:
    return {
        "font_scale": item.spec.font_scale,
        "acuity_px": px_for_fs(item.spec.font_scale),
        "size": item.spec.size,
        "n_categories": item.spec.n_categories,
        "n_series": item.spec.n_series,
    }


__all__ = ["LEVELS", "LADDER", "at_px", "ladder", "level", "describe", "fs_for_px"]
