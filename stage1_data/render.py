"""合成图表渲染器（S1 的地基）。

设计要点（改之前先读规划文档 §7）：

1. **逻辑尺寸固定**：图永远是 6×6 英寸，dpi = size/6。
   于是"字号 pt → 模型侧像素"的换算是解析的：
       glyph_px = pt × font_scale × (dpi/72) × (384/size)
                = pt × font_scale × 384/432        （与 size 无关）
   所以**渲染 DPI 本身不改变模型侧 acuity**——只有 font_scale、
   内容密度、以及"裁剪放大"才改变。这是本项目最容易搞错的一点。

2. 一切随机都来自 `ChartSpec.seed`，同一个 spec 渲染两次逐像素相同。

3. 只画英文标签（A/B/C、S1/S2），避免缺 CJK 字体导致豆腐块污染实验。
   中文题干作为 OOD 实验在 S4 单独加，不进主表。

4. 图像不落盘到这里；`build.py` 负责存。渲染器只返回 PIL + 数值表。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

FIG_IN = 6.0            # 逻辑边长（英寸）
BASE_PT = 9.0           # 刻度字号（acuity 以它为准）
TITLE_PT = 12.0
ANNOT_PT = 7.0
PALETTE = ("#4C72B0", "#DD8452", "#55A868", "#C44E52")
# OOD 风格用一套**视觉上明显不同**的配色与字体，但信息量不变（E5 迁移实验用）
PALETTE_OOD = ("#1A1A1A", "#7F7F7F", "#C00000", "#2E75B6")
STYLES = ("in", "ood")
KINDS = ("bar", "grouped_bar", "line")
DISTRACTOR_TEXTS = (
    "Source: FY2024", "n = 1,204", "*p < 0.05", "Note: rounded",
    "Data: internal", "Q3 only", "preliminary", "see appendix",
)


@dataclass(frozen=True)
class ChartSpec:
    seed: int
    kind: str = "bar"
    n_categories: int = 6
    n_series: int = 1
    font_scale: float = 1.0
    size: int = 1152              # 必须是 6 的整数倍
    legend_gap: float = 0.02
    distractors: int = 0
    noise: float = 0.0
    annotate: bool = False
    style: str = "in"             # "in" 训练分布 / "ood" 只用于 E5 迁移评测

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind 必须是 {KINDS}")
        if self.style not in STYLES:
            raise ValueError(f"style 必须是 {STYLES}")
        if self.size % 6:
            raise ValueError("size 必须是 6 的整数倍（保证 dpi 整除）")
        if self.n_series > len(PALETTE):
            raise ValueError(f"最多 {len(PALETTE)} 个系列")
        if self.kind == "bar" and self.n_series != 1:
            raise ValueError("bar 只能有 1 个系列；多系列请用 grouped_bar")


@dataclass
class RenderResult:
    image: Image.Image
    values: np.ndarray            # (n_series, n_categories)
    categories: list[str]
    series: list[str]
    spec: ChartSpec
    glyph_px: float               # 解析式模型侧字形高度（input=384，未裁剪）
    measure: dict = None          # 实测统计（render(..., measure=True) 时才有）

    @property
    def table(self) -> dict[str, Any]:
        return {
            "categories": self.categories,
            "series": self.series,
            "values": self.values.round(4).tolist(),
        }


def glyph_px_modelled(spec: ChartSpec, input_size: int = 384,
                      crop: float = 1.0) -> float:
    """把"字号"换算成"模型实际看到的字形高度（px）"。

    crop: 输入图占原图的比例。crop=0.5 表示裁一半再缩到 input_size，
    等效放大 2 倍，acuity 也翻倍。
    """
    if not 0 < crop <= 1:
        raise ValueError("crop 必须在 (0, 1]")
    dpi = spec.size / FIG_IN
    render_px = BASE_PT * spec.font_scale * dpi / 72.0
    return round(render_px * (input_size / spec.size) / crop, 3)


def make_spec(seed: int, **overrides: Any) -> ChartSpec:
    """由 seed 决定结构性随机量（类别数/系列数/噪声），其余走 overrides。"""
    rng = np.random.default_rng(seed)
    defaults: dict[str, Any] = {
        "kind": "bar" if rng.random() < 0.6 else "grouped_bar",
        "n_categories": int(rng.choice([4, 6, 8, 10])),
        "n_series": 1 if rng.random() < 0.6 else int(rng.choice([2, 3])),
        "font_scale": 1.0,
        "distractors": int(rng.integers(0, 2)),
        "noise": float(rng.uniform(0, 0.02)),
        "annotate": bool(rng.random() < 0.3),
        "legend_gap": float(rng.uniform(0.0, 0.06)),
    }
    if defaults["kind"] == "bar":
        defaults["n_series"] = 1
    defaults.update(overrides)
    return ChartSpec(seed=seed, **defaults)


def _values(spec: ChartSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed * 7919 + 13)
    base = rng.uniform(5, 95, size=(spec.n_series, spec.n_categories))
    noise = rng.normal(0, spec.noise * 100, size=base.shape)
    vals = np.clip(base + noise, 1, 99).round(1)
    return vals


def render(spec: ChartSpec, values: np.ndarray | None = None,
           measure: bool = False) -> RenderResult:
    """渲染一张图。返回 PIL 图 + 底层数值表（答案从表里算，不从图上猜）。

    values: 显式指定数值表（形状 n_series × n_categories）。
    冲突对/位置对需要"控制谁是最大值"，所以必须能传表进来。
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if values is None:
        values = _values(spec)
    values = np.asarray(values, dtype=float)
    if values.shape != (spec.n_series, spec.n_categories):
        raise ValueError(f"values 形状应为 {(spec.n_series, spec.n_categories)}，"
                         f"收到 {values.shape}")
    categories = [chr(ord("A") + i) for i in range(spec.n_categories)]
    series = [f"S{i + 1}" for i in range(spec.n_series)]
    dpi = spec.size / FIG_IN
    fs = spec.font_scale

    fig = Figure(figsize=(FIG_IN, FIG_IN), dpi=dpi)
    FigureCanvasAgg(fig)
    ood = spec.style == "ood"
    palette = PALETTE_OOD if ood else PALETTE
    ax = fig.add_axes((0.16, 0.20, 0.78, 0.66) if ood else (0.13, 0.12, 0.75, 0.72))
    if ood:
        # 衬线字体 + 全框 + 双轴网格 + 斜的刻度标签：模型没见过的渲染风格
        fig.set_facecolor("white")
        ax.set_facecolor("#FAFAFA")

    x = np.arange(spec.n_categories)
    if spec.kind == "line":
        for i in range(spec.n_series):
            ax.plot(x, values[i], marker="s" if ood else "o", ms=3 * fs,
                    lw=(2.4 if ood else 1.6) * fs,
                    color=palette[i], label=series[i])
    elif spec.kind == "grouped_bar" and spec.n_series > 1:
        width = 0.8 / spec.n_series
        for i in range(spec.n_series):
            offset = (i - (spec.n_series - 1) / 2) * width
            ax.bar(x + offset, values[i], width * 0.92, color=palette[i],
                   label=series[i])
    else:
        ax.bar(x, values[0], 0.7, color=palette[0],
               edgecolor="#222222" if ood else "none", linewidth=0.8 if ood else 0)
        if spec.annotate:
            for xi, v in zip(x, values[0]):
                ax.text(xi, v + 1.5, f"{v:g}", ha="center", va="bottom",
                        fontsize=ANNOT_PT * fs, color="#333333")

    ax.set_xticks(x)
    if ood:
        ax.set_xticklabels(categories, fontsize=BASE_PT * fs, rotation=45,
                           fontfamily="serif")
    else:
        ax.set_xticklabels(categories, fontsize=BASE_PT * fs)
    ax.tick_params(axis="y", labelsize=BASE_PT * fs)
    ax.set_ylim(0, max(110, float(values.max()) * 1.18))
    title = "Category breakdown (2024)" if ood else "Quarterly metric by category"
    ax.set_title(title, fontsize=TITLE_PT * fs, pad=8 * fs,
                 fontfamily="serif" if ood else None)
    ax.set_xlabel("category" if ood else "Category", fontsize=BASE_PT * fs,
                  fontfamily="serif" if ood else None)
    ax.set_ylabel("value" if ood else "Value", fontsize=BASE_PT * fs,
                  fontfamily="serif" if ood else None)
    if ood:
        ax.grid(True, alpha=0.5, lw=0.7, ls=":")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.4)
    else:
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    if spec.n_series > 1:
        ax.legend(fontsize=BASE_PT * fs, frameon=False, ncol=min(spec.n_series, 4),
                  loc="lower center", bbox_to_anchor=(0.5, 1.02 + spec.legend_gap))

    if spec.distractors:
        drng = np.random.default_rng(spec.seed * 31 + 7)
        for i in range(spec.distractors):
            text = DISTRACTOR_TEXTS[int(drng.integers(0, len(DISTRACTOR_TEXTS)))]
            fig.text(float(drng.uniform(0.02, 0.9)), float(drng.uniform(0.02, 0.9)),
                     text, fontsize=6 * fs, alpha=0.55, color="#555555",
                     rotation=float(drng.choice([0, 0, 0, 90])))

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    img = Image.fromarray(buf.astype(np.uint8), mode="RGB")
    if (w, h) != (spec.size, spec.size):
        img = img.resize((spec.size, spec.size), Image.LANCZOS)

    meas = None
    if measure:
        from .measure import measure_figure
        meas = measure_figure(fig, spec.size, input_size=384)

    return RenderResult(image=img, values=values, categories=categories,
                        series=series, spec=spec,
                        glyph_px=glyph_px_modelled(spec), measure=meas)


def acuity_ladder(base: ChartSpec, target_px: tuple[float, ...] = (4, 5, 6, 7, 8, 10, 12)
                  ) -> list[tuple[float, ChartSpec]]:
    """给定目标字号高度，反解 font_scale。用于 S1 的 acuity 阶梯。"""
    out = []
    for px in target_px:
        factor = px / glyph_px_modelled(base)
        spec = ChartSpec(**{**base.__dict__, "font_scale": base.font_scale * factor})
        out.append((px, spec))
    return out


def resize_for_model(img: Image.Image, input_size: int = 384) -> Image.Image:
    """模拟真实推理：任意渲染分辨率都会被 resize 到模型输入。"""
    return img.convert("RGB").resize((input_size, input_size), Image.BICUBIC)


def crop_zoom(img: Image.Image, crop: float, anchor: tuple[float, float] = (0.5, 0.5),
              input_size: int = 384) -> Image.Image:
    """裁剪 + 放大：S6 里"看细节"的最朴素实现。anchor 是裁剪窗中心（相对坐标）。"""
    w, h = img.size
    side = int(round(min(w, h) * crop))
    cx, cy = anchor
    left = int(round(cx * (w - side)))
    top = int(round(cy * (h - side)))
    box = (max(0, left), max(0, top), min(w, left + side), min(h, top + side))
    return img.crop(box).resize((input_size, input_size), Image.BICUBIC)


def estimate_text_px(n_categories: int, fig_in: float = FIG_IN) -> float:
    """内容密度的粗估：类别越多，每个标签分到的水平空间越小（英寸）。"""
    return round(fig_in * 0.75 / max(n_categories, 1) * 72, 2)


__all__ = ["ChartSpec", "RenderResult", "make_spec", "render", "glyph_px_modelled",
           "acuity_ladder", "resize_for_model", "crop_zoom", "estimate_text_px"]
