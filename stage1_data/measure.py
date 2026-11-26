"""实测字形像素：把"字号 9pt × font_scale"变成"图上真正有多少像素高的字"。

为什么不能只靠解析式（2.0 §4.2）：
    解析式假设"9pt 的字高 = 9pt × dpi/72"，但数字 `7` 和字母 `g` 的实际着墨高度不同，
    抗锯齿、matplotlib 的字体回退、坐标轴预留的边距都会让它偏几像素。
    而 acuity 实验的自变量就是这个高度——**自变量测不准，整个 2×2 就没意义**。

做法（可复核）：
    1. 渲染完成后，从 figure 上取每个文字对象的 bbox（display 坐标）；
    2. 把 bbox 映射回图像像素坐标，裁出这一小块；
    3. 数"含有深色像素的行数" = 这一串字的着墨高度；
    4. 只保留刻度数字与数值标注（题目真正要读的字），取中位数作为该样本的 acuity；
    5. 按 `model_px = render_px × input_size / render_size` 换算到模型输入尺度。

产物：
    `data/acuity_thresholds.json` 里的分档阈值由 `build` 依据全体实测分布生成，
    一经写入即冻结；S4 只按实测值分箱，不再看 font_scale。
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

import numpy as np

INK_THRESHOLD = 200          # 灰度低于它算着墨
MIN_INK_ROWS = 2             # 少于 2 行视为噪点/标点
ACUITY_KINDS = ("tick", "annot")


def _gray(fig) -> np.ndarray:
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    # 亮度加权，比简单均值更接近肉眼
    return (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])


def _iter_texts(fig) -> list[tuple[Any, str]]:
    """列出 (Text 对象, 类别)。类别决定它算不算 acuity 的证据。"""
    from matplotlib.text import Text

    ax = fig.axes[0] if fig.axes else None
    tagged: dict[int, str] = {}
    if ax is not None:
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            tagged[id(t)] = "tick"
        for t in ax.texts:
            tagged[id(t)] = "annot"
        legend = ax.get_legend()
        if legend is not None:
            for t in legend.get_texts():
                tagged[id(t)] = "legend"
        for name in ("title", "xlabel", "ylabel"):
            got = getattr(ax, f"get_{name}", None)
            if got is not None:
                tagged[id(got())] = "axis"

    out = []
    for t in fig.findobj(Text):
        text = (t.get_text() or "").strip()
        if not text:
            continue
        out.append((t, tagged.get(id(t), "other")))
    return out


def measure_figure(fig, render_size: int, input_size: int = 384,
                   ax_index: int = 0) -> dict:
    """测量一张已 draw 过的 figure，返回统计字典（可 JSON 序列化）。"""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gray = _gray(fig)
    h_px = gray.shape[0]
    scale = input_size / float(render_size)

    regions: list[dict] = []
    for text_artist, kind in _iter_texts(fig):
        try:
            ext = text_artist.get_window_extent(renderer)
        except Exception:
            continue
        if ext.width < 2 or ext.height < 2:
            continue
        x0 = max(0, int(np.floor(ext.x0)))
        x1 = min(gray.shape[1], int(np.ceil(ext.x1)))
        y0 = max(0, int(np.floor(h_px - ext.y1)))
        y1 = min(h_px, int(np.ceil(h_px - ext.y0)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        crop = gray[y0:y1, x0:x1]
        ink = crop < INK_THRESHOLD
        rows = int(ink.any(axis=1).sum())
        if rows < MIN_INK_ROWS:
            continue
        regions.append({
            "text": (text_artist.get_text() or "")[:24],
            "kind": kind,
            "ink_px_render": rows,
            "ink_px_model": round(rows * scale, 3),
            "box": [x0, y0, x1 - x0, y1 - y0],
        })

    core = [r["ink_px_render"] for r in regions if r["kind"] in ACUITY_KINDS]
    if not core:                       # 退化情况：图里没有刻度/标注
        core = [r["ink_px_render"] for r in regions]
    if not core:
        return {"n_regions": 0, "glyph_px_render": 0.0, "glyph_px_model": 0.0,
                "n_acuity_regions": 0, "regions": []}

    core_sorted = sorted(core)
    q1, q3 = np.percentile(core_sorted, [25, 75])
    summary = {
        "n_regions": len(regions),
        "n_acuity_regions": len(core),
        "glyph_px_render": round(st.median(core_sorted), 3),
        "glyph_px_model": round(st.median(core_sorted) * scale, 3),
        "glyph_px_min_model": round(min(core_sorted) * scale, 3),
        "glyph_px_max_model": round(max(core_sorted) * scale, 3),
        "iqr_model": round(float(q3 - q1) * scale, 3),
        "scale": round(scale, 4),
    }
    # 只留少量样例，避免 jsonl 膨胀；完整测量可用 --spotcheck 复核
    summary["regions"] = regions[:8]
    return summary


def spotcheck(specs, out_dir: Path, input_size: int = 384, crops_per_spec: int = 2) -> Path:
    """把测量到的文字块裁出来存成图片 + 一张对照表，供人工核验。"""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, spec in enumerate(specs):
        from .render import render

        res = render(spec, measure=True)
        meas = res.measure
        img_path = out_dir / f"{i:02d}_fs{spec.font_scale:.2f}.png"
        res.image.save(img_path)
        crops = meas.get("regions", [])[:crops_per_spec]
        for j, region in enumerate(crops):
            x, y, w, h = region["box"]
            crop = res.image.crop((x, y, x + w, y + h))
            crop = crop.resize((max(w * 3, 8), max(h * 3, 8)), Image.NEAREST)
            crop.save(out_dir / f"{i:02d}_crop{j}_{region['kind']}.png")
        rows.append((spec.font_scale, meas.get("glyph_px_render"),
                     meas.get("glyph_px_model"), meas.get("n_acuity_regions")))
    md = ["# acuity 人工核验表", "",
          "| # | font_scale | 实测 render px（中位） | 换算 model px | 参与统计的文字块 |",
          "|---|---:|---:|---:|---:|"]
    for i, (fs, rp, mp, n) in enumerate(rows):
        md.append(f"| {i} | {fs:.2f} | {rp} | {mp} | {n} |")
    md += ["", "看 `crop*.png`：字形高度是否与表中数字观感一致。", ""]
    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8", newline="\n")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stage1_data.measure")
    ap.add_argument("--demo", action="store_true", help="打印几个字号档的实测值")
    ap.add_argument("--spotcheck", type=int, default=0, help="导出 N 组人工核验图")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args(argv)

    from .render import ChartSpec, render

    if args.demo or not args.spotcheck:
        print("| font_scale | 解析式 model px | 实测 render px | 实测 model px | IQR | 块数 |")
        print("|---|---:|---:|---:|---:|---:|")
        for fs in (0.55, 0.7, 0.85, 1.0, 1.2, 1.45):
            spec = ChartSpec(seed=args.seed, font_scale=fs)
            res = render(spec, measure=True)
            m = res.measure
            print(f"| {fs:.2f} | {res.glyph_px:.2f} | {m.get('glyph_px_render')} | "
                  f"{m.get('glyph_px_model')} | {m.get('iqr_model')} | {m.get('n_acuity_regions')} |")

    if args.spotcheck:
        specs = [ChartSpec(seed=args.seed + i, font_scale=fs)
                 for i, fs in enumerate((0.55, 0.85, 1.0, 1.45))]
        out = spotcheck(specs, Path("reports/acuity_spotcheck"))
        print(f"[measure] 人工核验材料 → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
