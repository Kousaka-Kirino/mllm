"""S1 · E5：OOD 风格测试集（同一批题，换一套渲染风格）。

E5 问的是：**换了没见过的画风，模型还答得对吗**。
所以这里不动题目、不动答案、不动 family——只把同一张表的图重画成
"训练分布之外的样子"（衬线字体、全框、双轴网格、斜刻度、另一套配色）。

这样 `sft_dev` 与 `sft_ood` 两轮评测就是**同实例配对**，
可以直接做 McNemar，不用造新题，也不会被"题目难度不同"污染。

用法：

    python -m stage1_data.ood                 # 构建 + 自检
    python -m stage1_data.ood --limit 40      # 只做 40 条（调试）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from picovlm.config import DATA
from picovlm.io_utils import ensure_dir, write_json, write_jsonl

from stage1_data.render import ChartSpec, render
from stage3_train.dataset import load_samples

OOD_SPLIT = "ood_style"
OOD_DIR = DATA / "synth" / "ood_images"


def build(limit: int | None = None, out_dir: Path | None = None) -> dict:
    """把 dev_test 的每条样本按 ood 风格重画，返回统计表。"""
    out_dir = ensure_dir(Path(out_dir or OOD_DIR))
    samples = load_samples("dev_test", limit=limit)
    rows, dx, bins = [], [], {}
    for s in samples:
        spec = ChartSpec(**{**s.meta["spec"], "style": "ood"})
        table = np.asarray(s.meta["table"], dtype=float)
        rr = render(spec, values=table, measure=True)
        img_rel = (out_dir / f"{s.id}.png").relative_to(DATA.parent).as_posix()
        rr.image.save(DATA.parent / img_rel)
        meas = rr.measure or {}
        px = float(meas.get("glyph_px_model", s.glyph_px_model))
        dx.append(abs(px - float(s.glyph_px_model)))
        bins[s.acuity_bin] = bins.get(s.acuity_bin, 0) + 1
        rows.append({
            **json.loads(json.dumps(s.__dict__, default=str)),
            "image_path": img_rel,
            "glyph_px_render": float(meas.get("glyph_px_render", s.glyph_px_render)),
            "glyph_px_model": px,
            "meta": {**s.meta, "style": "ood", "in_domain_px": s.glyph_px_model},
        })
    path = write_jsonl(DATA / "synth" / f"{OOD_SPLIT}.jsonl", rows)
    dstats = {
        "n": len(rows),
        "splits": {"dev_test": len(rows)},
        "acuity_bins": bins,
        "glyph_px_model": {
            "min": min((r["glyph_px_model"] for r in rows), default=0.0),
            "max": max((r["glyph_px_model"] for r in rows), default=0.0),
        },
        "px_drift_vs_in_domain": {
            "n": len(dx),
            "max": max(dx) if dx else 0.0,
            "mean": (sum(dx) / len(dx)) if dx else 0.0,
        },
        "note": ("同一批题换画风；model 侧字形像素应与 in-domain 一致"
                 "（字号没变，只有字体/配色/坐标轴变了）"),
        "file": path,
    }
    write_json(DATA / "ood_datacard.json", dstats)
    return dstats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    d = build(args.limit, Path(args.out_dir) if args.out_dir else None)
    print(f"[ood] {d['n']} 条 → {d['file']}")
    print(f"[ood] 分箱 {d['acuity_bins']}，字形像素 "
          f"{d['glyph_px_model']['min']:.2f}–{d['glyph_px_model']['max']:.2f} px")
    drift = d["px_drift_vs_in_domain"]
    print(f"[ood] 与 in-domain 的字形像素偏差 max={drift['max']:.2f}px "
          f"mean={drift['mean']:.2f}px（应为 0，说明只换了画风）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
