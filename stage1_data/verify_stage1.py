"""S1 验收：数据能不能用来下结论。

`python -m stage1_data.verify_stage1`

本体只做一件事——把 `stage1_data.validate` 的检查全跑一遍，再加上两条
"这一阶段特有"的检查（渲染确定性、配置与产物一致）。这样验收和报告的
口径永远不会分叉：验收过的就是报告里用的那一份数据。
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from picovlm import io_utils as io
from picovlm.config import DATA, ROOT, load_config
from picovlm.report import require, run
from picovlm.schema import Sample

from . import validate as V
from .render import render


def _load() -> tuple[dict[str, list[Sample]], dict]:
    card = io.read_json(DATA / "datacard.json")
    data = {s: V.load_split(s, DATA) for s in V.SPLITS}
    return data, card


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stage1_data.verify_stage1")
    ap.add_argument("--profile", default="pilot", choices=("pilot", "full"))
    args = ap.parse_args(argv)

    data, card = _load()
    th = card["acuity_thresholds"]

    def render_deterministic() -> str:
        """同一 spec + 同一数值表，两次渲染必须逐字节相同。"""
        from .make import gen_maxcat
        item = gen_maxcat(12345, prior_hit=True, font_scale=1.0, size=576)
        a = render(item.spec, item.values)
        b = render(item.spec, item.values)
        ha = hashlib.sha1(a.image.tobytes()).hexdigest()
        hb = hashlib.sha1(b.image.tobytes()).hexdigest()
        require(ha == hb, f"两次渲染不一致：{ha[:8]} vs {hb[:8]}")
        return f"384×384 渲染两次 sha1 相同（{ha[:8]}）"

    def measured_matches_scale() -> str:
        """实测像素必须随字号单调：小字号的行高不能反而更大。"""
        by_fs: dict[float, list[float]] = {}
        for v in data.values():
            for s in v:
                fs = round(float(s.meta.get("spec_font_scale", 0.0)), 3)
                if fs > 0:
                    by_fs.setdefault(fs, []).append(s.glyph_px_model)
        if len(by_fs) < 2:
            # 数据里没记 spec_font_scale 时退化为整体检查
            px = [s.glyph_px_model for v in data.values() for s in v if s.glyph_px_model]
            require(min(px) > 0 and max(px) / min(px) >= 1.5,
                    f"像素跨度太小：{min(px)}..{max(px)}")
            return f"实测字形像素 {min(px)}..{max(px)} px，跨度足够"
        xs = sorted(by_fs)
        med = [float(np.median(by_fs[x])) for x in xs]
        require(all(b >= a for a, b in zip(med, med[1:])),
                f"字号与像素不单调：{dict(zip(xs, med))}")
        return f"字号 {xs[0]}→{xs[-1]} 对应中位像素 {med[0]:.1f}→{med[-1]:.1f}"

    def config_matches() -> str:
        cfg = load_config("data.yaml")
        prof = cfg["profiles"][args.profile]
        got = card["counts"]
        for key, n in (("train", prof["n_train"]), ("val", prof["n_val"]),
                       ("dev_test", prof["n_dev_test"]), ("locked_test", prof["n_locked_test"])):
            # 生成器按"每种题型取整"分配，允许最多 6 条误差
            require(abs(got.get(key, 0) - int(n)) <= 6,
                    f"{key} 实际 {got.get(key)} 与配置 {n} 差太多")
        return f"四个 split 的条数与 {args.profile} 配置相符（±6）"

    checks = [
        ("splits-present", lambda: V.check_files(ROOT, DATA)),
        ("ids-unique", lambda: V.check_ids(data)),
        ("images-exist", lambda: V.check_images(data, ROOT)),
        ("answers-recomputable", lambda: V.check_answers(data)),
        ("family-no-leak", lambda: V.check_family_split(data)),
        ("test-balanced", lambda: V.check_balance(data, args.profile)),
        ("prior-labels", lambda: V.check_prior_labels(data)),
        ("acuity-measured", lambda: V.check_acuity(data, th)),
        ("acuity-geometry", measured_matches_scale),
        ("image-dedup", lambda: V.check_dedup(data, ROOT)),
        ("render-deterministic", render_deterministic),
        ("rho-matched", lambda: V.check_rho(data, card)),
        ("thresholds-frozen", lambda: V.check_thresholds_frozen(ROOT, card)),
        ("config-matches", config_matches),
    ]
    return run(checks, "S1 验收（2.0）")


if __name__ == "__main__":
    raise SystemExit(main())
