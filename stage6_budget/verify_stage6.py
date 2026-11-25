"""S6 验收：Pareto 前沿、iso-token 选择、token 账本与连接器一一对应。"""

from __future__ import annotations

import sys

from picovlm.budget import count_tokens, iso_token_sweep
from picovlm.config import ModelConfig
from picovlm.report import require, run

from .pareto import Point, gains, iso_token, pareto_front


def main() -> int:
    def token_counts() -> str:
        cfg = ModelConfig()
        got = {c: count_tokens(c, cfg.n_patches, cfg.n_queries, cfg.shuffle_factor)
               for c in ("mlp", "resampler", "pixelshuffle")}
        require(got == {"mlp": 576, "resampler": 64, "pixelshuffle": 144},
                f"token 账本漂了：{got}")
        return f"mlp 576 / resampler 64 / pixelshuffle 144"

    def sweep_consistent() -> str:
        cfg = ModelConfig()
        rows = iso_token_sweep(cfg.input_size, cfg.patch)
        require(len(rows) == 3, f"应有 3 档，得到 {len(rows)}")
        require(rows[0]["tokens"] < rows[-1]["tokens"], "未按 token 升序")
        require(all(0 < r["ratio"] <= 1 for r in rows), "ratio 越界")
        return " / ".join(f"{r['connector']}:{r['tokens']}" for r in rows)

    def pareto_works() -> str:
        pts = [Point("a", 576, 0.80), Point("b", 144, 0.78),
               Point("c", 64, 0.70), Point("d", 576, 0.75)]
        front = pareto_front(pts)
        names = [p.name for p in front]
        require(names == ["c", "b", "a"], f"前沿错：{names}")
        require("d" not in names, "被支配点不应在前沿")
        return " -> ".join(f"{p.name}({p.tokens})" for p in front)

    def iso_token_pick() -> str:
        pts = [Point("a", 576, 0.80), Point("b", 144, 0.78), Point("c", 64, 0.70)]
        best = iso_token(pts, 144)
        require(best is not None and best.name == "b", f"iso-token 选错：{best}")
        require(iso_token(pts, 32) is None, "预算过小应返回 None")
        return "B=144 选 b"

    def gain_math() -> str:
        pts = [Point("a", 576, 0.80), Point("c", 64, 0.70)]
        g = gains(pts)
        require(abs(g["token_saving"] - (1 - 64 / 576)) < 1e-9, "省 token 比例错")
        require(abs(g["accuracy_drop"] - 0.10) < 1e-9, "掉点计算错")
        return f"省 {g['token_saving']:.1%} token，掉 {g['accuracy_drop']:.1%}"

    return run([
        ("token-counts", token_counts),
        ("sweep", sweep_consistent),
        ("pareto-front", pareto_works),
        ("iso-token", iso_token_pick),
        ("gain-math", gain_math),
    ], title="stage6_budget")


if __name__ == "__main__":
    sys.exit(main())
