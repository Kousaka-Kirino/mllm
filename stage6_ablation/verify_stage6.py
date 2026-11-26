"""S6 验收：预算账本（token 数 / 参数 / 实测延迟与显存）。

`connector-bench` 读 `outputs/bench/connectors_*.json`，没有就 SKIP。
`pareto` 用写死的四个点检查 Pareto 前沿算法本身（不依赖 GPU）。
"""

from __future__ import annotations

import json

from picovlm.budget import CONNECTORS, count_tokens, patches
from picovlm.config import OUTPUTS
from picovlm.report import Skip, approx, require, run
from stage6_ablation.measure import stat
from stage6_ablation.pareto import Point, pareto_front


def c_tokens() -> str:
    n = patches(384, 16)
    require(n == 576, f"384/16 应有 576 个 patch，实际 {n}")
    require(count_tokens("mlp", n) == 576, "mlp 应 1:1")
    require(count_tokens("resampler", n, n_queries=64) == 64, "resampler 应为 query 数")
    require(count_tokens("pixelshuffle", n, factor=2) == 144, "pixelshuffle 2×2 应缩 4 倍")
    for name in ("mlp", "resampler", "pixelshuffle"):
        require(name in CONNECTORS, f"{name} 不在 CONNECTORS")
    return "576 / 64 / 144 三档与手算一致"


def c_pareto() -> str:
    pts = [Point("a", 576, 0.38), Point("b", 144, 0.30),
           Point("c", 64, 0.20), Point("d", 576, 0.10)]
    front = pareto_front(pts)
    require([p.name for p in front] == ["c", "b", "a"],
            f"前沿应为 c,b,a，实际 {[p.name for p in front]}")
    approx(front[0].tokens, 64, 1e-9, "最左点")
    require(pareto_front([]) == [], "空输入应为空")
    dup = pareto_front([Point("x", 100, 0.5), Point("y", 100, 0.5)])
    require(len(dup) == 1, "同 token 同精度应只留一个")
    return "去支配正确（同 token 低精度被淘汰）"


def c_math() -> str:
    s = stat([3.0, 1.0, 2.0])
    require(s["median"] == 2.0 and s["min"] == 1.0 and s["max"] == 3.0, "stat 不对")
    s2 = stat([5.0, 1.0, 9.0, 3.0])
    require(s2["median"] == 4.0, f"偶数个应取两中位均值，实际 {s2['median']}")
    return "median/min/max 手算一致"


def c_bench() -> str:
    files = sorted((OUTPUTS / "bench").glob("connectors_*.json"))
    if not files:
        raise Skip("没有 outputs/bench/connectors_*.json（先跑 stage6_ablation.measure）")
    rows = json.loads(files[-1].read_text(encoding="utf-8"))
    names = [r["connector"] for r in rows]
    require(set(names) == set(CONNECTORS), f"应覆盖三个连接器，实际 {names}")
    tok = {r["connector"]: r["tokens"] for r in rows}
    require(tok["mlp"] > tok["pixelshuffle"] > tok["resampler"],
            f"token 数应满足 mlp > shuffle > resampler，实际 {tok}")
    for r in rows:
        require(r["latency_ms"]["median"] > 0, f"{r['connector']} 延迟非正")
        require(r["peak_mib"]["median"] > 0, f"{r['connector']} 显存非正")
        require(r["accuracy"] is None, "未训练的档位不应报精度")
    lat = {r["connector"]: r["latency_ms"]["median"] for r in rows}
    return (f"{files[-1].name}: mlp {lat['mlp']:.0f}ms / shuffle "
            f"{lat['pixelshuffle']:.0f}ms / resampler {lat['resampler']:.0f}ms")


def main() -> int:
    return run([
        ("token-budget", c_tokens),
        ("pareto-front", c_pareto),
        ("stat-math", c_math),
        ("connector-bench", c_bench),
    ], "S6 验收（2.0）")


if __name__ == "__main__":
    raise SystemExit(main())
