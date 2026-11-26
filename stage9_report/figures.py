"""把 RESULTS 里的核心数字画成图（报告用）。

两张图，都是"一页能看懂结论"的类型：
- `fig_2x2.png`：2×2 四格准确率（含 Wilson CI），配 prior 效应箭头。
- `fig_cost.png`：三个连接器的 token / 延迟 / 峰值显存。

用法：`python -m stage9_report.figures`
输出：`reports/figs/*.png`

刻意不画"训练 loss 曲线下降"这种图——它只能证明代码在跑，
证明不了任何研究问题。图要能支撑结论，否则不如不画。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Windows 上默认只有 DejaVu Sans（无中文），换成本机自带的中文字体；
# 缺字体时 matplotlib 会把中文画成方框，所以这里显式指定并关掉负号替换。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from picovlm import io_utils as io  # noqa: E402
from picovlm.config import OUTPUTS, REPORTS, ROOT  # noqa: E402

FIGS = REPORTS / "figs"

CELL_ORDER = ("easy_consistent", "easy_conflict", "hard_consistent", "hard_conflict")
CELL_LABEL = {
    "easy_consistent": "easy · consistent",
    "easy_conflict": "easy · conflict",
    "hard_consistent": "hard · consistent",
    "hard_conflict": "hard · conflict",
}


def _latest_grid() -> dict | None:
    """优先取 SFT 那一轮（主结果），没有就取最新的一轮。"""
    cands = sorted((OUTPUTS / "eval").glob("*/grid.json"))
    if not cands:
        return None
    preferred = [p for p in cands if "sft" in p.parent.name]
    return io.read_json((preferred or cands)[-1])


def fig_2x2(grid: dict, out: Path) -> Path:
    cells = grid["cells"]
    accs = np.array([cells[c]["acc"] for c in CELL_ORDER])
    lo = np.array([cells[c]["ci"][0] for c in CELL_ORDER])
    hi = np.array([cells[c]["ci"][1] for c in CELL_ORDER])
    counts = np.array([cells[c]["n"] for c in CELL_ORDER])
    mat = accs.reshape(2, 2)
    err = np.vstack([accs - lo, hi - accs]).reshape(2, 2, 2)

    fig, ax = plt.subplots(figsize=(6.0, 4.6), dpi=200)
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["prior consistent", "prior conflict"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["acuity easy\n(6.5–9 px)", "acuity hard\n(3–4.5 px)"])
    for i in range(2):
        for j in range(2):
            idx = i * 2 + j
            ax.text(j, i - 0.18, f"{mat[i, j]:.3f}\nn={counts[idx]}",
                    ha="center", va="center", fontsize=12, fontweight="bold")
            ax.errorbar(j, i + 0.16, yerr=err[i, j, :].reshape(2, 1), fmt="none",
                        ecolor="black", elinewidth=1.2, capsize=5)
    ax.set_title(
        f"2×2 accuracy — {grid.get('model_id', '')}\n"
        f"prior effect {grid['prior_effect']:+.3f}, "
        f"acuity effect {grid['acuity_effect']:+.3f}",
        fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="accuracy")
    ax.set_xlabel("答案是否与先验一致（列）")
    ax.set_ylabel("字形是否可辨（行）")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_cost(rows: list[dict], out: Path) -> Path:
    names = [r["connector"] for r in rows]
    toks = [r["tokens"] for r in rows]
    lat = [r["latency_ms_median"] for r in rows]
    vram = [r["peak_mib"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.4), dpi=200)
    for ax, vals, title, unit in (
        (axes[0], toks, "visual tokens", "个"),
        (axes[1], lat, "latency", "ms"),
        (axes[2], vram, "peak VRAM", "MiB"),
    ):
        bars = ax.bar(names, vals, color="#4c78a8")
        ax.set_title(f"{title}（{unit}）", fontsize=10)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=9)
        ax.margins(y=0.18)
        ax.tick_params(labelsize=9)
    fig.suptitle("连接器三档：token 越多不一定越慢（batch=1, 384²）", fontsize=11)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _bench_rows() -> list[dict]:
    files = sorted((OUTPUTS / "bench").glob("connectors_*.json"))
    if not files:
        return []
    rows = io.read_json(files[-1])
    keep = []
    for r in rows:
        lat = r.get("latency_ms") or {}
        vram = r.get("peak_mib") or {}
        keep.append({
            "connector": r["connector"],
            "tokens": r["tokens"],
            "latency_ms_median": lat.get("median", 0.0) if isinstance(lat, dict) else float(lat),
            "peak_mib": vram.get("median", 0.0) if isinstance(vram, dict) else float(vram),
        })
    return keep


def main() -> int:
    io.ensure_dir(FIGS)
    made: list[Path] = []
    grid = _latest_grid()
    if grid:
        made.append(fig_2x2(grid, FIGS / "fig_2x2.png"))
    rows = _bench_rows()
    if rows:
        made.append(fig_cost(rows, FIGS / "fig_cost.png"))
    for p in made:
        print(f"[figures] {p.relative_to(ROOT).as_posix()}")
    if not made:
        print("[figures] 没有可用数据：先跑 stage4_eval.run 与 stage6_ablation.measure")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
