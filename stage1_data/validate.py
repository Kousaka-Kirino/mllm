"""S1 数据验收：把"数据能不能用来下结论"拆成 12 条可自动检查的断言。

跑法：
    python -m stage1_data.validate                # 默认 pilot
    python -m stage1_data.validate --profile full

它检查的不是"文件在不在"，而是：

- **可复算**：答案能不能从渲染用的底层数值表重新算出来（防止手滑改坏标注）；
- **无泄漏**：同一 family 的变体有没有跨 split（family = 一张底层表）；
- **平衡**：dev_test / locked_test 的 2×2 格是否精确平衡，族数够不够；
- **实测**：每条样本是否真的记录了模型输入侧的字形像素，分箱是否与冻结阈值一致；
- **冻结**：阈值文件与 datacard 里的是不是同一份。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from picovlm import io_utils as io
from picovlm.config import DATA, REPORTS, ROOT, load_config
from picovlm.report import FAIL, PASS, SKIP, Skip, approx, require, run
from picovlm.schema import Sample

from .make import fmt_num

SPLITS = ("train", "val", "dev_test", "locked_test")
MIN_FAMILIES_PER_CELL = {"full": 200, "pilot": 10}


# --------------------------------------------------------------------- 载入
def load_split(split: str, data_base: Path) -> list[Sample]:
    path = data_base / "synth" / f"{split}.jsonl"
    out: list[Sample] = []
    for row in io.read_jsonl(path):
        out.append(Sample(**row))
    return out


def recompute(s: Sample) -> str | None:
    """从 meta 里的底层表重算答案；没有表就返回 None（不硬编造）。"""
    table = s.meta.get("table")
    if not table:
        return None
    rows = table if isinstance(table[0], list) else [table]
    q = s.qtype
    if q == "value":
        j = int(s.meta["col"])
        return fmt_num(rows[0][j])
    if q == "compare":
        a, b = int(s.meta["a"]), int(s.meta["b"])
        return "yes" if rows[0][a] > rows[0][b] else "no"
    if q == "count":
        th = float(s.meta["threshold"])
        return str(sum(1 for v in rows[0] if v > th))
    if q == "maxcat":
        j = int(s.meta["argmax_col"])
        return chr(ord("A") + j)
    if q == "order":
        return str(s.meta.get("true_label"))
    if q == "maxseries":
        j = int(s.meta["col"])
        cols = [rows[i][j] for i in range(len(rows))]
        i = int(max(range(len(cols)), key=lambda k: cols[k]))
        s_ = s.meta.get("series_names")
        return str(s_[i]) if s_ else None
    return None


# --------------------------------------------------------------------- 检查
def check_files(root: Path, data_base: Path) -> str:
    missing = [s for s in SPLITS if not (data_base / "synth" / f"{s}.jsonl").exists()]
    require(not missing, f"缺 split 文件：{missing}")
    return f"4 个 split 文件都在（{data_base / 'synth'}）"


def check_counts(data: dict[str, list[Sample]], profile: str) -> str:
    parts = {k: len(v) for k, v in data.items()}
    require(min(parts.values()) > 0, f"有空 split：{parts}")
    return "  ".join(f"{k}={v}" for k, v in parts.items())


def check_ids(data: dict[str, list[Sample]]) -> str:
    ids = [s.id for v in data.values() for s in v]
    dup = [k for k, c in Counter(ids).items() if c > 1]
    require(not dup, f"重复 id：{dup[:3]}")
    return f"{len(ids)} 条样本 id 唯一"


def check_images(data: dict[str, list[Sample]], root: Path) -> str:
    bad, n = [], 0
    for v in data.values():
        for s in v:
            p = root / s.image_path
            if not p.exists():
                bad.append(f"{s.id}: 缺图 {s.image_path}")
                continue
            with Image.open(p) as im:
                if im.size != (s.image_size, s.image_size):
                    bad.append(f"{s.id}: 尺寸 {im.size} != {s.image_size}")
            n += 1
    require(not bad, f"{len(bad)} 张图有问题：{bad[:3]}")
    return f"{n} 条样本都有图，尺寸与 image_size 一致"


def check_answers(data: dict[str, list[Sample]]) -> str:
    bad, checked = [], 0
    for v in data.values():
        for s in v:
            gold = recompute(s)
            if gold is None:
                continue
            checked += 1
            if str(gold).strip() != str(s.answer).strip():
                bad.append(f"{s.id}[{s.qtype}] 重算={gold} 标注={s.answer}")
    require(not bad, f"{len(bad)} 条答案对不上：{bad[:3]}")
    return f"{checked} 条答案可由底层表重算"


def check_family_split(data: dict[str, list[Sample]]) -> str:
    where: dict[str, set] = defaultdict(set)
    for split, v in data.items():
        for s in v:
            where[s.family_id].add(split)
    leak = {f: sorted(sp) for f, sp in where.items() if len(sp) > 1}
    require(not leak, f"{len(leak)} 个 family 跨 split：{list(leak)[:3]}")
    return f"{len(where)} 个 family 全部单 split"


def check_balance(data: dict[str, list[Sample]], profile: str) -> str:
    need = MIN_FAMILIES_PER_CELL[profile]
    msgs, bad = [], []
    for split in ("dev_test", "locked_test"):
        cells = Counter()
        fams: dict[str, set] = defaultdict(set)
        for s in data[split]:
            key = f"{s.meta.get('cell')}/{s.meta.get('prior_role')}"
            cells[key] += 1
            fams[key].add(s.family_id)
        counts = sorted(cells.values())
        # 允许极小偏差：先验标签是从数据重算的（build._add 会把"自报 consistent
        # 但实际不成立"的样本改标为 conflict），改动后四格不可能永远严格整除。
        # 容差取每格数量的 0.2%（最少 1），超过就算失败。
        tol = max(1, round(0.002 * (sum(counts) / max(len(counts), 1))))
        if counts and counts[-1] - counts[0] > tol:
            bad.append(f"{split} 四格不均（差 {counts[-1] - counts[0]} > 容差 {tol}）：{dict(cells)}")
        for key, fs in fams.items():
            if len(fs) < need:
                bad.append(f"{split}/{key} 只有 {len(fs)} 个 family（目标 ≥{need}）")
        msgs.append(f"{split} 四格各 {counts[0] if counts else 0}")
    require(not bad, "；".join(bad[:3]))
    return "；".join(msgs)


def check_prior_labels(data: dict[str, list[Sample]]) -> str:
    bad = []
    for v in data.values():
        for s in v:
            if s.prior_type == "none":
                if s.prior_answer is not None:
                    bad.append(f"{s.id}: 无先验却带 prior_answer")
                continue
            if s.prior_answer is None:
                bad.append(f"{s.id}: 有先验但缺 prior_answer")
                continue
            role = s.meta.get("prior_role")
            holds = str(s.answer).strip() == str(s.prior_answer).strip()
            if (role == "consistent") != holds:
                bad.append(f"{s.id}: role={role} 但 holds={holds}")
    require(not bad, f"{len(bad)} 条先验标注不一致：{bad[:3]}")
    return "四类先验标注与 prior_holds 一致"


def check_pair_integrity(data: dict[str, list[Sample]]) -> str:
    """反事实对（pair_id）的现状：**每个 pair 在一个 split 里只留一条**。

    这是刻意的设计（2×2 是组间比较：每格用不同的图），所以这里断言的正是
    "不多于一条"；如果哪天改成成对投放，这条会失败并提醒你重写分析。
    """
    where: dict[str, set[str]] = {}
    count: Counter = Counter()
    for split, rows in data.items():
        for s in rows:
            if not s.pair_id:
                continue
            where.setdefault(s.pair_id, set()).add(split)
            count[s.pair_id] += 1
    cross = [p for p, sp in where.items() if len(sp) > 1]
    require(not cross, f"{len(cross)} 个 pair 跨 split：{cross[:3]}")
    multi = [p for p, c in count.items() if c > 1]
    require(not multi, f"{len(multi)} 个 pair 在同一 split 里有多条（当前设计不支持）：{multi[:3]}")
    return f"{len(count)} 个 pair 标签，均单条且不跨 split（组间设计）"


def check_acuity(data: dict[str, list[Sample]], th: dict) -> str:
    bad, bins = [], Counter()
    for v in data.values():
        for s in v:
            if not s.glyph_px_model or s.glyph_px_model <= 0:
                bad.append(f"{s.id}: glyph_px_model={s.glyph_px_model}")
                continue
            px = s.glyph_px_model
            want = ("extreme" if px < th["extreme_max_px_model"]
                    else "easy" if px >= th["easy_min_px_model"]
                    else "mid" if px >= th["mid_min_px_model"] else "hard")
            if s.acuity_bin != want:
                bad.append(f"{s.id}: bin={s.acuity_bin} 期望 {want}（{px}px）")
            bins[s.acuity_bin] += 1
    require(not bad, f"{len(bad)} 条分箱不对：{bad[:3]}")
    require(len(bins) >= 2, f"分箱只有 {dict(bins)}，acuity 轴没有变化")
    return f"实测像素分箱一致：{dict(bins)}"


def check_dedup(data: dict[str, list[Sample]], root: Path) -> str:
    paths = {s.image_path for v in data.values() for s in v}
    by_hash: dict[str, str] = {}
    bad = []
    for rel in paths:
        h = io.sha256_file(root / rel)
        if h in by_hash and by_hash[h] != rel:
            bad.append(f"{rel} 与 {by_hash[h]} 内容相同却不同名")
        by_hash[h] = rel
    require(not bad, f"{len(bad)} 组重复图片：{bad[:2]}")
    return f"{len(paths)} 张图按内容去重（sha256 唯一）"


def check_rho(data: dict[str, list[Sample]], card: dict) -> str:
    want = float(card["rho_requested"])
    got = float(card["rho_measured"])
    require(abs(got - want) <= 0.05, f"ρ 实测 {got} 偏离要求 {want} 超过 0.05")
    return f"ρ 实测 {got}（要求 {want}）"


def check_thresholds_frozen(root: Path, card: dict) -> str:
    path = root / "data" / "acuity_thresholds.json"
    require(path.exists(), "缺 data/acuity_thresholds.json（阈值未冻结）")
    disk = io.read_json(path)
    for k in ("easy_min_px_model", "mid_min_px_model", "extreme_max_px_model"):
        approx(float(disk[k]), float(card["acuity_thresholds"][k]), 1e-6, k)
    return (f"阈值已冻结：easy≥{disk['easy_min_px_model']}px "
            f"mid≥{disk['mid_min_px_model']}px extreme<{disk['extreme_max_px_model']}px")


def check_spotcheck(root: Path) -> str:
    d = root / "reports" / "acuity_spotcheck"
    files = list(d.glob("*.md")) if d.exists() else []
    if not files:
        raise Skip("没跑过目视核验：python -m stage1_data.measure --spotcheck 12")
    return f"目视核验记录 {len(files)} 份（{files[0].name}）"


# --------------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stage1_data.validate")
    ap.add_argument("--profile", default="pilot", choices=("pilot", "full"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--data-dir", default="data",
                    help="数据根目录名（pilot 用 data，full 用 data/full）")
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else ROOT
    data_base = root / args.data_dir
    card = io.read_json(data_base / "datacard.json")

    data = {s: load_split(s, data_base) for s in SPLITS}
    th = card["acuity_thresholds"]

    checks = [
        ("splits-present", lambda: check_files(root, data_base)),
        ("counts", lambda: check_counts(data, args.profile)),
        ("ids-unique", lambda: check_ids(data)),
        ("images-exist", lambda: check_images(data, root)),
        ("answers-recomputable", lambda: check_answers(data)),
        ("family-no-leak", lambda: check_family_split(data)),
        ("pair-integrity", lambda: check_pair_integrity(data)),
        ("test-balanced", lambda: check_balance(data, args.profile)),
        ("prior-labels", lambda: check_prior_labels(data)),
        ("acuity-measured", lambda: check_acuity(data, th)),
        ("image-dedup", lambda: check_dedup(data, root)),
        ("rho-matched", lambda: check_rho(data, card)),
        ("thresholds-frozen", lambda: check_thresholds_frozen(root, card)),
        ("spotcheck", lambda: check_spotcheck(root)),
    ]
    rc = run(checks, "S1 数据验收")

    # 顺手写一份给人看的报告
    io.ensure_dir(REPORTS)
    lines = [f"# S1 数据验收（profile={args.profile}, seed={card['seed']}）", "",
             f"- 样本：{card['counts']}", f"- family：{card['families']}",
             f"- ρ 实测：{card['rho_measured']}（要求 {card['rho_requested']}）",
             f"- 实测字形像素：{card['glyph_px_model']}",
             f"- 分箱：{card['acuity_bins']}", ""]
    (REPORTS / "data_validate.md").write_text("\n".join(lines), encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
