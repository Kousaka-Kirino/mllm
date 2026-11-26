"""S1 数据构建：从"一张表"到"一个带 family_id 的样本"，一次成型。

2.0 相对 v1 的四点改造（都是为了让结论站得住）：

1. **family 划分**：一个 family = 一张底层数值表 + 它的模板族。同一个 family 的样本
   只能落在一个 split 里，否则"同一张表的变体"会横跨训练与测试（2.0 §4.4）。
   实现上每个 split 用互不相交的种子段生成，family_id 由种子确定性算出，
   于是 `validate` 只要检查"family 不跨 split"就能证明没泄漏。
2. **四个 split**：train / val / dev_test / locked_test（80/10/5/5）。
   dev_test 用来选配置，locked_test 只在分析方案冻结后跑一次。
3. **实测字形像素**：每条样本都记录渲染尺度与模型尺度上的实际着墨高度
   （`stage1_data.measure`），acuity 档位按实测值分箱，不看 font_scale 参数名。
4. **先验可扫描**：训练集的 ρ（先验答案正好正确的比例）由配置给，测试集四格平衡。

产物（都可以由 `--seed` 重建）：
    data/synth/{train,val,dev_test,locked_test}.jsonl
    data/synth/images/<sha1>.png       （按内容去重，顺序对共用同一张图）
    data/datacard.json                 （计数、ρ 实测、阈值、像素分布）
    data/hashes.json                   （全量 sha256，供 validate 复核）
    data/acuity_thresholds.json        （分箱阈值，写入即冻结）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from picovlm import io_utils as io
from picovlm.config import CONFIGS, DATA, ROOT, load_config
from picovlm.schema import ACUITY_BINS, Sample

from .make import (Item, draw_values, gen_compare, gen_count, gen_maxcat,
                   gen_maxseries, gen_order_pair, gen_value)
from .render import ChartSpec, render

PRIOR_TYPES = ("position", "marginal", "numeric", "order")
NEUTRAL_TYPES = ("value", "maxseries")
CELL_FONT_SCALES = {
    "easy": (1.15, 1.3, 1.45),     # 实测着墨高约 7-9 px（模型输入侧）：模型"看得见"
    "hard": (0.5, 0.6, 0.7),       # 实测约 3-4.3 px：模型"看不清"
}

# 分箱阈值：**预注册的绝对像素**，不随数据分布漂移（2.0 §4.2）。
# 依据：pilot 实测着墨高 ≈ 6.2 × font_scale；acuity ladder [4,5,6,7,8,10,12,16]
# 收敛成四个可操作区间。改这里 == 改实验定义，必须重跑全部结果。
EASY_MIN_PX = 6.5
MID_MIN_PX = 4.5
EXTREME_MAX_PX = 3.0
CELLS = ("easy", "hard")
PRIORS = ("consistent", "conflict")


@dataclass
class RawRec:
    """落盘前的记录：Sample 的字段 + 实测像素。"""

    split: str
    cell: str
    prior: str
    family_id: str
    payload: dict[str, Any]
    image_path: str
    image_size: int
    glyph_px_render: float
    glyph_px_model: float
    pixel_stats: dict[str, Any]
    tag: str


# --------------------------------------------------------------------- 工具
def family_id_of(spec: ChartSpec, values: np.ndarray) -> str:
    """family = 底层数值表（而不是图片）：同一张表的题干变体必须同族。"""
    key = f"{spec.seed}|{spec.n_categories}|{spec.n_series}|{spec.kind}"
    tab = np.asarray(values, dtype=np.float64).round(4).tobytes()
    return "fam-" + hashlib.sha1(key.encode() + tab).hexdigest()[:12]


def seed_of(*parts: Any) -> int:
    """把 (seed, salt, 类型, 序号) 搅成一个不重叠的整数种子。

    直接用乘法加法拼种子会出现"跨 split 撞号"（实测过：train 的
    value 第 51 条与 val 的 position 第 0 条撞成同一个 seed），
    所以这里用哈希，让不同 namespace 的种子天然分散。
    """
    raw = "|".join(str(p) for p in parts).encode()
    return int(hashlib.blake2b(raw, digest_size=8).hexdigest(), 16) % (2**31 - 1)


def _order_item(seed: int, hit: bool, fs: float) -> Item:
    a, b = gen_order_pair(seed, font_scale=fs)
    ok = a if a.question.answer == a.prior_answer else b
    bad = b if ok is a else a
    return ok if hit else bad


def make_item(prior_type: str, seed: int, hit: bool, fs: float) -> Item:
    """按先验类型造一条"先验成立/不成立"的样本。fs 是逻辑字号倍数。"""
    if prior_type == "position":
        return gen_maxcat(seed, prior_hit=hit, font_scale=fs)
    if prior_type == "marginal":
        return gen_compare(seed, prior_hit=hit, font_scale=fs)
    if prior_type == "numeric":
        return gen_count(seed, prior_hit=hit, font_scale=fs)
    if prior_type == "order":
        return _order_item(seed, hit, fs)
    raise KeyError(prior_type)


def neutral_item(tag: str, seed: int, fs: float) -> Item:
    if tag == "value":
        item = gen_value(seed, font_scale=fs)
    else:
        item = gen_maxseries(seed, font_scale=fs)
    return item


def compute_thresholds(px: list[float]) -> dict[str, Any]:
    """预注册阈值 + 实测分布摘要。

    阈值是**常量**（上面的 EASY_MIN_PX 等），不会因为数据里恰好没有某个字号
    就把边界挪过去；实测分布只用于记录"这套数据真的覆盖到了哪一段"。
    """
    arr = np.asarray(sorted(px), dtype=float)
    if arr.size < 8:
        raise ValueError(f"实测像素太少（{arr.size}），无法分箱")
    cover = {
        "below_extreme": int((arr < EXTREME_MAX_PX).sum()),
        "hard": int(((arr >= EXTREME_MAX_PX) & (arr < MID_MIN_PX)).sum()),
        "mid": int(((arr >= MID_MIN_PX) & (arr < EASY_MIN_PX)).sum()),
        "easy": int((arr >= EASY_MIN_PX).sum()),
    }
    return {
        "easy_min_px_model": EASY_MIN_PX,
        "mid_min_px_model": MID_MIN_PX,
        "extreme_max_px_model": EXTREME_MAX_PX,
        "n_measurements": int(arr.size),
        "median": round(float(np.median(arr)), 3),
        "p05": round(float(np.percentile(arr, 5)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "coverage": cover,
        "rule": "easy: px >= easy_min; mid: mid_min <= px < easy_min; "
                "hard: extreme_max <= px < mid_min; extreme: px < extreme_max",
        "source": "预注册常量（pilot 实测着墨高≈6.2×font_scale 时校准），写入即冻结",
    }


def bin_of(px: float, th: dict[str, Any]) -> str:
    if px < th["extreme_max_px_model"]:
        return "extreme"
    if px >= th["easy_min_px_model"]:
        return "easy"
    if px >= th["mid_min_px_model"]:
        return "mid"
    return "hard"


# --------------------------------------------------------------------- 构建
class Builder:
    def __init__(self, seed: int, profile: str = "pilot", rho: float = 0.7,
                 cfg: dict | None = None, out: "Path | str" = "data") -> None:
        cfg = cfg or load_config("data.yaml")
        prof = cfg["profiles"][profile]
        self.seed = seed
        self.profile = profile
        self.rho = rho
        self.out_root = ROOT / str(out)
        self.img_dir = self.out_root / "synth" / "images"
        self.n = {k: int(prof[k]) for k in
                  ("n_train", "n_val", "n_dev_test", "n_locked_test")}
        self.size = int(cfg["render"]["size"])
        self.records: list[RawRec] = []
        self._img_cache: dict[str, str] = {}     # 内容哈希 -> 相对路径
        self._measure_cache: dict[str, dict] = {}
        self._family_seen: dict[str, str] = {}   # family_id -> split
        self.serial: Counter = Counter()

    # ------------------------------------------------------------------ 图像
    def _save_image(self, item: Item) -> tuple[str, dict]:
        res = render(item.spec, item.values, measure=True)
        h = hashlib.sha1(res.image.tobytes()).hexdigest()[:16]
        if h not in self._img_cache:
            rel = (self.img_dir / f"{h}.png").relative_to(ROOT).as_posix()
            out = ROOT / rel
            io.ensure_dir(out.parent)
            res.image.save(out, "PNG")
            self._img_cache[h] = rel
        return self._img_cache[h], (res.measure or {})

    # ------------------------------------------------------------------ 记录
    def _add(self, split: str, cell: str, prior: str, item: Item, tag: str) -> None:
        payload = item.to_payload()
        fam = family_id_of(item.spec, item.values)
        prev = self._family_seen.setdefault(fam, split)
        if prev != split:
            raise AssertionError(
                f"family {fam} 出现在 {prev} 与 {split}：种子段重叠了，必须修生成逻辑")
        img_rel, meas = self._save_image(item)
        self.records.append(RawRec(
            split=split, cell=cell, prior=prior, family_id=fam, payload=payload,
            image_path=img_rel, image_size=int(item.spec.size),
            glyph_px_render=float(meas.get("glyph_px_render") or 0.0),
            glyph_px_model=float(meas.get("glyph_px_model") or 0.0),
            pixel_stats={k: meas.get(k) for k in
                         ("n_acuity_regions", "iqr_model", "glyph_px_min_model",
                          "glyph_px_max_model")},
            tag=tag,
        ))

    # ------------------------------------------------------------------ 生成
    def _rng(self, salt: int) -> np.random.Generator:
        return np.random.default_rng(self.seed * 1009 + salt)

    def build_train_like(self, split: str, count: int, salt: int) -> None:
        """train / val：先验按 ρ 偏斜，字形与题型混合，模拟"有偏训练分布"。"""
        rng = self._rng(salt)
        per_type = max(1, count // (len(PRIOR_TYPES) + len(NEUTRAL_TYPES)))
        for ti, ptype in enumerate(PRIOR_TYPES):
            for i in range(per_type):
                hit = bool(rng.random() < self.rho)
                cell = str(rng.choice(CELLS, p=(0.5, 0.5)))
                fs = float(rng.choice(CELL_FONT_SCALES[cell]))
                seed = seed_of(self.seed, "trainlike", salt, ptype, i)
                item = make_item(ptype, seed, hit, fs)
                self._add(split, cell, "consistent" if hit else "conflict",
                          item, f"{ptype}")
        for ti, tag in enumerate(NEUTRAL_TYPES):
            for i in range(per_type):
                fs = float(rng.choice((0.7, 1.0, 1.45)))
                seed = seed_of(self.seed, "neutral", salt, tag, i)
                self._add(split, "any", "none", neutral_item(tag, seed, fs), tag)

    def build_balanced(self, split: str, count: int, salt: int) -> None:
        """dev_test / locked_test：2×2 精确平衡，每个 family 只出一题（独立样本）。"""
        per_cell = max(1, count // (len(CELLS) * len(PRIORS)))
        for cell in CELLS:
            for prior in PRIORS:
                for i in range(per_cell):
                    hit = prior == "consistent"
                    ptype = PRIOR_TYPES[(i + (0 if cell == "easy" else 2)) % len(PRIOR_TYPES)]
                    fs = float(CELL_FONT_SCALES[cell][i % len(CELL_FONT_SCALES[cell])])
                    seed = seed_of(self.seed, "balanced", salt, cell, prior, i)
                    item = make_item(ptype, seed, hit, fs)
                    self._add(split, cell, prior, item, f"{cell}-{prior}-{ptype}")

    # ------------------------------------------------------------------ 收尾
    def finalize(self, thresholds: dict | None = None) -> tuple[list[Sample], dict, dict]:
        """分箱、写 jsonl、算数据卡。返回 (samples, datacard, thresholds)。

        thresholds 传入时直接沿用（full 构建必须复用 pilot 冻结的阈值，
        否则分箱会跟着数据分布漂移，跨规模结果不可比）。
        """
        th = thresholds or compute_thresholds([r.glyph_px_model for r in self.records])
        samples: list[Sample] = []
        serial: Counter = Counter()
        for rec in self.records:
            serial[(rec.split, rec.tag)] += 1
            sid = f"synth-{rec.split}-{rec.tag}-{serial[(rec.split, rec.tag)]:05d}"
            p = rec.payload
            meta = dict(p["meta"])
            spec_meta = p["meta"].get("spec", {})
            meta.update({"cell": rec.cell, "prior_role": rec.prior,
                         "family_id": rec.family_id,
                         "spec_font_scale": float(spec_meta.get("font_scale", 0.0)),
                         "spec_kind": spec_meta.get("kind"),
                         "glyph_px_measured": rec.glyph_px_model,
                         "pixel_stats": rec.pixel_stats})
            samples.append(Sample(
                id=sid, source="synth", split=rec.split, image_path=rec.image_path,
                image_size=rec.image_size, question=p["question"], answer=p["answer"],
                answer_kind=p["answer_kind"], qtype=p["qtype"],
                prior_type=p["prior_type"], prior_answer=p["prior_answer"],
                family_id=rec.family_id,
                glyph_px_render=round(rec.glyph_px_render, 3),
                glyph_px_model=round(rec.glyph_px_model, 3),
                acuity_bin=bin_of(rec.glyph_px_model, th),
                pixel_stats=rec.pixel_stats, pair_id=p["pair_id"], meta=meta,
            ))
        return samples, self._datacard(samples, th), th

    def _datacard(self, samples: list[Sample], th: dict) -> dict:
        by_split: dict[str, int] = defaultdict(int)
        by_cell: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        by_prior: dict[str, int] = defaultdict(int)
        by_bin: dict[str, int] = defaultdict(int)
        fams: dict[str, set] = defaultdict(set)
        for s in samples:
            by_split[s.split] += 1
            by_cell[s.split][f"{s.acuity_bin}/{('conflict' if s.is_conflict else ('consistent' if s.prior_type != 'none' else 'neutral'))}"] += 1
            by_prior[s.prior_type] += 1
            by_bin[s.acuity_bin or "?"] += 1
            fams[s.split].add(s.family_id)
        train = [s for s in samples if s.split == "train"]
        cons = [s for s in train if s.prior_type != "none" and s.prior_holds]
        with_prior = [s for s in train if s.prior_type != "none"]
        px = [s.glyph_px_model for s in samples if s.glyph_px_model]
        return {
            "profile": self.profile,
            "seed": self.seed,
            "rho_requested": self.rho,
            "rho_measured": round(len(cons) / max(1, len(with_prior)), 4),
            "counts": dict(by_split),
            "families": {k: len(v) for k, v in fams.items()},
            "cells": {k: dict(v) for k, v in by_cell.items()},
            "prior_types": dict(by_prior),
            "acuity_bins": dict(by_bin),
            "glyph_px_model": {
                "n": len(px),
                "median": round(float(np.median(px)), 3),
                "min": round(float(np.min(px)), 3),
                "max": round(float(np.max(px)), 3),
                "iqr": round(float(np.percentile(px, 75) - np.percentile(px, 25)), 3),
            },
            "acuity_thresholds": th,
            "images": len(self._img_cache),
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stage1_data.build")
    ap.add_argument("--profile", default="pilot", choices=("pilot", "full"))
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--prior-strength", type=float, default=0.7)
    ap.add_argument("--out", default="data")
    ap.add_argument("--recalibrate", action="store_true",
                    help="重新计算 acuity 阈值（默认复用 data/acuity_thresholds.json）")
    args = ap.parse_args(argv)

    frozen = None
    th_path = DATA / "acuity_thresholds.json"
    if th_path.exists() and not args.recalibrate and args.profile == "full":
        frozen = io.read_json(th_path)
        print(f"[build] 复用冻结阈值：{th_path.relative_to(ROOT)}")

    b = Builder(seed=args.seed, profile=args.profile, rho=args.prior_strength,
                out=args.out)
    b.build_train_like("train", b.n["n_train"], salt=1)
    b.build_train_like("val", b.n["n_val"], salt=2)
    b.build_balanced("dev_test", b.n["n_dev_test"], salt=3)
    b.build_balanced("locked_test", b.n["n_locked_test"], salt=4)
    samples, card, thresholds = b.finalize(frozen)

    out_root = ROOT / args.out
    out = out_root / "synth"
    io.ensure_dir(out)
    hashes: dict[str, str] = {}
    for split in ("train", "val", "dev_test", "locked_test"):
        part = [s for s in samples if s.split == split]
        path = out / f"{split}.jsonl"
        io.write_jsonl(path, [s.to_dict() for s in part])
        hashes[path.relative_to(ROOT).as_posix()] = io.sha256_file(path)
        print(f"[build] {split:<11} {len(part):>5} 条  family {len({s.family_id for s in part}):>4}")
    for img in sorted(b.img_dir.glob("*.png"))[:5]:
        hashes[img.relative_to(ROOT).as_posix()] = io.sha256_file(img)

    io.write_json(out_root / "datacard.json", card)
    io.write_json(DATA / "acuity_thresholds.json", thresholds)
    io.write_json(out_root / "hashes.json", {"seed": args.seed, "profile": args.profile,
                                             "files": hashes})
    print(f"[build] 图 {card['images']} 张；ρ 实测 {card['rho_measured']}（要求 {args.prior_strength}）")
    print(f"[build] acuity 阈值 easy≥{thresholds['easy_min_px_model']}px "
          f"mid≥{thresholds['mid_min_px_model']}px extreme<{thresholds['extreme_max_px_model']}px")
    print(f"[build] 实测覆盖 {thresholds.get('coverage')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
