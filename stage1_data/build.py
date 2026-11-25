"""S1 数据引擎入口：`python -m stage1_data.build [--full]`。

产出（都在 data/ 下，不进 git）：
    data/images/synth/*.png     渲染图
    data/synth/{train,dev,test}.jsonl
    data/datacard.json          数据卡：计数、旋钮、哈希、实测先验强度

设计约定：
- 训练集按 `prior_strength` 人为偏斜 → 先验是**被制造出来的**，不是碰巧存在；
- 测试集的 2×2 四格（easy/hard × consistent/conflict）每格等量，
  所以四格精度可以直接读，不需要加权；
- 所有 id 由 (split, 类型, 序号) 决定，重跑同参数得到同一批 id。
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from picovlm import io_utils as io
from picovlm.config import DATA, IMAGES, ROOT
from picovlm.config import DataConfig
from picovlm.schema import Sample
from picovlm.seeds import rng_for

from .conflict import fs_for_px
from .make import (Item, gen_compare, gen_count, gen_maxcat, gen_maxseries,
                   gen_order_pair, gen_value)

PRIOR_TYPES = ("position", "marginal", "numeric", "order")
NEUTRAL = ("value", "maxseries")
CELLS = {
    "easy_consistent": (12.0, True),
    "easy_conflict": (12.0, False),
    "hard_consistent": (5.0, True),
    "hard_conflict": (5.0, False),
}


def _order_item(seed: int, hit: bool, px: float) -> Item:
    a, b = gen_order_pair(seed, font_scale=fs_for_px(px))
    ok = a if a.question.answer == a.prior_answer else b
    bad = b if ok is a else a
    return ok if hit else bad


def make_item(prior_type: str, seed: int, hit: bool, px: float) -> Item:
    fs = fs_for_px(px)
    if prior_type == "position":
        return gen_maxcat(seed, prior_hit=hit, font_scale=fs)
    if prior_type == "marginal":
        return gen_compare(seed, prior_hit=hit, font_scale=fs)
    if prior_type == "numeric":
        return gen_count(seed, prior_hit=hit, font_scale=fs)
    if prior_type == "order":
        return _order_item(seed, hit, px)
    raise KeyError(prior_type)


class Builder:
    def __init__(self, seed: int, prior_strength: float, full: bool) -> None:
        self.seed = seed
        self.rho = prior_strength
        self.full = full
        self.n_train = DataConfig.n_train if full else 160
        self.n_dev = DataConfig.n_dev if full else 40
        self.n_cell = 100 if full else 24
        self.samples: list[Sample] = []
        self.serial: Counter = Counter()

    # ------------------------------------------------------------------ 工具
    def _next_id(self, split: str, tag: str) -> str:
        self.serial[(split, tag)] += 1
        return f"synth-{split}-{tag}-{self.serial[(split, tag)]:05d}"

    def _emit(self, split: str, item: Item, tag: str, suite: str | None = None) -> Sample:
        sid = self._next_id(split, tag)
        payload = item.to_payload()
        meta = dict(payload["meta"])
        if suite:
            meta["suite"] = suite
        img_rel = (IMAGES / "synth" / f"{sid}.png").relative_to(ROOT).as_posix()
        sample = Sample(
            id=sid, source="synth", split=split, image_path=img_rel,
            image_size=int(item.spec.size), question=payload["question"],
            answer=payload["answer"], answer_kind=payload["answer_kind"],
            qtype=payload["qtype"], prior_type=payload["prior_type"],
            prior_answer=payload["prior_answer"],
            acuity=round(item.spec.font_scale * 8.0, 3),
            pair_id=payload["pair_id"], meta=meta,
        )
        out = IMAGES / "synth" / f"{sid}.png"
        io.ensure_dir(out.parent)
        item.render().image.save(out, "PNG")
        self.samples.append(sample)
        return sample

    # ------------------------------------------------------------------ 划分
    def build_train(self) -> None:
        rng = rng_for("data", self.seed)
        per = max(2, self.n_train // (len(PRIOR_TYPES) + len(NEUTRAL)))
        px_choices = (5.0, 8.0, 12.0)
        for t in PRIOR_TYPES:
            for i in range(per):
                hit = bool(rng.random() < self.rho)
                px = float(rng.choice(px_choices))
                self._emit("train", make_item(t, self.seed * 101 + i, hit, px), t)
        for t in NEUTRAL:
            for i in range(per):
                s = self.seed * 211 + i
                item = gen_value(s) if t == "value" else gen_maxseries(s)
                item.spec = item.spec.__class__(**{**item.spec.__dict__,
                                                   "font_scale": fs_for_px(float(rng.choice(px_choices)))})
                self._emit("train", item, t)

    def build_dev(self) -> None:
        rng = rng_for("split", self.seed)
        per = max(2, self.n_dev // (len(PRIOR_TYPES) + len(NEUTRAL)))
        for t in PRIOR_TYPES:
            for i in range(per):
                hit = bool(rng.random() < 0.5)
                px = float(rng.choice((5.0, 8.0, 12.0)))
                self._emit("dev", make_item(t, self.seed * 307 + i, hit, px), t)
        for t in NEUTRAL:
            for i in range(per):
                s = self.seed * 401 + i
                self._emit("dev", gen_value(s) if t == "value" else gen_maxseries(s), t)

    def build_test(self) -> None:
        per_cell = max(1, self.n_cell // len(PRIOR_TYPES))
        for suite, (px, hit) in CELLS.items():
            for t in PRIOR_TYPES:
                for i in range(per_cell):
                    item = make_item(t, self.seed * 503 + i, hit, px)
                    self._emit("test", item, t, suite=suite)
        # 中性对照：无先验的纯净能力题
        for t in NEUTRAL:
            for i in range(self.n_cell // 2):
                s = self.seed * 601 + i
                item = gen_value(s) if t == "value" else gen_maxseries(s)
                self._emit("test", item, t, suite="neutral")

    # ------------------------------------------------------------------ 汇总
    def datacard(self) -> dict[str, Any]:
        by_split: dict[str, int] = Counter(s.split for s in self.samples)
        by_suite: dict[str, int] = Counter(s.meta.get("suite", "-") for s in self.samples)
        by_prior: dict[str, dict[str, int]] = defaultdict(Counter)
        by_qtype: dict[str, int] = Counter(s.qtype for s in self.samples)
        train = [s for s in self.samples if s.split == "train"]
        prior_rows = [s for s in train if s.prior_type != "none"]
        measured = (sum(1 for s in prior_rows if s.prior_answer == s.answer)
                    / len(prior_rows)) if prior_rows else 0.0
        majority: dict[str, str] = {}
        for s in train:
            majority.setdefault(s.qtype, Counter(
                x.answer for x in train if x.qtype == s.qtype).most_common(1)[0][0])
        for s in self.samples:
            by_prior[s.prior_type][s.pair_id is not None] += 1
        return {
            "project": "PicoVLM",
            "version": "0.1.0",
            "seed": self.seed,
            "config": {"full": self.full, "n_train": len(train), "n_dev": self.n_dev,
                       "n_cell": self.n_cell, "prior_strength_requested": self.rho},
            "counts": {"by_split": dict(by_split), "by_suite": dict(by_suite),
                       "by_qtype": dict(by_qtype),
                       "by_prior": {k: dict(v) for k, v in by_prior.items()}},
            "prior_strength_measured": round(measured, 4),
            "acuity_levels_px": {"easy": 12.0, "mid": 8.0, "hard": 5.0},
            "majority_answer_by_qtype": majority,
            "files": {},
        }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成 PicoVLM 合成数据")
    ap.add_argument("--seed", type=int, default=DataConfig.seed)
    ap.add_argument("--prior-strength", type=float, default=DataConfig.prior_strength)
    ap.add_argument("--full", action="store_true", help="生成完整规模（约 5k 张图）")
    args = ap.parse_args(argv)

    b = Builder(args.seed, args.prior_strength, args.full)
    b.build_train()
    b.build_dev()
    b.build_test()

    card = b.datacard()
    for split in ("train", "dev", "test"):
        rows = [s for s in b.samples if s.split == split]
        path = DATA / "synth" / f"{split}.jsonl"
        n = io.write_jsonl(path, (s.to_dict() for s in rows))
        card["files"][path.name] = {"rows": n, "sha256": io.sha256_file(path)}
    card_path = DATA / "datacard.json"
    io.write_json(card_path, card)

    print(f"[build] seed={args.seed} rho={args.prior_strength} full={args.full}")
    print(f"[build] splits={card['counts']['by_split']} suites={card['counts']['by_suite']}")
    print(f"[build] prior_strength_measured={card['prior_strength_measured']}")
    print(f"[build] datacard -> {card_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
