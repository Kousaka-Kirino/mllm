"""S1 验收：答案能不能重算、先验是不是真的、有没有泄漏、渲染是否确定。

`python -m stage1_data.verify_stage1`

这一关过不了，后面所有实验数字都不可信——所以它检查得比别的阶段细。
"""

from __future__ import annotations

import hashlib
import sys
from collections import Counter, defaultdict

import numpy as np

from picovlm import io_utils as io
from picovlm.config import DATA, ROOT
from picovlm.report import Skip, approx, require, run
from picovlm.schema import Sample
from picovlm.seeds import rng_for

from .make import fmt_num

CELLS = ("easy_consistent", "easy_conflict", "hard_consistent", "hard_conflict")


def load() -> tuple[dict[str, list[Sample]], dict]:
    splits = {}
    for split in ("train", "dev", "test"):
        path = DATA / "synth" / f"{split}.jsonl"
        if not path.exists():
            raise Skip("数据未生成，先跑 python -m stage1_data.build")
        splits[split] = [Sample.from_dict(r) for r in io.read_jsonl(path)]
    card = io.read_json(DATA / "datacard.json")
    return splits, card


def recompute(s: Sample) -> str:
    """只用 meta 里的底层表与题面参数重算答案——不看模型、不看渲染图。"""
    m = s.meta
    table = np.asarray(m["table"], dtype=float)
    if s.qtype == "value":
        return fmt_num(table[0, int(m["col"])])
    if s.qtype == "compare":
        return "yes" if table[0, int(m["a"])] > table[0, int(m["b"])] else "no"
    if s.qtype == "count":
        return str(int((table[0] > float(m["threshold"])).sum()))
    if s.qtype == "maxcat":
        return chr(ord("A") + int(np.argmax(table[0])))
    if s.qtype == "order":
        return str(m["true_label"])
    if s.qtype == "maxseries":
        return f"S{int(m['argmax_series']) + 1}"
    raise AssertionError(f"未知 qtype {s.qtype}")


def main() -> int:
    splits, card = load()
    all_samples = [s for v in splits.values() for s in v]

    def files_ok() -> str:
        require((DATA / "datacard.json").exists(), "datacard.json 缺失")
        for k in ("train.jsonl", "dev.jsonl", "test.jsonl"):
            require((DATA / "synth" / k).exists(), f"{k} 缺失")
        return f"{len(all_samples)} 条样本"

    def counts_match() -> str:
        rows = card["files"]["train.jsonl"]["rows"]
        require(rows == len(splits["train"]), "train 行数与数据卡不符")
        total = sum(len(v) for v in splits.values())
        require(sum(card["counts"]["by_split"].values()) == total, "split 计数不符")
        return f"train={len(splits['train'])} test={len(splits['test'])}"

    def ids_unique() -> str:
        ids = [s.id for s in all_samples]
        dup = [k for k, v in Counter(ids).items() if v > 1]
        require(not dup, f"重复 id: {dup[:3]}")
        return f"{len(ids)} 个唯一 id"

    def images_exist() -> str:
        missing, wrong = [], []
        for s in all_samples:
            p = ROOT / s.image_path
            if not p.exists():
                missing.append(s.id)
                continue
            from PIL import Image
            with Image.open(p) as im:
                if im.size != (s.image_size, s.image_size):
                    wrong.append((s.id, im.size))
        require(not missing, f"缺图 {len(missing)} 张，如 {missing[:2]}")
        require(not wrong, f"{len(wrong)} 张尺寸不符，如 {wrong[:2]}")
        return f"{len(all_samples)} 张图尺寸一致"

    def answers_recompute() -> str:
        bad = []
        for s in all_samples:
            got = recompute(s)
            if str(got) != str(s.answer):
                bad.append((s.id, s.qtype, s.answer, got))
        require(not bad, f"{len(bad)} 条答案对不上，如 {bad[:3]}")
        return f"全部 {len(all_samples)} 条答案可由底层表重算"

    def prior_semantics() -> str:
        bad = []
        for s in all_samples:
            if s.prior_type == "none":
                continue
            holds = str(s.prior_answer) == str(s.answer)
            if s.pair_id and s.meta.get("pair_role"):
                expect = s.meta["pair_role"] == "consistent"
                if holds != expect:
                    bad.append((s.id, s.meta["pair_role"], holds))
            if s.meta.get("suite") == "easy_conflict" or s.meta.get("suite") == "hard_conflict":
                if holds:
                    bad.append((s.id, "conflict 但先验成立"))
            if s.meta.get("suite") in ("easy_consistent", "hard_consistent"):
                if not holds:
                    bad.append((s.id, "consistent 但先验不成立"))
        require(not bad, f"{len(bad)} 条先验标注错误，如 {bad[:3]}")
        return "四格先验标注与构造一致"

    def no_leak() -> str:
        where: dict[str, set[str]] = defaultdict(set)
        for s in all_samples:
            if s.pair_id:
                where[s.pair_id].add(s.split)
        leak = {k: v for k, v in where.items() if len(v) > 1}
        require(not leak, f"{len(leak)} 个 pair 跨 split，如 {list(leak)[:2]}")
        img_split: dict[str, set[str]] = defaultdict(set)
        for s in all_samples:
            img_split[s.image_path].add(s.split)
        require(not {k for k, v in img_split.items() if len(v) > 1}, "同一张图跨 split")
        return f"{len(where)} 个 pair 均在单一 split 内"

    def prior_strength() -> str:
        req = card["config"]["prior_strength_requested"]
        got = card["prior_strength_measured"]
        approx(got, req, 0.12, "prior_strength")
        return f"requested={req} measured={got}"

    def acuity_formula() -> str:
        for s in all_samples:
            approx(s.acuity, 8.0 * s.meta["spec"]["font_scale"], 0.06, s.id)
        return "acuity == 8 × font_scale（解析式成立）"

    def order_pair_same_image() -> str:
        groups: dict[str, list[Sample]] = defaultdict(list)
        for s in all_samples:
            if s.pair_id and s.pair_id.startswith("ord"):
                groups[s.pair_id].append(s)
        require(groups, "没有 order pair")
        for pid, rows in groups.items():
            if len(rows) != 2:
                continue
            h = {io.sha256_file(ROOT / r.image_path) for r in rows}
            require(len(h) == 1, f"{pid} 两条样本图像不一致")
            require(len({r.question.text if hasattr(r.question, 'text') else r.question
                         for r in rows}) == 2, f"{pid} 问法没有变化")
        return f"{len(groups)} 组顺序对共享同一张图"

    def render_determinism() -> str:
        from .render import ChartSpec, render
        s = splits["train"][0]
        spec = ChartSpec(**s.meta["spec"])
        values = np.asarray(s.meta["table"], dtype=float)
        a = np.asarray(render(spec, values).image)
        b = np.asarray(render(spec, values).image)
        require(np.array_equal(a, b), "两次渲染不一致")
        from PIL import Image
        with Image.open(ROOT / s.image_path) as im:
            require(np.array_equal(np.asarray(im.convert("RGB")), a), "落盘图与重渲染不一致")
        return f"{s.id} 逐像素可复现"

    def cell_balance() -> str:
        rows = [s for s in splits["test"] if s.meta.get("suite") in CELLS]
        cnt = Counter(r.meta["suite"] for r in rows)
        require(len(set(cnt.values())) == 1, f"四格不均衡：{dict(cnt)}")
        return f"每格 {cnt[CELLS[0]]} 条"

    def answers_in_options() -> str:
        bad = [s.id for s in all_samples if s.meta.get("options")
               and str(s.answer) not in [str(o) for o in s.meta["options"]]]
        require(not bad, f"答案不在候选集里：{bad[:3]}")
        return "答案均落在候选集内"

    return run([
        ("dataset-files", files_ok),
        ("counts-match", counts_match),
        ("ids-unique", ids_unique),
        ("images-exist", images_exist),
        ("answers-recompute", answers_recompute),
        ("prior-semantics", prior_semantics),
        ("no-split-leak", no_leak),
        ("prior-strength", prior_strength),
        ("acuity-formula", acuity_formula),
        ("order-pair-image", order_pair_same_image),
        ("render-determinism", render_determinism),
        ("cell-balance", cell_balance),
        ("answers-in-options", answers_in_options),
    ], title="stage1_data")


if __name__ == "__main__":
    sys.exit(main())
