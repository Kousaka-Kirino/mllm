"""S7 · 评测：视觉记忆到底有没有用？

两个可测量的问题：

1. **页内召回**：拿某页的一个裁片当查询，能在 80 页里找回那一页吗？
   （recall@1 / recall@3；纯视觉，不发一次 LLM）
2. **放大有没有用**：对 hard 档（字号小）的题，把最"字多"的一块裁出来放大到
   384²，准确率会不会比整图高？（对照：随机裁一块）

第 2 问是 S1 acuity 恒等式的直接推论——如果模型的失败真是 acuity 造成的，
那么"裁+放大"应该立刻见效；如果不见效（或者只对某类题见效），就说明失败
不在像素，而在"没看"。这正是 RQ1 想区分的东西。

    python -m stage7_memory.evaluate --part all --limit 24
"""

from __future__ import annotations

import argparse
import json

import numpy as np
from PIL import Image

from picovlm.answers import is_correct
from picovlm.config import OUTPUTS, REPORTS, ROOT
from picovlm.io_utils import ensure_dir, read_jsonl, utc_stamp, write_json
from picovlm.stats import wilson_ci
from stage7_memory.retrieve import (Region, best_salient, crop_zoom, full_image,
                                    random_region)
from stage3_train.dataset import load_image


def _tower():
    import torch
    from picovlm.config import load_config, model_config_from
    from picovlm.model.vision import FrozenVisionTower
    mc = model_config_from(load_config("model.yaml"))
    t = FrozenVisionTower(mc.vision, image_size=mc.input_size).cuda().eval()
    return t, torch, mc


def _embed(tower, arrs: list[np.ndarray], torch, batch: int = 8) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(arrs), batch):
            px = torch.from_numpy(np.stack(arrs[i:i + batch])).cuda()
            f = tower(px).float().mean(dim=1)          # (B,D) 平均池化
            out.append(f.cpu().numpy())
    return np.concatenate(out, axis=0)


def recall_part(limit: int) -> dict:
    tower, torch, mc = _tower()
    samples = list(read_jsonl(ROOT / "data" / "synth" / "dev_test.jsonl"))[:limit]
    pages, crops = [], []
    for s in samples:
        img = Image.open(ROOT / s["image_path"]).convert("RGB")
        pages.append(full_image(img))
        w, h = img.size
        # 查询裁片：中心 1/3 区域（模拟"用户指着一段问"）
        crops.append(crop_zoom(img, Region(w // 3, h // 3, 2 * w // 3, 2 * h // 3)))
    P = _embed(tower, pages, torch)
    Q = _embed(tower, crops, torch)
    P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-9)
    Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
    S = Q @ P.T
    order = np.argsort(-S, axis=1)
    r1 = float(np.mean(order[:, 0] == np.arange(len(samples))))
    r3 = float(np.mean([i in order[i, :3] for i in range(len(samples))]))
    lo, hi = wilson_ci(int(round(r1 * len(samples))), len(samples))
    return {"n": len(samples), "recall_at_1": r1, "recall_at_3": r3,
            "recall_at_1_ci": [lo, hi]}


def zoom_part(limit: int, seed: int = 0) -> dict:
    from stage4_eval.predictor import PicoPredictor, load_pico
    model, tok = load_pico("sft", connector_ckpt="outputs/ckpt/sft_pilot/connector.pt",
                           adapter="outputs/ckpt/sft_pilot/lora")
    pred = PicoPredictor(model, tok, model_id="pico:sft")
    samples = [s for s in read_jsonl(ROOT / "data" / "synth" / "dev_test.jsonl")
               if s["acuity_bin"] == "hard"]
    rng = np.random.default_rng(seed)
    rng.shuffle(samples)
    samples = samples[:limit]

    rows = {"full": [], "salient": [], "random": []}
    for s in samples:
        img = Image.open(ROOT / s["image_path"]).convert("RGB")
        variants = {
            "full": full_image(img),
            "salient": crop_zoom(img, best_salient(img)),
            "random": crop_zoom(img, random_region(img, seed=int(rng.integers(1 << 30)))),
        }
        for name, arr in variants.items():
            raw = pred.answer(type("S", (), {"question": s["question"]})(), arr)
            rows[name].append(is_correct(raw, s["answer"], s["answer_kind"]))

    out = {}
    for name, vals in rows.items():
        k = int(sum(vals))
        lo, hi = wilson_ci(k, len(vals))
        out[name] = {"accuracy": k / max(1, len(vals)), "ci": [lo, hi],
                     "correct": k, "n": len(vals)}
    out["delta_salient_minus_full"] = (out["salient"]["accuracy"]
                                       - out["full"]["accuracy"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["recall", "zoom", "all"])
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)

    result: dict = {"stamp": utc_stamp()}
    if args.part in ("recall", "all"):
        result["recall"] = recall_part(args.limit)
    if args.part in ("zoom", "all"):
        result["zoom"] = zoom_part(args.limit, args.seed)

    out = ensure_dir(OUTPUTS / "memory") / f"memory_eval_{result['stamp']}.json"
    write_json(out, result)
    lines = ["# S7 · 视觉记忆评测", "", f"- 时间：{result['stamp']}", ""]
    if "recall" in result:
        r = result["recall"]
        lines += [f"- 页内召回（{r['n']} 页，查询=中心 1/3 裁片）："
                  f"recall@1 **{r['recall_at_1']:.3f}** "
                  f"[{r['recall_at_1_ci'][0]:.3f}, {r['recall_at_1_ci'][1]:.3f}]，"
                  f"recall@3 {r['recall_at_3']:.3f}", ""]
    if "zoom" in result:
        z = result["zoom"]
        lines += ["| 输入 | 准确率 | 95% CI | n |", "|---|---:|---|---:|"]
        for k in ("full", "salient", "random"):
            v = z[k]
            lines.append(f"| {k} | {v['accuracy']:.3f} | "
                         f"[{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] | {v['n']} |")
        lines += ["", f"salient − full = **{z['delta_salient_minus_full']:+.3f}**", ""]
    md = REPORTS / f"memory_{result['stamp']}.md"
    ensure_dir(REPORTS)
    md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"saved {out}\n      {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
