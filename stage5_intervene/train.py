"""S5 · 三种干预的统一入口。

    python -m stage5_intervene.train --method curriculum --steps 20
    python -m stage5_intervene.train --method counterfactual --steps 20
    python -m stage5_intervene.train --method grpo --steps 10 --group-size 2

三种干预的**唯一区别是"喂什么数据/给什么信号"**，模型、优化器、评测都不变——
这是"受控实验"的要求：变一个因子，其他全部钉死。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from picovlm.config import OUTPUTS, load_config
from picovlm.io_utils import ensure_dir, utc_stamp, write_json
from picovlm.model.vlm import Batch
from picovlm.schema import Sample

from stage3_train.align import build_model, make_loader, to_device
from stage3_train.dataset import (PROMPT, ChartSFT, EOS_SUFFIX, load_image,
                                  load_samples)
from stage5_intervene import grpo as G


def load_sft(cfg, device):
    """从 SFT pilot 继续训：载入连接器与 LoRA 适配器。"""
    from peft import PeftModel

    model, tok = build_model(cfg, load_weights=True, device=device)
    ck = OUTPUTS / "ckpt" / "sft_pilot" / "connector.pt"
    if ck.exists():
        model.connector.load_state_dict(torch.load(ck, map_location="cpu"))
        print(f"connector <- {ck}")
    ad = OUTPUTS / "ckpt" / "sft_pilot" / "lora"
    if ad.exists():
        model.llm = PeftModel.from_pretrained(model.llm, ad)
        print(f"lora       <- {ad}")
    model.connector.train()
    return model, tok


def trainable(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def sft_step(model, tok, batch, device) -> float:
    out = model(to_device(batch, device))
    loss = out["loss"]
    loss.backward()
    return float(loss.detach())


def curriculum_order(samples: list[Sample], shuffle_seed: int = 0,
                     ratio: float = 0.4) -> list[Sample]:
    """前 40% 步数只用"易"样本，之后混合全部——从粗到细的课程。"""
    easy = [s for s in samples if s.acuity_bin in ("easy", "mid")]
    hard = [s for s in samples if s.acuity_bin == "hard"]
    rng = random.Random(shuffle_seed)
    rng.shuffle(easy)
    rng.shuffle(hard)
    n = int(len(samples) * ratio)
    head = easy[:n]
    # 头段为易样本：不够就用 hard 补，但仍然"先易后难"
    head = head + hard[:max(0, n - len(head))]
    tail = [s for s in samples if s not in head]
    rng.shuffle(tail)
    return head + tail


def augment_counterfactual(samples: list[Sample]) -> list[Sample]:
    """原样本 + 措辞变体（金标不变）+ maxcat 数值变体（金标改变）。"""
    from stage5_intervene.counterfactual import numeric_variant, wording_variant

    out: list[Sample] = []
    for s in samples:
        out.append(s)
        w = wording_variant(s)
        if w:
            out.append(w)
        if s.qtype == "maxcat":
            nv = numeric_variant(s)
            if nv:
                out.append(nv)
    return out


def run_grpo(model, tok, samples, *, steps: int, group_size: int = 2,
             device: str = "cuda", lr: float = 1e-5, image_size: int = 384,
             conflict_weight: float = 1.0, log_path: Path | None = None) -> dict:
    from picovlm.answers import is_correct

    params = trainable(model)
    opt = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(0)
    stats = {"steps": 0, "reward_mean": 0.0, "correct_rate": 0.0,
             "format_rate": 0.0, "zero_adv_groups": 0}
    t0 = time.time()
    hist = []
    for step in range(steps):
        s = samples[rng.randrange(len(samples))]
        px = torch.from_numpy(load_image(s.image_path, image_size)[None]).to(device)
        ids = tok(PROMPT.format(q=s.question), return_tensors="pt",
                  add_special_tokens=False)["input_ids"].to(device)
        # 1) 同一份像素做 G 次采样
        gen = G.rollout(model, tok, px, ids, group_size=group_size)
        texts = [tok.decode(g, skip_special_tokens=True).strip() for g in gen]
        rewards, corrects, fmts = [], [], []
        for t in texts:
            ok = is_correct(t, s.answer, s.answer_kind)
            f = G.format_ok(t)
            r = G.reward(ok, f, prior_conflict=s.is_conflict,
                         w_conflict=conflict_weight)
            rewards.append(r["total"])
            corrects.append(float(ok))
            fmts.append(float(f))
        adv = G.group_advantages(rewards)
        if all(a == 0.0 for a in adv):
            stats["zero_adv_groups"] += 1
            print(f"step {step:3d} 整组同分（r={[round(r,2) for r in rewards]}）跳过")
            continue
        # 2) 反传：-adv * logp(答案)（单次更新，ratio≡1）
        opt.zero_grad(set_to_none=True)
        losses = []
        for g, a in zip(gen, adv):
            if not g:
                continue
            ans = torch.tensor([g], device=device)
            lp = G.answer_logprob(model, px, ids, ans)
            losses.append(-a * lp.mean())
        if not losses:
            continue
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        stats["steps"] += 1
        k = stats["steps"]
        stats["reward_mean"] += (sum(rewards) / len(rewards) - stats["reward_mean"]) / k
        stats["correct_rate"] += (sum(corrects) / len(corrects) - stats["correct_rate"]) / k
        stats["format_rate"] += (sum(fmts) / len(fmts) - stats["format_rate"]) / k
        rec = {"step": step, "rewards": rewards, "adv": [round(a, 3) for a in adv],
               "texts": texts, "loss": float(loss.detach()),
               "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20)
               if device == "cuda" else None}
        hist.append(rec)
        if log_path:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if step % 2 == 0:
            print(f"step {step:3d} r={[round(r,2) for r in rewards]} "
                  f"adv={[round(a,2) for a in adv]} loss={float(loss):.3f}")
    stats["elapsed_s"] = round(time.time() - t0, 1)
    stats["history"] = hist[-5:]
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["curriculum", "counterfactual", "grpo"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--group-size", type=int, default=2)
    ap.add_argument("--conflict-weight", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    cfg = load_config("sft_pilot.yaml")
    out_dir = Path(args.out or f"outputs/ckpt/{args.method}")
    if not out_dir.is_absolute():
        out_dir = OUTPUTS.parent / out_dir
    run_dir = ensure_dir(OUTPUTS / "train" / f"{args.method}_{utc_stamp()}")
    print(f"== S5 {args.method} ==\nsteps={args.steps} limit={args.limit}")

    model, tok = load_sft(cfg, args.device)
    samples = load_samples("train", limit=args.limit)
    lr = args.lr or (1e-5 if args.method == "grpo" else 5e-5)

    if args.method == "grpo":
        stats = run_grpo(model, tok, samples, steps=args.steps,
                         group_size=args.group_size, device=args.device, lr=lr,
                         conflict_weight=args.conflict_weight,
                         log_path=run_dir / "grpo_log.jsonl")
        print(json.dumps({k: v for k, v in stats.items() if k != "history"},
                         ensure_ascii=False))
    else:
        if args.method == "curriculum":
            order = curriculum_order(samples)
        else:
            order = augment_counterfactual(samples)
        loader = make_loader(cfg, tok, split="train", limit=len(order) + 8,
                             batch_size=args.batch_size)
        ds = ChartSFT(order, tok, image_size=384)
        from torch.utils.data import DataLoader
        from stage3_train.dataset import collate

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate(b, tok.pad_token_id))
        params = trainable(model)
        opt = torch.optim.AdamW(params, lr=lr)
        log = run_dir / "log.jsonl"
        t0 = time.time()
        i, step = 0, 0
        while step < args.steps:
            batch = loader.dataset[i % len(ds)]
            b = collate([batch], tok.pad_token_id)
            opt.zero_grad(set_to_none=True)
            loss = sft_step(model, tok, b, args.device)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            i += 1
            rec = {"step": step, "loss": round(loss, 4),
                   "elapsed_s": round(time.time() - t0, 1)}
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            if step % 5 == 0:
                print(f"step {step:3d} loss {loss:.4f} {rec['elapsed_s']}s")
        stats = {"steps": step, "elapsed_s": round(time.time() - t0, 1),
                 "n_examples": len(order)}

    ensure_dir(out_dir)
    torch.save(model.connector.state_dict(), out_dir / "connector.pt")
    model.llm.save_pretrained(out_dir / "lora")
    write_json(out_dir / "meta.json", {"method": args.method, "steps": args.steps,
                                       "lr": lr, "limit": args.limit,
                                       "group_size": args.group_size,
                                       "conflict_weight": args.conflict_weight,
                                       "stats": {k: v for k, v in stats.items()
                                                 if k != "history"}})
    print(f"saved {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
