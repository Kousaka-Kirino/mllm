"""S3-1 对齐训练：冻 ViT、冻 LLM，只训连接器。

目的：让随机初始化的连接器先学会"把 patch 特征翻译成 LLM 认识的词"。
500 万到 2000 万可训练参数、5 万对、单卡 4060 约 2 小时（fp16 + 梯度检查点）。

    python -m stage3_train.train_align --dry-run        # 不下载权重，只验证形状
    python -m stage3_train.train_align --limit 50000
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from picovlm import io_utils as io
from picovlm.config import DATA, ModelConfig, OUTPUTS, TrainConfig
from picovlm.model import Batch, PicoVLM
from picovlm.schema import Sample
from picovlm.seeds import seed_everything

from .data import SynthSFT, collate


def build(dry_run: bool, cfg: ModelConfig) -> PicoVLM:
    model = PicoVLM(cfg, load_weights=not dry_run)
    if dry_run:
        return model
    for p in model.vision.parameters():
        p.requires_grad_(False)
    for p in model.llm.parameters():
        p.requires_grad_(False)
    model.llm.gradient_checkpointing_enable()
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--steps", type=int, default=0, help="0 = 跑完一个 epoch")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lr", type=float, default=TrainConfig().lr_align)
    args = ap.parse_args(argv)

    cfg = ModelConfig()
    tcfg = TrainConfig()
    seed_everything(0)

    path = DATA / "synth" / "train.jsonl"
    samples = [Sample.from_dict(r) for r in io.read_jsonl(path)]
    if args.limit:
        samples = samples[: args.limit]

    model = build(args.dry_run, cfg)
    device = "cuda" if torch.cuda.is_available() and not args.dry_run else "cpu"
    model.to(device)

    if args.dry_run:
        # 用字符级假 tokenizer，避开 1.5GB 权重下载
        class _CharTok:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [ord(c) % 1000 + 10 for c in text]}

        ds = SynthSFT(samples[:8], _CharTok(), cfg.input_size)
        batch = collate([ds[i] for i in range(4)])
        out = model(Batch(**{k: v.to(device) for k, v in batch.items()}))
        print(f"[dry-run] batch={tuple(batch['pixel_values'].shape)} "
              f"seq={tuple(batch['input_ids'].shape)} loss={float(out['loss']):.4f}")
        return 0

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.llm)
    ds = SynthSFT(samples, tok, cfg.input_size)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate, num_workers=2)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total = args.steps or len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total, pct_start=tcfg.warmup_ratio)

    model.train()
    model.connector.train()
    t0, run_loss, step = time.time(), 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(Batch(**batch))
        (out["loss"] / tcfg.grad_accum).backward()
        run_loss += float(out["loss"])
        if (step + 1) % tcfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(params, tcfg.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if step % tcfg.log_every == 0:
                print(f"step {step:5d} loss {run_loss / tcfg.log_every:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0):.0f}s")
                run_loss = 0.0
        step += 1
        if args.steps and step >= args.steps:
            break

    saveto = OUTPUTS / "ckpt" / "align"
    saveto.mkdir(parents=True, exist_ok=True)
    torch.save({"connector": model.connector.state_dict(),
                "cfg": cfg.__dict__}, saveto / "connector.pt")
    print(f"保存 -> {saveto / 'connector.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
