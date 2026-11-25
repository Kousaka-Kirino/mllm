"""S3-2 监督微调：连接器 + LoRA(r=16) 一起训。

这一阶段才真正让模型"学会回答问题"。8GB 上的配置：
batch=2、grad_accum=8、max_text=128、梯度检查点开、bf16、视觉塔冻结。

    python -m stage3_train.train_sft --dry-run
    python -m stage3_train.train_sft --epochs 2
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


def attach_lora(model: PicoVLM, cfg: ModelConfig) -> int:
    from peft import LoraConfig, get_peft_model

    peft_cfg = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model.llm = get_peft_model(model.llm, peft_cfg)
    return sum(p.numel() for p in model.llm.parameters() if p.requires_grad)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--epochs", type=int, default=TrainConfig().epochs_sft)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = ModelConfig()
    tcfg = TrainConfig()
    seed_everything(1)

    samples = [Sample.from_dict(r) for r in
               io.read_jsonl(DATA / "synth" / "train.jsonl")]
    if args.limit:
        samples = samples[: args.limit]

    model = PicoVLM(cfg, load_weights=not args.dry_run)
    device = "cuda" if torch.cuda.is_available() and not args.dry_run else "cpu"

    if args.dry_run:
        class _CharTok:
            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [ord(c) % 1000 + 10 for c in text]}

        ds = SynthSFT(samples[:8], _CharTok(), cfg.input_size)
        batch = collate([ds[i] for i in range(4)])
        out = model(Batch(**batch))
        print(f"[dry-run] loss={float(out['loss']):.4f} "
              f"trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        return 0

    ckpt = OUTPUTS / "ckpt" / "align" / "connector.pt"
    if ckpt.exists():
        model.connector.load_state_dict(torch.load(ckpt, map_location="cpu")["connector"])
        print(f"载入对齐阶段连接器：{ckpt}")
    else:
        print("[warn] 没有对齐阶段 checkpoint，直接从头 SFT（不推荐）")

    for p in model.vision.parameters():
        p.requires_grad_(False)
    n_lora = attach_lora(model, cfg)
    model.to(device)
    model.llm.gradient_checkpointing_enable()
    print(f"LoRA 可训练参数 {n_lora / 1e6:.2f}M")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.llm)
    ds = SynthSFT(samples, tok, cfg.input_size)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=tcfg.batch_size, shuffle=True, collate_fn=collate, num_workers=2)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg.lr_sft, weight_decay=0.01)
    total = args.steps or len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=tcfg.lr_sft, total_steps=total, pct_start=tcfg.warmup_ratio)

    model.train()
    t0, run_loss, step = time.time(), 0.0, 0
    for epoch in range(args.epochs):
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
                    print(f"ep{epoch} step {step:5d} loss {run_loss / tcfg.log_every:.4f} "
                          f"{(time.time() - t0):.0f}s")
                    run_loss = 0.0
            step += 1
            if args.steps and step >= args.steps:
                break

    saveto = OUTPUTS / "ckpt" / "sft"
    saveto.mkdir(parents=True, exist_ok=True)
    model.llm.save_pretrained(saveto / "lora")
    torch.save({"connector": model.connector.state_dict(), "cfg": cfg.__dict__},
               saveto / "connector.pt")
    print(f"保存 -> {saveto}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
