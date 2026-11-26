"""S3-A · 对齐训练：冻结视觉塔与 LLM，只训连接器。

为什么第一步只训连接器：视觉特征（768 维）和 LLM 词嵌入（896 维）不在同一个
空间里。直接上 LoRA 会让"对齐"和"学任务"两件事同时发生，而 8GB 显存和几百条
样本都不足以同时学好。先让连接器把视觉 token 映射到 LLM 能用的空间，再上 LoRA
学任务——这也是 LLaVA 两阶段训练的理由（这里做了缩放版）。

跑：
    python -m stage3_train.align --dry-run          # CPU + 替身模型，验证管线
    python -m stage3_train.align --limit 128        # GPU 真训（约几分钟）
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from picovlm.config import OUTPUTS, load_config, model_config_from
from picovlm.io_utils import ensure_dir, utc_stamp, write_json
from picovlm.model.vlm import Batch, PicoVLM

from .dataset import ChartSFT, collate, load_samples


def build_model(cfg: dict, *, load_weights: bool, device: str) -> tuple[PicoVLM, any]:
    from transformers import AutoTokenizer

    mc = model_config_from(cfg)
    tok = AutoTokenizer.from_pretrained(mc.llm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = PicoVLM(mc, load_weights=load_weights)
    if load_weights:
        model = model.to(device).to(torch.bfloat16)
    else:
        model = model.to(device)
    return model, tok


def freeze_for_align(model: PicoVLM) -> list[torch.nn.Parameter]:
    """视觉塔 + LLM 全部冻结，只有连接器可训。返回可训参数列表。"""
    for p in model.vision.parameters():
        p.requires_grad_(False)
    for p in model.llm.parameters():
        p.requires_grad_(False)
    for p in model.connector.parameters():
        p.requires_grad_(True)
    model.vision.eval()
    model.llm.eval()
    model.connector.train()
    return [p for p in model.parameters() if p.requires_grad]


def make_loader(cfg: dict, tok, *, split: str, limit: int | None,
                batch_size: int) -> DataLoader:
    samples = load_samples(split, limit=limit)
    ds = ChartSFT(samples, tok, image_size=cfg["vision"]["input_size"])
    return DataLoader(ds, batch_size=batch_size, shuffle=split == "train",
                      collate_fn=lambda b: collate(b, tok.pad_token_id))


def to_device(b: dict, device: str) -> Batch:
    return Batch(
        pixel_values=torch.from_numpy(b["pixel_values"]).to(device),
        input_ids=torch.from_numpy(b["input_ids"]).to(device),
        attention_mask=torch.from_numpy(b["attention_mask"]).to(device),
        labels=torch.from_numpy(b["labels"]).to(device),
    )


@torch.no_grad()
def eval_loss(model: PicoVLM, loader: DataLoader, device: str) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        out = model(to_device(b, device))
        tot += float(out["loss"])
        n += 1
    model.train()
    return tot / max(n, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="align_pilot.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU + 替身模型，只验证数据/前向/反传/存盘")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tr = dict(cfg.get("train", {}))
    for key, val in (("epochs", args.epochs), ("max_steps", args.max_steps),
                     ("lr", args.lr), ("batch_size", args.batch_size),
                     ("grad_accum", args.grad_accum)):
        if val is not None:
            tr[key] = val
    lr = float(tr.get("lr", 1e-3))
    bsz = int(tr.get("batch_size", 2))
    accum = int(tr.get("grad_accum", 8))
    epochs = int(tr.get("epochs", 1))
    max_steps = int(tr.get("max_steps", 0)) or None
    warmup = float(tr.get("warmup_ratio", 0.03))
    clip = float(tr.get("max_grad_norm", 1.0))
    limit = args.limit if args.limit is not None else cfg.get("data", {}).get("limit")
    out_dir = Path(args.out or cfg.get("output", "outputs/ckpt/align"))
    if not out_dir.is_absolute():
        out_dir = OUTPUTS.parent / out_dir
    run_dir = ensure_dir(OUTPUTS / "train" / f"align_{utc_stamp()}")

    device = args.device or ("cpu" if args.dry_run else "cuda")
    if device == "cpu":
        torch.set_num_threads(4)
    print(f"== S3-A 对齐 ==\nconfig {args.config}  device={device}  "
          f"bs={bsz} accum={accum} lr={lr:g} limit={limit}")

    model, tok = build_model(cfg, load_weights=not args.dry_run, device=device)
    params = freeze_for_align(model)
    n_train = sum(p.numel() for p in params)
    print(f"trainable {n_train/1e6:.2f}M / total "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    train_loader = make_loader(cfg, tok, split="train", limit=limit,
                               batch_size=bsz)
    val_loader = make_loader(cfg, tok, split="val", limit=32, batch_size=bsz)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = max_steps or steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=warmup)

    log_path = run_dir / "log.jsonl"
    step, t0 = 0, time.time()
    running = []
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            out = model(to_device(batch, device))
            loss = out["loss"] / accum
            if not torch.isfinite(loss):
                print(f"!! loss 非有限 ({loss.item()})，停止")
                return 2
            loss.backward()
            running.append(float(out["loss"].detach()))
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, clip)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 5 == 0 or step == total_steps:
                    rec = {"step": step, "epoch": epoch, "loss": sum(running) / len(running),
                           "lr": sched.get_last_lr()[0],
                           "elapsed_s": round(time.time() - t0, 1),
                           "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None}
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"step {step:4d}  loss {rec['loss']:.4f}  "
                          f"lr {rec['lr']:.2e}  {rec['elapsed_s']}s  "
                          f"peak {rec['peak_mib']}")
                    running = []
                if max_steps and step >= max_steps:
                    break
        val = eval_loss(model, val_loader, device)
        print(f"epoch {epoch}  val_loss {val:.4f}")
        if max_steps and step >= max_steps:
            break

    ensure_dir(out_dir)
    torch.save(model.connector.state_dict(), out_dir / "connector.pt")
    write_json(out_dir / "meta.json", {
        "config": args.config, "lr": lr, "bs": bsz, "accum": accum,
        "epochs": epochs, "max_steps": max_steps, "trainable_params": n_train,
        "dry_run": args.dry_run, "connector": cfg["connector"]["kind"],
    })
    print(f"saved {out_dir/'connector.pt'}\nlog   {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
