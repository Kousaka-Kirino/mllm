"""S3-B · LoRA SFT：连接器 + LoRA，其余全部冻结。

和 S3-A 的区别只有一句：**现在开始学任务，而不是学对齐**。
LoRA 只挂在 q/k/v/o 上（r=8，alpha=16），可训参数约 4–5M。

跑：
    python -m stage3_train.sft --dry-run --limit 8 --max-steps 2
    python -m stage3_train.sft --limit 318 --max-steps 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from picovlm.answers import is_correct
from picovlm.config import OUTPUTS, load_config, model_config_from
from picovlm.io_utils import ensure_dir, utc_stamp, write_json
from picovlm.model.vlm import PicoVLM

from .align import build_model, make_loader, to_device
from .dataset import EOS_SUFFIX


def attach_lora(model: PicoVLM, lora_cfg: dict) -> int:
    """把 LoRA 挂到 LLM 上；返回 LoRA 可训参数量。

    dry-run 用的是 `_DummyLLM`（只有 embedding + 一个 Linear head，没有注意力层），
    自然找不到 q_proj/k_proj/... ——这时跳过 LoRA 并明确提示，不要静默降级。
    dry-run 验证的是"数据 → 前向 → 反传 → 存盘"这条管线，LoRA 通路由真权重跑通。
    """
    from peft import LoraConfig, get_peft_model

    targets = list(lora_cfg.get("targets",
                                ["q_proj", "k_proj", "v_proj", "o_proj"]))
    names = {n for n, _ in model.llm.named_modules()}
    if not all(any(n.endswith(t) for n in names) for t in targets):
        print("[warn] 替身 LLM 没有注意力层，dry-run 跳过 LoRA（真权重训练会挂上）")
        return 0

    cfg = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=targets,
        task_type="CAUSAL_LM",
        bias="none",
    )
    model.llm = get_peft_model(model.llm, cfg)
    return sum(p.numel() for p in model.llm.parameters() if p.requires_grad)


def prepare_trainable(model: PicoVLM) -> list[torch.nn.Parameter]:
    for p in model.vision.parameters():
        p.requires_grad_(False)
    for p in model.connector.parameters():
        p.requires_grad_(True)
    model.vision.eval()
    model.connector.train()
    return [p for p in model.parameters() if p.requires_grad]


@torch.no_grad()
def quick_eval(model: PicoVLM, loader: DataLoader, tok, device: str,
               max_batches: int = 8, max_new_tokens: int = 8) -> dict:
    """生成式小评测：用同一套 `answers.is_correct` 判分，避免"训练/评测两套口径"。"""
    if not hasattr(model.llm, "generate"):
        return {"n": 0, "acc": None, "skipped": "替身 LLM 没有 generate"}
    model.eval()
    ok = tot = 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        batch = to_device(b, device)
        gen = model.generate(batch.pixel_values, batch.input_ids,
                             batch.attention_mask, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for gold, text in zip(b["answer"], texts):
            tot += 1
            ok += int(is_correct(text.strip(), gold))
    model.train()
    return {"n": tot, "acc": (ok / tot) if tot else 0.0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="sft_pilot.yaml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--init-connector", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    tr = dict(cfg.get("train", {}))
    for key, val in (("epochs", args.epochs), ("max_steps", args.max_steps),
                     ("lr", args.lr), ("batch_size", args.batch_size),
                     ("grad_accum", args.grad_accum)):
        if val is not None:
            tr[key] = val
    lr = float(tr.get("lr", 2e-4))
    bsz = int(tr.get("batch_size", 2))
    accum = int(tr.get("grad_accum", 8))
    epochs = int(tr.get("epochs", 2))
    max_steps = int(tr.get("max_steps", 0)) or None
    warmup = float(tr.get("warmup_ratio", 0.03))
    clip = float(tr.get("max_grad_norm", 1.0))
    limit = args.limit if args.limit is not None else cfg.get("data", {}).get("limit")
    init_ckpt = args.init_connector or cfg.get("init_connector")
    out_dir = Path(args.out or cfg.get("output", "outputs/ckpt/sft_pilot"))
    if not out_dir.is_absolute():
        out_dir = OUTPUTS.parent / out_dir
    run_dir = ensure_dir(OUTPUTS / "train" / f"sft_{utc_stamp()}")

    device = args.device or ("cpu" if args.dry_run else "cuda")
    if device == "cpu":
        torch.set_num_threads(4)
    print(f"== S3-B LoRA SFT ==\nconfig {args.config}  device={device}  "
          f"bs={bsz} accum={accum} lr={lr:g} epochs={epochs}")

    model, tok = build_model(cfg, load_weights=not args.dry_run, device=device)
    if init_ckpt:
        p = Path(init_ckpt)
        if not p.is_absolute():
            p = OUTPUTS.parent / p
        if p.exists():
            state = torch.load(p, map_location="cpu")
            model.connector.load_state_dict(state)
            print(f"loaded connector  {p}")
        else:
            print(f"[warn] 找不到 {p}，连接器从随机初始化开始")

    n_lora = attach_lora(model, cfg.get("lora", {}))
    params = prepare_trainable(model)
    n_train = sum(p.numel() for p in params)
    print(f"trainable {n_train/1e6:.2f}M "
          f"(lora {n_lora/1e6:.2f}M + connector "
          f"{sum(p.numel() for p in model.connector.parameters())/1e6:.2f}M) "
          f"/ total {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    if cfg.get("grad_checkpoint", True) and hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()
        if hasattr(model.llm, "config"):
            model.llm.config.use_cache = False

    train_loader = make_loader(cfg, tok, split="train", limit=limit, batch_size=bsz)
    val_loader = make_loader(cfg, tok, split="val", limit=64, batch_size=bsz)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps_per_epoch = max(1, len(train_loader) // accum)
    total_steps = max_steps or steps_per_epoch * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=total_steps, pct_start=warmup)

    log_path = run_dir / "log.jsonl"
    step, t0 = 0, time.time()
    running: list[float] = []
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
                if step % 10 == 0 or step == total_steps:
                    rec = {"step": step, "epoch": epoch,
                           "loss": sum(running) / len(running),
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
        metrics = quick_eval(model, val_loader, tok, device,
                             max_batches=4 if args.dry_run else 8)
        print(f"epoch {epoch}  val_loss/eval {metrics}")
        if max_steps and step >= max_steps:
            break

    ensure_dir(out_dir)
    torch.save(model.connector.state_dict(), out_dir / "connector.pt")
    if hasattr(model.llm, "save_pretrained"):
        model.llm.save_pretrained(out_dir / "lora")
    write_json(out_dir / "meta.json", {
        "config": args.config, "lr": lr, "bs": bsz, "accum": accum,
        "epochs": epochs, "max_steps": max_steps, "trainable_params": n_train,
        "lora_params": n_lora, "init_connector": str(init_ckpt),
        "dry_run": args.dry_run,
    })
    print(f"saved {out_dir}  (connector.pt + lora/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
