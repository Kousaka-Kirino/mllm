"""S2 · 冒烟测试：把"模型能跑通"变成可复现的三行输出。

跑法：
    python -m stage2_model.smoke              # 骨架（不下载权重，CPU/GPU 都行）
    python -m stage2_model.smoke --load       # 真权重（首次约 1.5GB 下载）
    python -m stage2_model.smoke --load --image data/synth/images/xxxx.png --ask "..."

它回答四个问题：
1. 显存够不够（峰值 allocated / reserved）；
2. 视觉 token 有多少（决定显存与注意力成本）；
3. 前向一次多久（决定单 epoch 时间）；
4. loss 是不是个正常数字（不是 nan / 不是常数）。
"""

from __future__ import annotations

import argparse
import statistics as st
import time
from pathlib import Path

from picovlm.config import load_config, model_config_from
from picovlm.model.vlm import Batch, PicoVLM

import torch


def _count_params(m: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def _dummy_batch(cfg, batch_size: int = 1, text_len: int = 24) -> Batch:
    g = torch.Generator().manual_seed(17)
    return Batch(
        pixel_values=torch.randn(batch_size, 3, cfg.input_size, cfg.input_size,
                                 generator=g),
        input_ids=torch.randint(0, 1000, (batch_size, text_len), generator=g),
        attention_mask=torch.ones(batch_size, text_len, dtype=torch.long),
        labels=torch.randint(0, 1000, (batch_size, text_len), generator=g),
    )


def _real_batch(cfg, image: str | None, ask: str | None, load_weights: bool) -> Batch:
    """用真 tokenizer 造一个"问题 + 答案"样本；没有图就随机像素。"""
    from PIL import Image

    from transformers import AutoTokenizer

    from picovlm.model.vision import FrozenVisionTower

    tok = AutoTokenizer.from_pretrained(cfg.llm)
    q = ask or "What is the value for category A?"
    ans = " 42"
    prompt = f"Question: {q}\nAnswer:"
    p_ids = tok(prompt, return_tensors="pt").input_ids
    a_ids = tok(ans, add_special_tokens=False, return_tensors="pt").input_ids
    input_ids = torch.cat([p_ids, a_ids], dim=1)
    labels = input_ids.clone()
    labels[:, : p_ids.size(1)] = -100

    if image:
        im = Image.open(image).convert("RGB")
        tower = FrozenVisionTower(cfg.vision, cfg.input_size)
        px = tower.preprocess(im).unsqueeze(0)
    else:
        px = torch.zeros(1, 3, cfg.input_size, cfg.input_size)
    return Batch(pixel_values=px, input_ids=input_ids,
                 attention_mask=torch.ones_like(input_ids), labels=labels)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stage2_model.smoke")
    ap.add_argument("--config", default="model.yaml")
    ap.add_argument("--connector", default=None,
                    help="覆盖 connector：mlp / resampler / pixelshuffle")
    ap.add_argument("--load", action="store_true", help="加载真权重（下载 SigLIP+Qwen）")
    ap.add_argument("--image", default=None)
    ap.add_argument("--ask", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args(argv)

    raw = load_config(args.config)
    if args.connector:
        raw.setdefault("connector", {})["kind"] = args.connector
    cfg = model_config_from(raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    model = PicoVLM(cfg, load_weights=args.load)
    build_ms = (time.perf_counter() - t0) * 1000
    if args.load:
        model = model.to(device).eval().to(torch.bfloat16)
        dtype = torch.bfloat16
    else:
        model = model.to(device).eval()
        dtype = torch.float32

    total, trainable = _count_params(model)
    conn_params = sum(p.numel() for p in model.connector.parameters())

    batch = _real_batch(cfg, args.image, args.ask, args.load) if args.load \
        else _dummy_batch(cfg)
    batch = Batch(pixel_values=batch.pixel_values.to(device),
                  input_ids=batch.input_ids.to(device),
                  attention_mask=batch.attention_mask.to(device),
                  labels=batch.labels.to(device) if batch.labels is not None else None)

    lat, losses = [], []
    for _ in range(args.repeats):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            out = model(batch)
        if device == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
        losses.append(float(out["loss"]))

    peak = {}
    if device == "cuda":
        peak = {"alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
                "reserved_mib": torch.cuda.max_memory_reserved() / 2**20}

    print("\n== S2 冒烟 ==")
    print(f"config        connector={cfg.connector}  input={cfg.input_size}  "
          f"patch={cfg.patch}  freeze_vision={cfg.freeze_vision}")
    print(f"build         {build_ms:.0f} ms  device={device}  weights={'real' if args.load else 'dummy'}")
    print(f"params        total={total/1e6:.1f}M  trainable={trainable/1e6:.1f}M  "
          f"connector={conn_params/1e6:.2f}M")
    if not args.load:
        print("              ↳ dummy 模式：total/trainable 里的大头是替身 LLM 的词表，"
              "不是真模型参数量；真实可训参数以 --load 为准")
    print(f"visual tokens {model.n_visual_tokens}")
    print(f"seq len       {batch.input_ids.size(1)} text + {model.n_visual_tokens} visual")
    print(f"loss          {st.median(losses):.4f}  ({', '.join(f'{l:.3f}' for l in losses)})")
    print(f"latency       {st.median(lat):.1f} ms/次（{args.repeats} 次中位数）")
    if peak:
        print(f"peak VRAM     alloc {peak['alloc_mib']:.0f} MiB / "
              f"reserved {peak['reserved_mib']:.0f} MiB")
    else:
        print("peak VRAM     n/a（CPU）")

    # 明显不正常的数字要显式报错，而不是让人误读
    if any(l != l for l in losses):
        raise SystemExit("[smoke] loss 是 nan：检查归一化与 dtype")
    if max(losses) - min(losses) > 1.0:
        print("[smoke] 警告：多次 loss 差很大，可能有随机性没固定（dropout 未 eval？）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
