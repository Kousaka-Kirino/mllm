"""S6 · 预算测量：同样的显存，换多少 token / 多大 batch / 多少精度。

三个可直接测量的量（不依赖训练好坏）：

1. **视觉 token 数**：mlp 576 / pixelshuffle 144 / resampler 64；
2. **延迟**：单次前向（含视觉塔 + 连接器 + LLM）；
3. **峰值显存**：单样本前向的 allocated 峰值。

诚实边界：换连接器需要重新训练才能谈精度。所以 S6 的报告里，
**只有 mlp 那一档有精度数字**（来自 S4），其余两档标 `accuracy: null（未训练）`，
只比成本，不比效果。这是"只测能测的"的写法。

    python -m stage6_ablation.measure --repeats 3
    python -m stage6_ablation.measure --batch-ladder
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from picovlm.budget import CONNECTORS, count_tokens, patches
from picovlm.config import OUTPUTS, load_config, model_config_from
from picovlm.io_utils import ensure_dir, utc_stamp, write_json
from picovlm.model.vlm import PicoVLM


def stat(vals: list[float]) -> dict:
    vals = sorted(vals)
    n = len(vals)
    return {"median": vals[n // 2], "min": vals[0], "max": vals[-1]}


@torch.no_grad()
def bench_one(model: PicoVLM, batch: int = 1, seq: int = 40,
              input_size: int = 384, repeats: int = 3, device: str = "cuda") -> dict:
    px = torch.randn(batch, 3, input_size, input_size, device=device)
    ids = torch.randint(0, 100, (batch, seq), device=device)
    mask = torch.ones_like(ids)
    lat, peaks = [], []
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model({"pixel_values": px, "input_ids": ids, "attention_mask": mask,
               "labels": None})
        lat.append((time.time() - t0) * 1000)
        peaks.append(torch.cuda.max_memory_allocated() / 2**20)
    return {"batch": batch, "seq": seq, "latency_ms": stat(lat),
            "peak_mib": stat(peaks)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--batch-ladder", action="store_true",
                    help="只跑 mlp，扫 batch 1/2/4")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    cfg = load_config("model.yaml")
    mc = model_config_from(cfg)
    if args.batch_ladder:
        model = PicoVLM(mc, load_weights=True).to(args.device).to(torch.bfloat16).eval()
        rows = [bench_one(model, batch=b, repeats=args.repeats, device=args.device)
                for b in (1, 2, 4)]
        print(f"{'batch':>6} {'latency_ms':>12} {'peak_mib':>10}")
        for r in rows:
            print(f"{r['batch']:>6} {r['latency_ms']['median']:>12.1f} "
                  f"{r['peak_mib']['median']:>10.0f}")
        out = OUTPUTS / "bench" / f"ladder_{utc_stamp()}.json"
        write_json(out, rows)
        print(f"saved {out}")
        return 0

    n_patch = patches(mc.input_size, mc.patch)
    rows = []
    for name in CONNECTORS:
        mc2 = type(mc)(**{**mc.__dict__, "connector": name})
        model = PicoVLM(mc2, load_weights=True).to(args.device).to(torch.bfloat16).eval()
        r = bench_one(model, repeats=args.repeats, device=args.device)
        rows.append({"connector": name,
                     "tokens": count_tokens(name, n_patch, mc.n_queries,
                                            mc.shuffle_factor),
                     "params_m": sum(p.numel() for p in model.connector.parameters()) / 1e6,
                     "latency_ms": r["latency_ms"], "peak_mib": r["peak_mib"],
                     "accuracy": None})
        print(f"{name:<12} tokens={rows[-1]['tokens']:>4} "
              f"connector={rows[-1]['params_m']:.2f}M "
              f"lat={r['latency_ms']['median']:.0f}ms "
              f"peak={r['peak_mib']['median']:.0f}MiB")
        del model
        torch.cuda.empty_cache()

    out = ensure_dir(OUTPUTS / "bench") / f"connectors_{utc_stamp()}.json"
    write_json(out, rows)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
