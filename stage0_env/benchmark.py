"""S0 硬件基准：在真机上量出"能不能跑"，而不是估算。

2.0 的 L0 门槛要求四件事都有实测数字：
  1. CUDA 可用、显卡型号/显存认对；
  2. 底座权重能加载（显存 + 耗时）；
  3. 单图前向不 NaN、不 OOM；
  4. 记录 idle 显存，供后面判断"桌面程序抢了多少"。

用法（真机，会下载约 1.2GB 权重）：
    python -m stage0_env.benchmark --suite load,infer --repeats 3

用法（无网/CI，只验证测量框架本身）：
    python -m stage0_env.benchmark --suite load,infer --repeats 1 --dry-run

产物：outputs/bench/bench_<时间戳>.json + .csv
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from picovlm import io_utils as io
from picovlm.config import OUTPUTS, load_config

MIB = 2 ** 20


@dataclass
class Row:
    suite: str
    repeat: int
    latency_ms: float
    peak_alloc_mib: float
    peak_reserved_mib: float
    ram_mib: float
    note: str = ""
    extra: dict = field(default_factory=dict)


def _ram_mib() -> float:
    try:
        import psutil  # 可选依赖
        return psutil.Process().memory_info().rss / MIB
    except Exception:
        return float("nan")


def _cuda_stats() -> tuple[float, float]:
    import torch
    if not torch.cuda.is_available():
        return float("nan"), float("nan")
    return (torch.cuda.max_memory_allocated() / MIB,
            torch.cuda.max_memory_reserved() / MIB)


def _idle_mib() -> float:
    import torch
    if not torch.cuda.is_available():
        return float("nan")
    free, total = torch.cuda.mem_get_info()
    return free / MIB


def _reset_peak() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def bench_load(cfg: dict, repeats: int, dry: bool) -> list[Row]:
    """加载视觉塔 + 语言模型的耗时与显存。这是 L0 的第一道门。"""
    rows: list[Row] = []
    for r in range(repeats):
        _reset_peak()
        t0 = time.perf_counter()
        if dry:
            time.sleep(0.01)
            note = "dry-run（未加载真实权重）"
            vram = (0.0, 0.0)
        else:
            import torch
            from transformers import AutoModelForCausalLM, SiglipVisionModel
            dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float16
            vis = SiglipVisionModel.from_pretrained(cfg["vision"]["model"], torch_dtype=dtype)
            llm = AutoModelForCausalLM.from_pretrained(cfg["language"]["model"],
                                                       torch_dtype=dtype)
            vis.to("cuda").eval()
            llm.to("cuda").eval()
            note = f"vision={cfg['vision']['model'].split('/')[-1]} llm={cfg['language']['model'].split('/')[-1]}"
            vram = _cuda_stats()
            del vis, llm
            torch.cuda.empty_cache()
        dt = (time.perf_counter() - t0) * 1000
        rows.append(Row("load", r, dt, vram[0], vram[1], _ram_mib(), note))
    return rows


def bench_infer(cfg: dict, repeats: int, dry: bool) -> list[Row]:
    """单图 + 短问题的一次前向：测延迟与峰值显存。"""
    rows: list[Row] = []
    if dry:
        import torch
        for r in range(repeats):
            _reset_peak()
            t0 = time.perf_counter()
            x = torch.randn(576, 768)
            w = torch.randn(768, 896)
            y = x @ w
            float(y.sum())
            dt = (time.perf_counter() - t0) * 1000
            rows.append(Row("infer", r, dt, 0.0, 0.0, _ram_mib(),
                            "dry-run（假张量，只验证计时/统计链路）"))
        return rows

    import torch
    from transformers import AutoModelForCausalLM, SiglipVisionModel

    dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float16
    vis = SiglipVisionModel.from_pretrained(cfg["vision"]["model"], torch_dtype=dtype).to("cuda").eval()
    llm = AutoModelForCausalLM.from_pretrained(cfg["language"]["model"],
                                               torch_dtype=dtype).to("cuda").eval()
    size = int(cfg["vision"].get("input_size", 384))
    # 坑一：`AutoModel` 加载 google/siglip-base-patch16-384 得到的是**双塔** SiglipModel，
    # forward 要 input_ids（文本侧），只传 pixel_values 会报
    # "You have to specify input_ids"。所以用 SiglipVisionModel（单塔）。
    # 坑二：SiglipVisionConfig 上才有 hidden_size。
    d_vis = vis.config.hidden_size
    d_llm = llm.config.hidden_size
    proj = torch.nn.Linear(int(d_vis), int(d_llm), dtype=dtype, device="cuda")
    pixel = (torch.rand(1, 3, size, size, device="cuda", dtype=dtype) - 0.5) / 0.5
    text = torch.randint(0, 1000, (1, 12), device="cuda")
    for r in range(repeats):
        _reset_peak()
        t0 = time.perf_counter()
        with torch.no_grad():
            v = vis(pixel_values=pixel).last_hidden_state
            v = proj(v)
            emb = llm.get_input_embeddings()(text)
            out = llm(inputs_embeds=torch.cat([v, emb], dim=1))
            _ = float(out.logits[:, -1].float().sum())
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        alloc, resv = _cuda_stats()
        rows.append(Row("infer", r, dt, alloc, resv, _ram_mib(),
                        f"visual_tokens={v.shape[1]} text_tokens={text.shape[1]}"))
    return rows


def write_csv(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "repeat", "latency_ms", "peak_alloc_mib",
                    "peak_reserved_mib", "ram_mib", "note"])
        for r in rows:
            w.writerow([r.suite, r.repeat, f"{r.latency_ms:.1f}",
                        f"{r.peak_alloc_mib:.0f}", f"{r.peak_reserved_mib:.0f}",
                        f"{r.ram_mib:.0f}", r.note])


def summarize(rows: list[Row]) -> str:
    """每个 suite 取中位数，太慢的抖动会被中位数吸收。"""
    import statistics as st
    lines = ["| suite | n | 延迟中位数 ms | 峰值 allocated MiB | 峰值 reserved MiB |",
             "|---|---:|---:|---:|---:|"]
    for suite in ("load", "infer"):
        rs = [r for r in rows if r.suite == suite]
        if not rs:
            continue
        lat = st.median([r.latency_ms for r in rs])
        alloc = st.median([r.peak_alloc_mib for r in rs])
        resv = st.median([r.peak_reserved_mib for r in rs])
        lines.append(f"| {suite} | {len(rs)} | {lat:.1f} | {alloc:.0f} | {resv:.0f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stage0_env.benchmark")
    ap.add_argument("--config", default="model.yaml")
    ap.add_argument("--suite", default="load,infer")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    suites = [s.strip() for s in args.suite.split(",") if s.strip()]
    rows: list[Row] = []

    idle = _idle_mib()
    print(f"[bench] idle 可用显存 {idle:.0f} MiB；suite={suites} repeats={args.repeats} dry={args.dry_run}")
    if "load" in suites:
        rows += bench_load(cfg, args.repeats, args.dry_run)
    if "infer" in suites:
        rows += bench_infer(cfg, args.repeats, args.dry_run)

    stamp = io.utc_stamp()
    out = OUTPUTS / "bench" / f"bench_{stamp}.json"
    io.write_json(out, {"stamp": stamp, "idle_free_mib": idle,
                        "dry_run": args.dry_run, "rows": [asdict(r) for r in rows]})
    write_csv(rows, out.with_suffix(".csv"))
    print(summarize(rows))
    print(f"\n[bench] 写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
