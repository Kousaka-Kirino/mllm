"""作业 1：给 PicoVLM 算一次显存账，再和实测对照。

要填的只有一个函数：`estimate_peak_mib`。
思路：峰值 ≈ 权重 + 激活 + logits 临时区。三个部分都是"参数量 × 字节数"的量级估计，
不需要精确到 MB，能判断"会不会 OOM"就够了。

跑法：
    python stage0_env/homework/hw1_vram_budget.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MIB = 2 ** 20

# ---- 已知量（来自 configs/model.yaml，不要在作业里改）----
SIGLIP_PARAMS = 93_000_000        # SigLIP-base-patch16-384
QWEN_PARAMS = 494_000_000         # Qwen2.5-0.5B-Instruct
LORA_PARAMS = 4_000_000           # r=8，只挂在 q/k/v/o 上
BYTES_BF16 = 2
VISUAL_TOKENS = 576               # 24×24 patch
TEXT_TOKENS = 32


def weights_mib() -> float:
    """所有常驻权重：视觉塔 + LLM + LoRA + 连接器（约 12M）。"""
    return (SIGLIP_PARAMS + QWEN_PARAMS + LORA_PARAMS + 12_000_000) * BYTES_BF16 / MIB


def estimate_peak_mib(batch: int = 2, grad_checkpoint: bool = True) -> float:
    """估算训练一步的峰值显存（MiB）。

    提示（自己推，不要照抄）：
      - 权重部分已经由 weights_mib() 给了；
      - 激活 ≈ batch × (视觉 token + 文本 token) × hidden × 层数 × 字节 × 系数，
        开了梯度检查点后，系数从"每层都存"降到"只存层输入"；
      - logits 临时区 ≈ batch × seq × vocab × 字节，这一步最容易被忽略。

    TODO: 写出你的公式并 return。写完把下面这行删掉。
    """
    raise NotImplementedError("补完 estimate_peak_mib，再跑这个脚本")


def measured_infer_mib() -> float | None:
    """从最近一次真机 benchmark 里取 infer 的峰值（没有就返回 None）。"""
    files = sorted(glob.glob(str(ROOT / "outputs/bench/bench_*.json")))
    for f in reversed(files):
        data = json.loads(Path(f).read_text(encoding="utf-8"))
        rows = [r for r in data.get("rows", []) if r["suite"] == "infer"]
        if rows and rows[0]["peak_alloc_mib"] > 0:
            return sum(r["peak_alloc_mib"] for r in rows) / len(rows)
    return None


def main() -> int:
    print(f"权重常驻：{weights_mib():.0f} MiB")
    for batch in (1, 2):
        est = estimate_peak_mib(batch)
        print(f"batch={batch} 估算峰值：{est:.0f} MiB")
    real = measured_infer_mib()
    if real is None:
        print("没找到真机 benchmark（先跑 python -m stage0_env.benchmark）")
        return 1
    est = estimate_peak_mib(1)
    diff = abs(est - real) / real * 100
    print(f"实测单图前向峰值：{real:.0f} MiB；估算差 {diff:.0f}%")
    print("结论：TODO 用一句话写清楚差在哪（这是作业的一部分）")
    return 0 if diff < 60 else 2


if __name__ == "__main__":
    raise SystemExit(main())
