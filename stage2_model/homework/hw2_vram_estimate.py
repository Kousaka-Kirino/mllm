"""HW2 · 显存估算 vs 实测。

目标：在真正开始训练之前，能用一张纸算出"这个配置会不会 OOM"，并知道自己
的公式比 reality 少了什么。

跑：python -m stage2_model.homework.hw2_vram_estimate
"""

from __future__ import annotations

MIB = 2 ** 20


def estimate_peak_mib(params_m: float, batch: int, seq_len: int, hidden: int,
                     bytes_per_param: int = 2, checkpoint: bool = False) -> float:
    """返回"权重 + 激活"的粗略峰值（MiB）。

    TODO: 写出你自己的公式，并在注释里解释每一项。
      权重      = params_m * 1e6 * bytes_per_param / MIB
      激活      ≈ batch * seq_len * hidden * bytes * k   （k 由你论证）
      注意力中间量 ≈ batch * heads * seq_len**2 * bytes
      checkpoint=True 时激活量应显著下降（只保留每层输入）
    """
    raise NotImplementedError("HW2: 实现 estimate_peak_mib")


def compare_with_measured(estimate_mib: float, measured_mib: float) -> float:
    """返回相对误差 (estimate - measured) / measured。"""
    raise NotImplementedError("HW2: 实现 compare_with_measured")


def main() -> int:
    measured = 2189.0  # smoke --load 实测：alloc MiB，576 视觉 token，bs=1
    est = estimate_peak_mib(589.5, batch=1, seq_len=591, hidden=896)
    err = compare_with_measured(est, measured)
    print(f"estimate {est:.0f} MiB  measured {measured:.0f} MiB  err {err:+.1%}")
    if abs(err) > 0.4:
        print("提示：检查是否漏算 KV cache / cuBLAS workspace / 临时 buffer / 碎片")
    print("hw2 done（把漏算项写进作业报告）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
