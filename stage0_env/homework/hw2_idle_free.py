"""作业 2：判断这台机器的真实训练预算。

idle free 不是一个常数：浏览器、聊天软件、IDE 都在抢显存，而且 WDDM 会共享。
所以要在"准备训练的那一刻"采样，而不是早上打开电脑时测一次。

跑法：
    python stage0_env/homework/hw2_idle_free.py
"""

from __future__ import annotations

import statistics
import time


def sample_free_mib(n: int = 5, interval_s: float = 2.0) -> list[float]:
    """采样 n 次可用显存（MiB）。这是已经实现好的部分，不用改。"""
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("没有 CUDA 设备")
    out = []
    for i in range(n):
        free, total = torch.cuda.mem_get_info()
        out.append(free / 2 ** 20)
        print(f"  第 {i + 1} 次：free {free / 2 ** 20:.0f} MiB / total {total / 2 ** 20:.0f} MiB")
        if i < n - 1:
            time.sleep(interval_s)
    return out


def verdict(samples: list[float]) -> str:
    """根据采样给出训练预算判断。

    TODO: 实现三档判断，返回一句话结论：
      - 中位数 ≥ 7000 MiB：``预算充足``，可以 batch=2
      - 5000–7000 MiB：``预算偏紧``，建议 batch=1 + 梯度累积
      - < 5000 MiB：``先关程序``，不要开训练
    同时把中位数、极差（max-min）写进结论里。
    """
    raise NotImplementedError("补完 verdict，再跑这个脚本")


def main() -> int:
    print("采样中（每 2 秒一次，共 5 次）：")
    samples = sample_free_mib()
    print(f"\n{verdict(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
