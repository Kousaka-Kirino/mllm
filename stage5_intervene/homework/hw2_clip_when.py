"""HW5-2 · clip 什么时候生效。

目标：扫 δ = logp_new − logp_old，找出 clip 开始改变梯度的位置，
并用它解释"单次更新 + 小 lr 时 clip 几乎不生效"。

跑：python -m stage5_intervene.homework.hw2_clip_when
"""

from __future__ import annotations

import torch

from stage5_intervene.grpo import grpo_loss


def effective_delta(adv: float = 1.0, clip: float = 0.2,
                    hi: float = 3.0, step: float = 0.01) -> float:
    """返回 clip 首次生效的 logp_new（相对 logp_old=0）。

    TODO:
      ratio = exp(logp_new)；adv > 0 时，clip 生效当且仅当 ratio > 1+clip，
      即 logp_new > ln(1+clip)。扫一遍验证这个解析值。
    """
    raise NotImplementedError("HW5-2")


def main() -> int:
    import math

    d = effective_delta()
    analytic = math.log(1.0 + 0.2)
    print(f"扫描到 {d:.3f}，解析值 {analytic:.3f}")
    print("hw5-2 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
