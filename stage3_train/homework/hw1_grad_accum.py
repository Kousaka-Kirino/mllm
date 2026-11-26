"""HW3-1 · 梯度累积等价性。

目标：亲手验证 `bs=1, accum=4` 与 `bs=4, accum=1` 在数学上等价
（除了 BN/dropout 这类随机层，所以先用 eval 模式关掉 dropout）。

跑：python -m stage3_train.homework.hw1_grad_accum
"""

from __future__ import annotations

import torch
from torch import nn


class Tiny(nn.Module):
    """和一个最小 LLM 同构：embedding → 线性 → 输出。"""

    def __init__(self, vocab=32, d=16) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.proj = nn.Linear(d, d)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids, labels=None):
        h = self.proj(self.emb(ids))
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1), ignore_index=-100)
        return logits, loss


def grad_accum_equivalent(bs=1, accum=4, steps=1, tol=1e-3) -> tuple[bool, float]:
    """返回 (是否等价, 相对误差)。

    TODO:
      1. 造一批 ids/labels（长度 = bs*accum 条，用 torch.manual_seed 固定）；
      2. 方式 A：逐条前向，loss/accum 后反传累积；
      3. 方式 B：整批一次前向，loss 直接反传；
      4. 比较两者所有参数梯度的相对误差（L2 距离 / 范数），< tol 视为等价。
    """
    raise NotImplementedError("HW3-1: 实现 grad_accum_equivalent")


def main() -> int:
    ok, err = grad_accum_equivalent()
    print(f"grad_accum equivalent={ok}  rel_err={err:.2e}")
    print("hw3-1 done" if ok else "hw3-1 FAIL: 两次梯度差异过大")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
