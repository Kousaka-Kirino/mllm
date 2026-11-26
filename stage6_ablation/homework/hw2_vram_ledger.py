"""HW2 · 显存账：把 2.6GB 拆成可解释的四块。

S2 实测（batch=1，bf16）：allocated 2189 MiB / reserved 2608 MiB。
你的任务是把这两个数字"算"出来，而不是背下来。

必做：`ledger(batch, seq_text=15, seq_visual=576) -> dict[str, float]`（单位 MiB）
返回 weights / logits / activations / fragmentation 四项，且
`abs(sum - 2608) / 2608 < 0.25`。

提示：
- 权重：93 + 494 + 2.3 = 589.5M 参数 × 2 字节
- logits：batch × (seq_text+seq_visual) × vocab(151936) × 4 字节（fp32！）
- 激活：主要来自 24 层 × 两个序列位置，粗估即可，公式写进 docstring
- fragmentation：reserved − allocated 的那一块，别把它塞进激活里

跑 `python -m stage6_ablation.homework.hw2_vram_ledger` 会跑内置自测。
"""

from __future__ import annotations

VOCAB = 151936
N_LAYERS = 24
PARAMS_M = 93.0 + 494.0 + 2.3


def ledger(batch: int, seq_text: int = 15, seq_visual: int = 576) -> dict[str, float]:
    """返回 {'weights','logits','activations','fragmentation'}，单位 MiB。

    要求：
    - `weights` 与 batch 无关
    - `logits` 与 batch 成正比
    - `activations` 大体与 batch 成正比（有常数项也没关系）
    - `fragmentation` 近似常数（缓存块 + workspace），不随 batch 线性增长
    并在 docstring 里写清每个公式。
    """
    raise NotImplementedError("HW2")


def _selftest() -> None:
    a = ledger(1)
    b = ledger(2)
    assert set(a) == {"weights", "logits", "activations", "fragmentation"}
    assert abs(a["weights"] - b["weights"]) < 1e-6, "权重不应随 batch 变"
    assert abs(b["logits"] / a["logits"] - 2.0) < 0.05, "logits 应随 batch 翻倍"
    assert 1.5 < b["activations"] / a["activations"] < 2.5, "激活近似翻倍"
    assert b["fragmentation"] / a["fragmentation"] < 1.5, "碎片不应陡增"
    total = sum(ledger(1).values())
    err = abs(total - 2608) / 2608
    assert err < 0.25, f"与实测 reserved 偏差 {err:.0%}，要求 <25%"
    print(f"HW2 selftest OK  total={total:.0f}MiB err={err:.1%}")


if __name__ == "__main__":
    _selftest()
