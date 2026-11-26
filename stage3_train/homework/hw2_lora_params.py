"""HW3-2 · 手算 LoRA 参数量。

目标：不查 peft 文档，推出 `lora_params()`，并与实测 **1,081,344**（1.08M）对上。

注意 Qwen2.5-0.5B 用的是 **GQA**：q_proj/o_proj 是 896×896，但 k_proj/v_proj 的
输出维度只有 `num_key_value_heads × head_dim = 2 × 64 = 128`。所以 k/v 的 LoRA
比 q/o 小——这是很多人第一次算错的地方。

跑：python -m stage3_train.homework.hw2_lora_params
"""

from __future__ import annotations

MEASURED = 1_081_344


def lora_params(d_model: int = 896, d_kv: int = 128, layers: int = 24,
                r: int = 8) -> int:
    """返回 LoRA 参数量（q/k/v/o 各一个，只有 weight，bias 不挂）。

    TODO: 每个投影的 LoRA 参数 = r * (in_dim + out_dim)：
          q_proj: r*(d_model + d_model)
          k_proj: r*(d_model + d_kv)
          v_proj: r*(d_model + d_kv)
          o_proj: r*(d_model + d_model)
          再乘层数。
    """
    raise NotImplementedError("HW3-2: 实现 lora_params")


def main() -> int:
    n = lora_params()
    err = abs(n - MEASURED) / MEASURED
    print(f"lora params {n:,}  measured {MEASURED:,}  err {err:+.2%}")
    if err > 0.02:
        print("提示：q/o 的 out_dim 是 896，k/v 是 128（GQA）")
        return 1
    print("hw3-2 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
