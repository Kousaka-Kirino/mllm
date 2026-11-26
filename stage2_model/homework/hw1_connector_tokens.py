"""HW1 · 视觉 token 数推导。

目标：不查文档，自己把"一张 384×384 的图进 LLM 时占几个 token"算对，
并且让代码在换输入尺寸/换连接器时不静默出错。

跑：python -m stage2_model.homework.hw1_connector_tokens
"""

from __future__ import annotations


def visual_tokens(connector: str, input_size: int = 384, patch: int = 16,
                  n_queries: int = 64, factor: int = 2) -> int:
    """返回连接器输出的视觉 token 数。

    TODO:
      1. 校验 input_size % patch == 0，否则 ValueError；
      2. patches = (input_size // patch) ** 2；
      3. mlp -> patches；resampler -> n_queries；
         pixelshuffle -> patches // factor**2（不能整除时 ValueError）；
      4. 未知 connector -> ValueError。
    """
    raise NotImplementedError("HW1: 实现 visual_tokens")


def check_budget() -> bool:
    """与 picovlm.budget.count_tokens 对照，三种连接器都必须一致。

    TODO: 遍历 ("mlp", "resampler", "pixelshuffle")，逐个比较，
          不一致时打印两边的值并 return False。
    """
    raise NotImplementedError("HW1: 实现 check_budget")


def main() -> int:
    if not check_budget():
        print("hw1 FAIL")
        return 1
    got = {c: visual_tokens(c) for c in ("mlp", "resampler", "pixelshuffle")}
    want = {"mlp": 576, "resampler": 64, "pixelshuffle": 144}
    if got != want:
        print(f"hw1 FAIL: {got} != {want}")
        return 1
    print(f"hw1 PASS {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
