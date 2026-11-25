"""视觉 token 预算的算术。纯 Python，不 import torch。

这个模块回答一个问题：**同一张图，不同连接器会喂给 LLM 多少 token？**
S6（效率）全部建立在这个小函数上。
"""

from __future__ import annotations

CONNECTORS = ("mlp", "resampler", "pixelshuffle")


def count_tokens(connector: str, n_patches: int, n_queries: int = 64,
                 factor: int = 2) -> int:
    if connector == "mlp":
        return n_patches
    if connector == "resampler":
        return n_queries
    if connector == "pixelshuffle":
        if n_patches % (factor * factor) != 0:
            raise ValueError("patch 数必须能被 factor^2 整除")
        return n_patches // (factor * factor)
    raise KeyError(f"未知连接器 {connector!r}；可选 {CONNECTORS}")


def patches(input_size: int, patch: int) -> int:
    if input_size % patch:
        raise ValueError("input_size 必须是 patch 的整数倍")
    return (input_size // patch) ** 2


def tokens_for(input_size: int, patch: int, connector: str,
               n_queries: int = 64, factor: int = 2) -> int:
    return count_tokens(connector, patches(input_size, patch), n_queries, factor)


def iso_token_sweep(input_size: int = 384, patch: int = 16) -> list[dict]:
    """给出 (连接器, token 数) 的完整档位表，S6 的矩阵实验直接用它。"""
    rows = []
    for conn in CONNECTORS:
        try:
            n = tokens_for(input_size, patch, conn)
        except ValueError:
            continue
        rows.append({"connector": conn, "tokens": n,
                     "ratio": n / patches(input_size, patch)})
    return sorted(rows, key=lambda r: r["tokens"])


if __name__ == "__main__":
    for row in iso_token_sweep():
        print(row)
