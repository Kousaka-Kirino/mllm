"""HW2：测量一块文字图的"着墨高度"，并换算到模型输入尺度。

这是本项目 acuity 轴的测量基础。写它的时候你会遇到三个真实的坑：

1. **阈值**：抗锯齿让边缘有浅灰像素，阈值差 10 就能差 1px；
2. **文字块里不止一行**：刻度标签可能是 "100" 也可能是 "1e3"，
   你要数的是"最高的那一行"，不是所有有墨行；
3. **空块**：裁偏了会得到全白图，必须返回 0 而不是崩。
"""

from __future__ import annotations

import numpy as np


def ink_height(patch: np.ndarray, thr: int = 200) -> int:
    """返回灰度图中"最长连续有墨行"的行数。

    patch: 2-D uint8（H, W）灰度图；thr: 像素值 < thr 视为有墨。
    空图 / 全白返回 0。
    """
    raise NotImplementedError("TODO: 按行求 bool，再数最长连续 True 段")


def scale_to_model(px_render: float, render_size: int, input_size: int = 384) -> float:
    """把渲染尺度上的像素高度换算到模型输入尺度。"""
    raise NotImplementedError("TODO: px_render * input_size / render_size")


if __name__ == "__main__":
    # 一个 3 行有墨、两侧各 1 行空的小块
    patch = np.full((5, 4), 255, dtype=np.uint8)
    patch[1:4, :] = 10
    print("ink_height =", ink_height(patch))          # 期望 3
    print("scale_to_model(27, 1152) =", scale_to_model(27, 1152))  # 期望 9.0
