"""S7 验收：索引、检索几何、引用格式。CPU 即可，不加载模型。

`memory-eval` 一项读 `outputs/memory/memory_eval_*.json`，没有就 SKIP——验收
脚本不应该偷偷跑一次 72 次生成的评测（那是 `evaluate.py` 的活）。
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image, ImageDraw

from picovlm.config import OUTPUTS
from picovlm.report import Skip, approx, require, run
from stage7_memory.build_index import pool_patches
from stage7_memory.index import PageIndex, citation_ok, recall_at_k
from stage7_memory.retrieve import (Region, best_salient, crop_zoom, full_image,
                                    grid_bbox, random_region, saliency_scores)


def c_index() -> str:
    idx = PageIndex(dim=8)
    for i in range(5):
        v = np.zeros(8, dtype=np.float32)
        v[i] = 1.0
        idx.add(f"p{i}", v)
    hit = idx.search(idx.vecs[2], k=1)
    require(hit[0][0] == "p2", f"top1 应为 p2，实际 {hit}")
    require(abs(hit[0][1] - 1.0) < 1e-5, "自查询余弦应为 1")
    require(idx.search(np.zeros(8))[0][1] == 0.0, "零向量的分数应为 0")
    require(len(PageIndex(dim=4)) == 0 and PageIndex(dim=4).search(np.ones(4)) == [],
            "空索引应返回 []")
    q = [(idx.vecs[i], f"p{i}") for i in range(5)]
    approx(recall_at_k(idx, q, k=1), 1.0, 1e-9, "recall@1")
    return f"5 页自查询 recall@1=1.0，零向量不崩"


def c_pool() -> str:
    f = np.random.default_rng(0).normal(size=(1, 24 * 24, 16)).astype(np.float32)
    p = pool_patches(f, side=24, pool=4)
    require(p.shape == (1, 36, 16), f"形状应为 (1,36,16)，实际 {p.shape}")
    p1 = pool_patches(f, side=24, pool=1)
    require(np.allclose(p1, f), "pool=1 应等价于原特征")
    # 每块平均必须等于该块原始值的均值
    g = f.reshape(24, 24, 16)
    blk = g[:4, :4].reshape(16, 16).mean(axis=0)
    require(np.allclose(p[0, 0], blk), "第一块的平均值不对")
    try:
        pool_patches(f, side=24, pool=5)
    except ValueError:
        pass
    else:
        raise AssertionError("不能被整除时应报 ValueError")
    return "24²→6²，块平均与手算一致"


def c_regions() -> str:
    size, side = 240, 16
    require(grid_bbox(0, side, size) == (0, 0, 15, 15), "第 0 格 bbox 不对")
    require(grid_bbox(side - 1, side, size) == (225, 0, 240, 15), "最后一格 bbox 不对")
    for i in range(side * side):
        r = Region(*grid_bbox(i, side, size))
        require(0 <= r.x0 < r.x1 <= size and 0 <= r.y0 < r.y1 <= size, f"格子 {i} 越界")
    r = Region(*grid_bbox(0, side, size))
    padded = Region(0, 0, 30, 30)
    require(padded.x0 >= 0 and padded.y1 <= size, "padding 后不应越界")

    img = Image.new("RGB", (240, 240), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([10, 200, 40, 230], fill="black")     # 左下角一块黑
    s = saliency_scores(img, side=16)
    require(s.shape == (16, 16), "saliency 形状不对")
    best = best_salient(img, side=16)
    require(best.x1 <= 240 and best.y1 <= 240, "salient 区域越界")
    require(best.y0 > 150, f"应挑到下半部，实际 y0={best.y0}")
    return f"格子几何合法，salient 落在黑块附近（y0={best.y0}）"


def c_zoom() -> str:
    img = Image.new("RGB", (1152, 1152), "white")
    ImageDraw.Draw(img).rectangle([0, 0, 575, 575], fill="black")
    arr = crop_zoom(img, Region(0, 0, 576, 576))
    require(arr.shape == (3, 384, 384), f"形状应为 (3,384,384)，实际 {arr.shape}")
    require(arr.dtype == np.float32, "dtype 应为 float32")
    require(-1.0 <= arr.min() and arr.max() <= 1.0, "取值应落在 [-1,1]")
    require(arr.max() < -0.5, "黑块放大后应接近 -1")
    full = full_image(img)
    require(full.shape == (3, 384, 384), "full_image 形状不对")
    r1 = random_region(img, seed=3)
    r2 = random_region(img, seed=3)
    require((r1.x0, r1.y0) == (r2.x0, r2.y0), "同种子应给同一区域")
    return "裁剪+放大形状/范围正确，随机区域可复现"


def c_citation() -> str:
    for ok in ("page 3", "p.3", "#3", "第3页", "见 p3"):
        require(citation_ok(ok, ["p3"]), f"应认作引用：{ok}")
    for bad in ("p30", "第 30 页", "no cite", "page 13"):
        require(not citation_ok(bad, ["p3"]), f"不应认作 p3 引用：{bad}")
    return "认 page/p./#/第N页，拒绝 p30 这类子串误判"


def c_eval() -> str:
    files = sorted((OUTPUTS / "memory").glob("memory_eval_*.json"))
    if not files:
        raise Skip("没有 outputs/memory/memory_eval_*.json（先跑 evaluate.py）")
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    require("recall" in d or "zoom" in d, "评测文件里没有结果")
    if "recall" in d:
        r = d["recall"]
        require(0.0 <= r["recall_at_1"] <= 1.0, "recall 越界")
        require(r["recall_at_3"] >= r["recall_at_1"] - 1e-9, "recall@3 不应低于 recall@1")
    if "zoom" in d:
        z = d["zoom"]
        for k in ("full", "salient", "random"):
            require(0.0 <= z[k]["accuracy"] <= 1.0, f"{k} 准确率越界")
    return f"{files[-1].name} 结构正确"


def main() -> int:
    return run([
        ("page-index", c_index),
        ("patch-pooling", c_pool),
        ("region-geometry", c_regions),
        ("crop-zoom", c_zoom),
        ("citation-format", c_citation),
        ("memory-eval", c_eval),
    ], "S7 验收（2.0）")


if __name__ == "__main__":
    raise SystemExit(main())
