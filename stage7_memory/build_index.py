"""S7 · 视觉记忆：把一页图切成 patch，建可检索的"页内记忆"。

RQ 的第三问是迁移能力，但真实用法里还有一层：**一张图很大，问题只关于其中
一小块**。8GB 卡上不可能把整页塞进 576 个 token 还保持小字号可读，所以工程
上的解法是"检索 + 放大"：先定位相关区域，再把那一块放大到 384² 喂给模型。

本文件只做索引：用**冻结的 SigLIP** 抽 patch 特征（不训练、不加参数）。
查询端可以是"另一张裁片"（visual query），这也是最诚实的设定——我们没有人
对齐好的文本编码器，用文本查图会引入一个未验证的假设。

    python -m stage7_memory.build_index --split dev_test --limit 80

输出 `outputs/memory/pages_<split>.npz`：page_ids、patch 特征（N×576×768 太大，
所以只存降采样到 24×24 的平均特征，N×576×768 → N×576×768 仍是 350MB；
改成 N×576×768 float16 ≈ 175MB，pilot 80 页可接受）。

为了省内存，pilot 默认存 `pool=4`（16×16 patch 块平均）→ N×144×768 float16。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from picovlm.config import OUTPUTS, ROOT, load_config, model_config_from
from picovlm.io_utils import ensure_dir, read_jsonl, utc_stamp
from stage3_train.dataset import load_image


def pool_patches(feats: np.ndarray, side: int, pool: int) -> np.ndarray:
    """feats: (1, side*side, D) → (1, (side/pool)^2, D) 块平均。"""
    if side % pool:
        raise ValueError(f"side={side} 不能被 pool={pool} 整除")
    d = feats.shape[-1]
    grid = feats.reshape(side, side, d)
    g2 = side // pool
    return grid.reshape(g2, pool, g2, pool, d).mean(axis=(1, 3)).reshape(1, g2 * g2, d)


def build(split: str, limit: int, pool: int, device: str = "cuda") -> Path:
    import torch

    from picovlm.model.vision import FrozenVisionTower

    cfg = load_config("model.yaml")
    mc = model_config_from(cfg)
    tower = FrozenVisionTower(mc.vision, image_size=mc.input_size).to(device).eval()
    side = mc.input_size // mc.patch

    samples = read_jsonl(ROOT / "data" / "synth" / f"{split}.jsonl")[:limit]
    feats, ids = [], []
    with torch.no_grad():
        for s in samples:
            img = load_image(s["image_path"], mc.input_size)
            px = torch.from_numpy(img[None]).to(device)
            f = tower(px).float().cpu().numpy()
            feats.append(pool_patches(f, side, pool).astype(np.float16))
            ids.append(s["id"])
    arr = np.concatenate(feats, axis=0)
    out = ensure_dir(OUTPUTS / "memory") / f"pages_{split}_{utc_stamp()}.npz"
    np.savez_compressed(out, page_ids=np.array(ids), feats=arr,
                        meta=np.array([split, str(pool), str(side)]))
    print(f"saved {out}  pages={arr.shape[0]} tokens={arr.shape[1]} dim={arr.shape[2]}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev_test")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--pool", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    build(args.split, args.limit, args.pool, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
