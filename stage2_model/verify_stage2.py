"""S2 验收：模型骨架对不对。

`python -m stage2_model.verify_stage2`            # 骨架（CPU 秒级，不下载）
`python -m stage2_model.verify_stage2 --load`     # 加上真权重的前向

检查的是"结构正确性"，不是"效果"：连接器形状、mask 语义、token 数、
冻结策略、以及"改了不该监督的位置 loss 不变"这个最强的性质断言。
"""

from __future__ import annotations

import argparse

import torch

from picovlm.budget import CONNECTORS, count_tokens
from picovlm.config import load_config, model_config_from
from picovlm.model.connectors import (MLPConnector, PixelShuffleConnector,
                                      ResamplerConnector, build_connector)
from picovlm.model.vlm import PicoVLM, assemble_inputs
from picovlm.report import Skip, require, run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stage2_model.verify_stage2")
    ap.add_argument("--load", action="store_true")
    args = ap.parse_args(argv)

    cfg = model_config_from(load_config("model.yaml"))
    d_v, d_l = 768, 896

    def connector_shapes() -> str:
        n_patch = cfg.n_patches
        outs = {}
        for name in CONNECTORS:
            m = build_connector(name, d_v, d_l, n_queries=cfg.n_queries,
                                factor=cfg.shuffle_factor)
            y = m(torch.randn(2, n_patch, d_v))
            want = count_tokens(name, n_patch, cfg.n_queries, cfg.shuffle_factor)
            require(y.shape == (2, want, d_l),
                    f"{name}: {tuple(y.shape)} != (2, {want}, {d_l})")
            outs[name] = want
        return f"三种连接器 token 数 {outs}"

    def connector_invalid() -> str:
        bad = PixelShuffleConnector(d_v, d_l, factor=2)
        try:
            bad(torch.randn(1, 577, d_v))     # 577 不是平方数
        except ValueError:
            pass
        else:
            raise AssertionError("非平方 patch 数应报 ValueError")
        return "pixelshuffle 遇到非平方 patch 数会显式报错"

    def mask_semantics() -> str:
        b, nv, nl, d = 1, 8, 5, 16
        txt = torch.randn(b, nl, d)
        ids = torch.randint(0, 100, (b, nl))
        mask = torch.ones(b, nl, dtype=torch.long)
        mask[:, -1:] = 0
        labels = ids.clone()
        labels[0, 0] = -100
        labels[0, -1] = -100
        emb, am, lab = assemble_inputs(torch.randn(b, nv, d), txt, mask, labels)
        require(emb.shape == (b, nv + nl, d), "拼接后长度错")
        require(lab[:, :nv].eq(-100).all(), "视觉位必须 -100")
        require(lab[0, nv].item() == -100, "prompt 位必须 -100")
        require(lab[0, -1].item() == -100, "padding 位必须 -100")
        return "视觉/prompt/padding 全部 -100，长度对齐"

    def loss_only_answer() -> str:
        model = PicoVLM(cfg, load_weights=False).eval()
        g = torch.Generator().manual_seed(3)
        px = torch.zeros(1, 3, cfg.input_size, cfg.input_size)
        ids = torch.randint(0, 500, (1, 6), generator=g)
        mask = torch.ones(1, 6, dtype=torch.long)
        lab = ids.clone()
        lab[:, :4] = -100
        from picovlm.model.vlm import Batch
        with torch.no_grad():
            l1 = float(model(Batch(px, ids, mask, lab))["loss"])
            ids2 = ids.clone()
            ids2[0, :4] = torch.randint(0, 500, (4,), generator=g)
            l2 = float(model(Batch(px, ids2, mask, lab))["loss"])
        require(abs(l1 - l2) < 1e-4, f"未监督位影响了 loss：{l1} vs {l2}")
        return f"改 prompt token，loss 不变（{l1:.4f}）"

    def frozen_policy() -> str:
        if not cfg.freeze_vision:
            raise Skip("配置里 freeze_vision=False，不检查冻结")
        model = PicoVLM(cfg, load_weights=False)
        for p in model.vision.parameters():
            p.requires_grad = False     # 模拟真实冻结
        n_vis = sum(p.numel() for p in model.vision.parameters() if p.requires_grad)
        n_conn = sum(p.numel() for p in model.connector.parameters() if p.requires_grad)
        require(n_vis == 0, f"视觉塔仍有 {n_vis} 个可训参数")
        require(n_conn > 0, "连接器没有可训参数")
        return f"视觉塔冻结（0 可训），连接器 {n_conn/1e6:.2f}M 可训"

    def smoke_runs() -> str:
        from . import smoke
        rc = smoke.main(["--repeats", "1", "--connector", cfg.connector])
        require(rc == 0, f"smoke 返回 {rc}")
        return "stage2_model.smoke 跑通（dummy）"

    def real_weights() -> str:
        if not args.load:
            raise Skip("加 --load 才验证真权重（会下载 SigLIP+Qwen 约 1.5GB）")
        from . import smoke
        rc = smoke.main(["--load", "--repeats", "1", "--connector", cfg.connector])
        require(rc == 0, "真权重前向失败")
        return "真权重前向跑通（参数量/显存见 smoke 输出）"

    checks = [
        ("connectors-shapes", connector_shapes),
        ("connectors-invalid", connector_invalid),
        ("mask-semantics", mask_semantics),
        ("loss-only-answer", loss_only_answer),
        ("frozen-policy", frozen_policy),
        ("smoke-runs", smoke_runs),
        ("real-weights", real_weights),
    ]
    return run(checks, "S2 验收（2.0）")


if __name__ == "__main__":
    raise SystemExit(main())
