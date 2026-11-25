"""S2 验收：连接器形状、token 预算、labels 掩码、可跑通的前向。

CPU 就能跑（`load_weights=False` 用替身模块）。真权重只在 `--load` 时加载，
加载失败记 SKIP 而不是 FAIL——因为骨架代码不该强迫你下 1.5GB 权重。
"""

from __future__ import annotations

import sys

import torch

from picovlm.budget import count_tokens
from picovlm.config import ModelConfig
from picovlm.model import Batch, PicoVLM, assemble_inputs, build_connector
from picovlm.report import Skip, require, run


def main() -> int:
    cfg = ModelConfig()
    n_patch = cfg.n_patches

    def connector_shapes() -> str:
        rows = []
        for name, expect in (("mlp", n_patch), ("resampler", cfg.n_queries),
                             ("pixelshuffle", n_patch // 4)):
            conn = build_connector(name, 32, 48, n_queries=cfg.n_queries, factor=2)
            x = torch.randn(2, n_patch, 32)
            y = conn(x)
            require(tuple(y.shape) == (2, expect, 48),
                    f"{name} 输出 {tuple(y.shape)}，期望 {(2, expect, 48)}")
            rows.append(f"{name}:{expect}")
        return " / ".join(rows)

    def budget_matches() -> str:
        m = PicoVLM(cfg, load_weights=False, d_vision=8, d_llm=8)
        expect = count_tokens(cfg.connector, n_patch, cfg.n_queries, cfg.shuffle_factor)
        require(m.n_visual_tokens == expect,
                f"模型报告 {m.n_visual_tokens}，budget 计算 {expect}")
        return f"3 种连接器 / {expect} tokens（默认配置）"

    def assemble_mask() -> str:
        b, nv, d, l = 2, 10, 8, 5
        vis = torch.randn(b, nv, d)
        txt = torch.randn(b, l, d)
        mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
        labels = torch.tensor([[-100, -100, 7, 8, 9], [-100, -100, 7, -100, -100]])
        embeds, am, lb = assemble_inputs(vis, txt, mask, labels)
        require(tuple(embeds.shape) == (b, nv + l, d), "拼接维度错")
        require(bool((am[:, :nv] == 1).all()), "视觉位置 attention 必须为 1")
        require(bool((lb[:, :nv] == -100).all()), "视觉位置 labels 必须为 -100")
        require(bool((lb[1, nv + 3:] == -100).all()), "padding 位置 labels 必须为 -100")
        require(bool((lb[0, nv + 2:nv + 5] != -100).all()), "答案 token 应保留梯度")
        return "6 项掩码规则全部成立"

    def forward_smoke() -> str:
        m = PicoVLM(cfg, load_weights=False, d_vision=16, d_llm=24, vocab=100)
        b = 2
        batch = Batch(
            pixel_values=torch.randn(b, 3, 64, 64),
            input_ids=torch.randint(0, 100, (b, 6)),
            attention_mask=torch.ones(b, 6, dtype=torch.long),
            labels=torch.tensor([[-100, -100, -100, 5, 6, 7]] * b),
        )
        out = m(batch)
        require(torch.isfinite(out["loss"]), "loss 不是有限值")
        require(out["logits"].shape[0] == b, "logits batch 错")
        expect_len = cfg.n_patches + 6 if cfg.connector == "mlp" else m.n_visual_tokens + 6
        require(out["logits"].shape[1] == expect_len,
                f"序列长 {out['logits'].shape[1]}，期望 {expect_len}")
        return f"loss={out['loss'].item():.3f} seq={out['logits'].shape[1]}"

    def loss_only_on_answer() -> str:
        """模型报的 loss 必须等于"只在答案 token 上算的交叉熵"。"""
        m = PicoVLM(cfg, load_weights=False, d_vision=8, d_llm=8, vocab=50)
        torch.manual_seed(0)
        vis = torch.randn(1, 3, 32, 32)
        ids = torch.randint(0, 50, (1, 4))
        am = torch.ones(1, 4, dtype=torch.long)
        nv = m.n_visual_tokens
        labels = torch.tensor([[-100, -100, 3, 4]])
        out = m(Batch(vis, ids, am, labels))
        text_logits = out["logits"][:, nv:]
        manual = torch.nn.functional.cross_entropy(
            text_logits[0, 2:], torch.tensor([3, 4]))
        require(abs(float(out["loss"]) - float(manual)) < 1e-5,
                f"loss={float(out['loss']):.5f} 与手算 {float(manual):.5f} 不符")
        # 再多掩掉一个答案 token，loss 必须变（不然说明掩码没生效）
        out2 = m(Batch(vis, ids, am, torch.tensor([[-100, -100, -100, 4]])))
        require(abs(float(out2["loss"]) - float(manual)) > 1e-5, "掩码没有生效")
        return f"loss={float(manual):.3f}，仅由 2 个答案 token 决定"

    def real_weights() -> str:
        if "--load" not in sys.argv:
            raise Skip("加 --load 才下载真权重（约 1.5GB）")
        m = PicoVLM(cfg, load_weights=True)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return f"可训练参数 {n / 1e6:.1f}M"

    return run([
        ("connector-shapes", connector_shapes),
        ("budget-matches", budget_matches),
        ("assemble-mask", assemble_mask),
        ("forward-smoke", forward_smoke),
        ("loss-only-on-answer", loss_only_on_answer),
        ("real-weights", real_weights),
    ], title="stage2_model")


if __name__ == "__main__":
    sys.exit(main())
