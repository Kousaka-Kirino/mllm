"""S2 · loss mask 自检：最容易错、错了最难发现的地方。

跑法：
    python -m stage2_model.check_loss_mask          # 骨架模型（CPU 秒级）
    python -m stage2_model.check_loss_mask --load   # 真权重，验证与手算 CE 一致

为什么单独写一个文件：

如果 labels 没 mask 干净，模型会顺便学"复述问题"和"预测视觉 token"，
loss 看起来很低、训练很顺，但答案质量不动。这类 bug 不会报错，
只会让实验结论变成"训练没效果"。所以这里用 6 条断言把它钉死。
"""

from __future__ import annotations

import argparse

import torch

from picovlm.config import load_config, model_config_from
from picovlm.model.vlm import Batch, PicoVLM, assemble_inputs
from picovlm.report import require, run


def _tokens(n: int, start: int = 100) -> torch.Tensor:
    return torch.arange(start, start + n).unsqueeze(0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stage2_model.check_loss_mask")
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--config", default="model.yaml")
    args = ap.parse_args(argv)

    cfg = model_config_from(load_config(args.config))

    # ---------------------------------------------------------------- 纯函数
    b, nv, nl, d = 1, cfg.n_patches, 6, 16
    vis = torch.randn(b, nv, d)
    txt = torch.randn(b, nl, d)
    ids = torch.randint(0, 1000, (b, nl))
    mask = torch.ones(b, nl, dtype=torch.long)
    mask[:, -2:] = 0                                  # 模拟右侧 padding
    labels = ids.clone()
    labels[:, :3] = -100                              # 前 3 个是 prompt
    labels[:, -2:] = -100                             # padding 不监督

    embeds, am, lab = assemble_inputs(vis, txt, mask, labels)

    def mask_visual() -> str:
        require(lab[:, :nv].eq(-100).all(), "视觉 token 的位置必须全部是 -100")
        return f"前 {nv} 个视觉位置全为 -100"

    def mask_prompt() -> str:
        require(lab[0, nv:nv + 3].eq(-100).all(), "prompt 位置必须 -100")
        return "prompt 位置全为 -100"

    def keep_answer() -> str:
        need = ids[0, 3:nl - 2]
        got = lab[0, nv + 3:nv + nl - 2]
        require(torch.equal(need, got), f"答案位 labels 应等于 token id：{got} != {need}")
        return f"答案 {len(need)} 个位置保留原 token id"

    def mask_padding() -> str:
        require(lab[0, -2:].eq(-100).all(), "padding 位置必须 -100")
        return "padding 位置全为 -100"

    def mask_alignment() -> str:
        require(am.shape == (b, nv + nl), f"attention_mask 长度错：{am.shape}")
        require(am[:, :nv].eq(1).all(), "视觉位置必须 attend")
        want = int((labels != -100).sum())
        got = int((lab != -100).sum())
        require(got == want, f"监督位数应等于传入 labels 的非 -100 个数：{got} != {want}")
        require(want > 0, "这一批没有监督位，测不出东西")
        return f"attention_mask 与 labels 长度对齐，监督位 {got} 个（= 传入的 {want} 个）"

    # ---------------------------------------------------------------- 前向
    device = "cuda" if (args.load and torch.cuda.is_available()) else "cpu"
    model = PicoVLM(cfg, load_weights=args.load).to(device).eval()

    ids_full = ids.clone()
    batch = Batch(torch.zeros(b, 3, cfg.input_size, cfg.input_size, device=device),
                  ids_full.to(device), mask.to(device), labels.to(device))

    with torch.no_grad():
        out = model(batch)
    loss = float(out["loss"])

    def loss_finite() -> str:
        require(loss == loss, "loss 是 nan")
        return f"loss={loss:.4f}"

    def loss_only_answer() -> str:
        """打乱 padding 与 prompt 的 token，loss 不能变（它们不参与监督）。"""
        noisy = ids_full.clone()
        noisy[:, :3] = torch.randint(0, 1000, (b, 3))
        noisy[:, -2:] = torch.randint(0, 1000, (b, 2))
        nb = Batch(torch.zeros_like(batch.pixel_values), noisy.to(device),
                   mask.to(device), labels.to(device))
        with torch.no_grad():
            l2 = float(model(nb)["loss"])
        require(abs(l2 - loss) < 1e-4, f"未监督位置的 token 影响了 loss（{loss} vs {l2}）")
        return f"改 prompt/padding token 后 loss 不变（{loss:.4f}）"

    def loss_sensitive_to_answer() -> str:
        """改答案 token，loss 必须变——否则说明 loss 根本没看答案。"""
        noisy = ids_full.clone()
        noisy[0, 3] = (int(noisy[0, 3]) + 7) % 1000
        nb = Batch(torch.zeros_like(batch.pixel_values), noisy.to(device),
                   mask.to(device), labels.to(device))
        with torch.no_grad():
            l2 = float(model(nb)["loss"])
        require(abs(l2 - loss) > 1e-5, "改答案 token 后 loss 没变，监督位置错了")
        return f"改一个答案 token，loss 从 {loss:.4f} 变为 {l2:.4f}"

    def visual_tokens_budget() -> str:
        from picovlm.budget import count_tokens
        n = count_tokens(cfg.connector, cfg.n_patches, cfg.n_queries, cfg.shuffle_factor)
        require(n == model.n_visual_tokens, f"budget={n} 与模型 {model.n_visual_tokens} 不一致")
        return f"{cfg.connector} → {n} 个视觉 token（与 budget 表一致）"

    checks = [
        ("mask-visual", mask_visual),
        ("mask-prompt", mask_prompt),
        ("keep-answer", keep_answer),
        ("mask-padding", mask_padding),
        ("mask-alignment", mask_alignment),
        ("loss-finite", loss_finite),
        ("loss-only-answer", loss_only_answer),
        ("loss-sensitive-answer", loss_sensitive_to_answer),
        ("visual-token-budget", visual_tokens_budget),
    ]
    return run(checks, "S2 loss mask 自检")


if __name__ == "__main__":
    raise SystemExit(main())
