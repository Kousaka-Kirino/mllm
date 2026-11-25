"""S5 验收：GRPO 的四个坑逐条测。

1. logprob 只算答案 token
2. 同组 rollout 共用同一份像素
3. 组内长度不一也能算优势
4. 格式分与正确分分开记
"""

from __future__ import annotations

import sys

import torch

from picovlm.report import require, run
from picovlm.schema import Sample

from .grpo_vlm import (format_ok, group_advantages, grpo_loss, make_group,
                       masked_logprob, reward, rft_select)


def main() -> int:
    def answer_only_logprob() -> str:
        torch.manual_seed(0)
        vocab, T = 12, 6
        logits = torch.randn(2, T, vocab)
        ids = torch.randint(0, vocab, (2, T))
        mask = torch.tensor([[0, 0, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0]])
        lp = masked_logprob(logits, ids, mask)
        require(bool((lp[mask == 0] == 0).all()), "mask 外位置不应贡献 logprob")
        require(torch.isfinite(lp).all(), "logprob 出现 inf/nan")
        return "视觉/prompt 位置的 logprob 被清零"

    def same_pixels_in_group() -> str:
        s = Sample(id="x", source="synth", split="test",
                   image_path="data/images/synth/x.png", image_size=384,
                   question="q", answer="a", prior_type="marginal",
                   prior_answer="yes")
        g = make_group(s, ["a", "b", "c"])
        require(g["image_path"] == s.image_path, "组内图像路径应只存一份")
        require("pixel_values" not in g, "组内不应复制像素张量")
        return f"G={len(g['answers'])} 共用 {g['image_path']}"

    def variable_length_advantages() -> str:
        # 组内 G 条，长度不一：优势只由 reward 决定，与长度无关
        rewards = torch.tensor([1.0, 0.0, 0.2, 0.2])
        adv = group_advantages(rewards)
        require(abs(float(adv.mean())) < 1e-5, "优势均值应为 0")
        require(abs(float(adv.std(unbiased=False)) - 1.0) < 1e-3, "优势标准差应为 1")
        same = group_advantages(torch.ones(4))
        require(bool((same == 0).all()), "全组同分应返回全 0（跳过）")
        return "A=(r-mean)/std，同分组返回 0"

    def clip_and_zero_grad() -> str:
        G, T = 3, 4
        old = torch.zeros(G, T)
        new = torch.zeros(G, T)              # ratio = 1
        adv = torch.tensor([1.0, -1.0, 0.0])
        loss1 = float(grpo_loss(new, old, adv))
        require(abs(loss1 - (-float(adv.mean()))) < 1e-5,
                f"ratio=1 时 loss 应等于 -mean(A)，得到 {loss1:.4f}")
        big = torch.full((G, T), 2.0)        # ratio = e^2，必须被 clip 掉
        loss2 = float(grpo_loss(big, old, adv))
        require(abs(loss2 - loss1) > 1e-6, "clip 未生效")
        return f"ratio=1 -> {loss1:.4f}；ratio=e² 被裁剪"

    def reward_decomposed() -> str:
        a = reward(True, True, prior_conflict=False)
        b = reward(True, True, prior_conflict=True)
        c = reward(False, True)
        require(b["total"] > a["total"], "冲突格答对应有额外奖励")
        require(c["correct"] == 0.0 and c["format"] == 1.0, "正确/格式未拆分")
        require(abs((a["total"] - a["format"]) - 0.2) < 1e-9, "格式项权重不对")
        require(abs(a["total"] - (a["correct"] + 0.2 * a["format"])) < 1e-9,
                "总奖励拆分不对")
        return f"普通对 {a['total']:.1f} / 冲突对 {b['total']:.1f} / 错但格式对 {c['total']:.1f}"

    def rft_and_format() -> str:
        g = make_group(
            Sample(id="y", source="synth", split="test", image_path="p", image_size=384,
                   question="q", answer="7"), ["3", "7", "9"])
        pick = rft_select(g, "7", str.strip)
        require(pick == "7", "RFT 应挑出判对的答案")
        require(format_ok("7", {"3", "7", "9"}), "合法答案被判格式错")
        require(not format_ok("", None), "空答案应为格式错")
        return "RFT 只留答对的 rollout"

    return run([
        ("answer-only-logprob", answer_only_logprob),
        ("shared-pixels", same_pixels_in_group),
        ("variable-length-adv", variable_length_advantages),
        ("clip-and-zero", clip_and_zero_grad),
        ("reward-decomposed", reward_decomposed),
        ("rft-select", rft_and_format),
    ], title="stage5_intervene")


if __name__ == "__main__":
    sys.exit(main())
