"""S5：GRPO 的核心数学（纯函数，CPU 可测）+ 整组 rollout 的骨架。

为什么把数学单独拆出来：VLM 版 GRPO 的坑几乎都在**张量形状与掩码**上，
而不是公式。把这些写成不依赖模型的纯函数，就能在没有 GPU 的情况下单测。

VLM 版四个坑（对应 verify_stage5 的四项检查）：
1. logprob 只算答案 token —— 视觉 token 与 prompt 必须 -inf 掩掉；
2. 同一组的 G 条 rollout 共用**同一份** pixel_values（否则不是同一道题）；
3. 不同分辨率的 rollout 序列长度不同 —— 组内必须按 mask 对齐再算优势；
4. 奖励拆成"正确"与"格式"两项分开记 —— 否则格式分涨了会掩盖正确率不涨。
"""

from __future__ import annotations

import torch
from torch import Tensor

from picovlm.schema import Sample


def reward(correct: bool, format_ok: bool = True,
           prior_conflict: bool = False,
           w_correct: float = 1.0, w_format: float = 0.2,
           w_conflict: float = 1.0) -> dict[str, float]:
    """总奖励 = 正确性 + 格式 + 冲突格加权。

    冲突格（答案与先验相反）给更高权重：这是本项目最想教会模型的部分。
    """
    r = float(correct) * w_correct + float(format_ok) * w_format
    if correct and prior_conflict:
        r += w_conflict * w_correct
    return {"total": r, "correct": float(correct), "format": float(format_ok),
            "conflict_bonus": w_conflict * w_correct if (correct and prior_conflict) else 0.0}


def group_advantages(rewards: Tensor, eps: float = 1e-4) -> Tensor:
    """组内标准化：A = (r - mean) / (std + eps)。全组同分时返回全 0。"""
    if rewards.numel() == 0:
        return rewards
    std = rewards.std(unbiased=False)
    if float(std) < eps:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / (std + eps)


def grpo_loss(new_logp: Tensor, old_logp: Tensor, advantages: Tensor,
              clip: float = 0.2) -> Tensor:
    """PPO/GRPO 的 clipped surrogate。new/old 形状都是 (G, T)。"""
    if new_logp.shape != old_logp.shape or new_logp.shape[0] != advantages.numel():
        raise ValueError(f"形状不匹配: {tuple(new_logp.shape)} vs "
                         f"{tuple(old_logp.shape)} vs {tuple(advantages.shape)}")
    ratio = torch.exp(new_logp - old_logp)
    adv = advantages.unsqueeze(-1)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    return -torch.min(unclipped, clipped).mean()


def masked_logprob(logits: Tensor, ids: Tensor, mask: Tensor) -> Tensor:
    """按 mask 只保留答案 token 的 logprob，其余置 0（不参与求和）。"""
    logp = torch.log_softmax(logits.float(), dim=-1)
    picked = logp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return picked * mask.float()


def format_ok(answer: str, allowed: set[str] | None = None) -> bool:
    """格式奖励：非空、不过长、在候选集里（若给了候选集）。"""
    a = answer.strip()
    if not a or len(a) > 40:
        return False
    return allowed is None or a in allowed


def make_group(sample: Sample, answers: list[str]) -> dict:
    """把同一道题的 G 个 rollout 打包。**像素只存一份**（坑 2）。"""
    return {"sample_id": sample.id, "image_path": sample.image_path,
            "answers": list(answers), "prior_type": sample.prior_type,
            "prior_answer": sample.prior_answer}


def rft_select(group: dict, gold: str, normalize) -> str | None:
    """拒绝采样：组里挑第一个判对的答案，没有就返回 None（整组丢弃）。"""
    for a in group["answers"]:
        if normalize(a) == normalize(gold):
            return a
    return None
