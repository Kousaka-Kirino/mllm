"""S5 · VLM 版 GRPO：四件事必须做对，否则训练"看起来在跑"但其实没学。

1. **loss 只算答案 token**：视觉 576 位 + prompt 位全部 -100；
2. **同一组 rollout 共用一份 pixel_values**：否则显存按 G 倍涨（8GB 直接 OOM）；
3. **组内优势**：A_i = (r_i − mean(r)) / (std(r) + eps)，整组同分时优势全 0；
4. **格式分与正确分分开报**：奖励 = 正确 + 0.2×格式，但报表要拆开。

关于 PPO clip 的一个诚实说明：如果每个 rollout 只做 **一次** 参数更新，新旧策略
完全相同，ratio ≡ 1，clip 永远不会生效——此时 GRPO 退化为"带组内基线的
REINFORCE"。我们仍然写进代码，是因为它决定了"能不能多 epoch 复用同一批 rollout"，
而 8GB 显存下我们**不能**复用（多 epoch 会把 ratio 推离 1，且显存不够存 logp_old）。
"""

from __future__ import annotations

import math
import re

import torch

FORMAT_MAX_CHARS = 12
_BAD = re.compile(r"[?\n]|Question|Answer:")


def format_ok(text: str) -> bool:
    t = (text or "").strip()
    return bool(t) and len(t) <= FORMAT_MAX_CHARS and not _BAD.search(t)


def reward(correct: bool, fmt_ok: bool, *, prior_conflict: bool = False,
           w_correct: float = 1.0, w_format: float = 0.2,
           w_conflict: float = 2.0) -> dict:
    """`total = correct + 0.2*format`；冲突格额外乘 `w_conflict`（可关掉做消融）。"""
    c = w_correct * float(bool(correct)) * (w_conflict if prior_conflict else 1.0)
    f = w_format * float(bool(fmt_ok))
    return {"total": c + f, "correct": c, "format": f,
            "conflict_bonus": c - w_correct * float(bool(correct))}


def group_advantages(rewards: list[float], eps: float = 1e-6) -> list[float]:
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    m = sum(rewards) / len(rewards)
    var = sum((r - m) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    if std < eps:
        return [0.0] * len(rewards)
    return [(r - m) / (std + eps) for r in rewards]


def masked_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """返回每条序列在 `labels != -100` 位置上的 log-prob 之和。"""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    logp = torch.log_softmax(shift_logits.float(), dim=-1)
    gather = logp.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return (gather * mask).sum(dim=1)


def grpo_loss(logp_new: torch.Tensor, logp_old: torch.Tensor,
              adv: torch.Tensor, clip: float = 0.2) -> torch.Tensor:
    """带 clip 的代理目标（单次更新时 ratio≡1，见模块 docstring）。"""
    ratio = torch.exp(logp_new - logp_old)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
    return -torch.min(unclipped, clipped).mean()


@torch.no_grad()
def rollout(model, tok, pixel_values, prompt_ids, *, group_size: int = 2,
            max_new_tokens: int = 8, temperature: float = 1.0) -> list[list[int]]:
    """同一份像素做 G 次采样；返回每条的**新生成 token id**。"""
    px = pixel_values.expand(group_size, *pixel_values.shape[1:])
    ids = prompt_ids.expand(group_size, -1)
    gen = model.generate(px, ids, torch.ones_like(ids),
                         max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=temperature, pad_token_id=tok.pad_token_id)
    return gen.tolist()


def answer_logprob(model, pixel_values, prompt_ids: torch.Tensor,
                   answer_ids: torch.Tensor) -> torch.Tensor:
    """给定 (图, prompt, 答案) 返回答案段的 log-prob 之和（可反传）。

    视觉前缀由模型内部拼，所以这里直接拼 input_ids 交给 forward，
    再截取"答案段"的 logits。
    """
    vis = model.encode_images(pixel_values)
    txt = model.llm.get_input_embeddings()(prompt_ids)
    embeds = torch.cat([vis, txt], dim=1)
    out = model.llm(inputs_embeds=embeds, return_dict=True)
    logits = out.logits[:, -answer_ids.shape[1] - 1:, :]
    shift_labels = answer_ids
    logp = torch.log_softmax(logits.float(), dim=-1)[:, :-1, :]
    gathered = logp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return gathered.sum(dim=1)


__all__ = ["format_ok", "reward", "group_advantages", "masked_logprob",
           "grpo_loss", "rollout", "answer_logprob", "FORMAT_MAX_CHARS"]
