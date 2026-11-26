"""S5 验收：干预信号与数据构造是否真的按设计工作。

    python -m stage5_intervene.verify_stage5
"""

from __future__ import annotations

import os
import subprocess
import sys
import torch

from picovlm.config import ROOT
from picovlm.report import Skip, approx, require, run

from stage3_train.dataset import load_samples
from stage5_intervene import grpo as G
from stage5_intervene.counterfactual import numeric_variant, wording_variant
from stage5_intervene.train import augment_counterfactual, curriculum_order


def py(*args: str, timeout: int = 1200) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "0"}
    return subprocess.run([sys.executable, *args], cwd=str(ROOT),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=env)


def check_reward() -> str:
    r = G.reward(True, True)
    approx(r["total"], 1.2, 1e-9, "standard")
    approx(r["correct"], 1.0, 1e-9, "correct")
    approx(r["format"], 0.2, 1e-9, "fmt")
    rc = G.reward(True, True, prior_conflict=True)
    approx(rc["correct"], 2.0, 1e-9, "conflict x2（默认 w_conflict=2）")
    approx(rc["conflict_bonus"], 1.0, 1e-9, "bonus")
    approx(G.reward(True, True, prior_conflict=True, w_conflict=1.0)["total"],
           1.2, 1e-9, "冲突加权可关掉（消融档）")
    rf = G.reward(False, True)
    approx(rf["total"], 0.2, 1e-9, "wrong but formatted")
    return f"total=1.2 conflict_total={rc['total']}"


def check_advantages() -> str:
    require(G.group_advantages([1.0, 1.0]) == [0.0, 0.0], "同分应得 0 优势")
    a = G.group_advantages([0.0, 1.0])
    # eps=1e-6 在分母上，所以结果是 ±1 的近似值
    approx(a[0], -1.0, 1e-3, "lo")
    approx(a[1], 1.0, 1e-3, "hi")
    require(G.group_advantages([0.5]) == [0.0], "单元素应为 0")
    require(G.group_advantages([]) == [], "空列表应为空")
    return f"同分=0，[0,1]={a[0]:.6f}/{a[1]:.6f}"


def check_clip() -> str:
    """ratio≡1 时：loss 等于 -mean(adv)，且对 logp 的梯度等于 -adv。"""
    lp = torch.tensor([-1.5, -2.0], requires_grad=True)
    adv = torch.tensor([1.0, -1.0])
    loss = G.grpo_loss(lp, lp.detach().clone(), adv)
    approx(float(loss), -float(adv.mean()), 1e-6, "ratio=1 的 loss")
    loss.backward()
    # grpo_loss 对 batch 取了 mean，所以梯度要除以 n
    approx(float(lp.grad[0]), -1.0 / 2, 1e-5, "dL/dlogp_0")
    approx(float(lp.grad[1]), 1.0 / 2, 1e-5, "dL/dlogp_1")
    # 偏差很大时 clip 必须改变数值
    lp_new = lp.detach() + 2.0
    clipped = float(G.grpo_loss(lp_new, lp.detach(), adv))
    unclipped = -float((adv * torch.exp(torch.tensor([2.0, 2.0]))).mean())
    require(abs(clipped - unclipped) > 1e-3, "clip 没有生效")
    return f"loss={float(loss):.3f} grad=(-1,+1) clipped_loss={clipped:.3f}"


def check_wording() -> str:
    ss = load_samples("train", limit=40)
    n = 0
    for s in ss:
        w = wording_variant(s)
        if not w:
            continue
        n += 1
        require(w.answer == s.answer, "措辞变体不该改金标")
        require(w.image_path == s.image_path, "措辞变体不该换图")
        require(w.family_id == s.family_id, "措辞变体必须同族")
        require(w.question != s.question, "措辞变体应该换了问法")
    require(n > 0, "没有可用的措辞变体")
    return f"wording variants={n}/{len(ss)}（金标与图均不变）"


def check_numeric() -> str:
    ss = [s for s in load_samples("train", limit=80) if s.qtype == "maxcat"]
    require(ss, "没有 maxcat 样本")
    changed = 0
    for s in ss[:10]:
        cf = numeric_variant(s)
        if not cf:
            continue
        changed += int(cf.answer != s.answer)
        require(cf.image_path != s.image_path, "数值变体应重新渲染")
        require(cf.family_id == s.family_id, "数值变体必须同族")
    require(changed >= 8, f"金标改变率太低：{changed}/10")
    return f"numeric variants={len(ss[:10])} gold_changed={changed}/10"


def check_curriculum() -> str:
    ss = load_samples("train", limit=80)
    order = curriculum_order(ss)
    require(len(order) == len(ss), "顺序长度不对")
    head = order[:int(len(ss) * 0.4)]
    easy = sum(1 for s in head if s.acuity_bin in ("easy", "mid"))
    require(easy == len(head), f"头部只有 {easy}/{len(head)} 是易样本")
    require({s.id for s in order} == {s.id for s in ss}, "课程打乱后丢了样本")
    return f"head={len(head)}/{len(head)} easy-mid"


def check_augment() -> str:
    ss = load_samples("train", limit=20)
    aug = augment_counterfactual(ss)
    require(len(aug) > len(ss), "增广没有增加样本")
    require({s.id for s in ss} <= {s.id for s in aug}, "原始样本丢了")
    return f"{len(ss)} → {len(aug)}"


def check_grpo_probe() -> str:
    import torch as t
    if not t.cuda.is_available():
        raise Skip("没有 CUDA")
    r = py("-m", "stage5_intervene.train", "--method", "grpo", "--steps", "2",
           "--limit", "4", "--group-size", "2",
           "--out", "outputs/ckpt/_verify_grpo")
    if r.returncode != 0:
        raise AssertionError(f"grpo 探针失败：\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    require("saved" in r.stdout, "grpo 探针没有存盘")
    return "2 步探针跑通（整组同分会跳过，属正常）"


def check_arms() -> str:
    from picovlm.config import OUTPUTS
    if not (OUTPUTS / "eval" / "prior_dev").exists():
        raise Skip("没有 prior_dev 评测结果")
    r = py("-m", "stage5_intervene.compare_arms", "--a", "prior_dev",
           "--b", "prior_dev")
    if r.returncode != 0:
        raise AssertionError(f"compare_arms 失败：{r.stderr[-500:]}")
    require("McNemar" in r.stdout, "对比报告缺 McNemar")
    return "A vs A 生成报告（delta=0）"


def main() -> int:
    return run([
        ("reward-decomposition", check_reward),
        ("group-advantages", check_advantages),
        ("clip-behavior", check_clip),
        ("counterfactual-wording", check_wording),
        ("counterfactual-numeric", check_numeric),
        ("curriculum-order", check_curriculum),
        ("counterfactual-augment", check_augment),
        ("grpo-probe", check_grpo_probe),
        ("arms-compare", check_arms),
    ], "S5 验收（2.0）")


if __name__ == "__main__":
    raise SystemExit(main())
