# HW5 提示与验收标准

## HW5-1 组内优势

参考实现：

```python
import math

def advantages(rewards, eps=1e-6):
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n      # 总体方差，÷n
    std = math.sqrt(var)
    if std < eps:                                        # 整组同分
        return [0.0] * n
    return [(r - mean) / (std + eps) for r in rewards]
```

算 `[0,0,0,1]`：mean=0.25，var=(3×0.0625+0.5625)/4=0.1875，std=0.4330，
于是 `−0.577, −0.577, −0.577, +1.732`。

验收：打印 `hw5-1 PASS`，四个数误差 < 1e-3。

**用 ÷(n−1) 会得到 −0.5/1.5**，和 GRPO 原文对不上。为什么是总体方差：
这一组不是从某个总体里抽的样本，它就是"当下这批 rollout 的全部"，方差是被
当成确定量算的。分母差一点点在 R=4 时是 13% 的优势差，会直接改梯度。

**整组同分必须返回全 0**，不是返回 nan。真实 GRPO 里这类组非常常见（本项目
12 条探针里 5/6 组是这样），它们的梯度应该是 0 而不是爆炸——这也是 S5 里
`zero_adv_groups` 这个计数器存在的原因。

## HW5-2 clip 什么时候生效

参考实现：

```python
def effective_delta(adv=1.0, clip=0.2, hi=3.0, step=0.01):
    base = 0.0
    for i in range(1, int(hi / step) + 1):
        d = i * step
        new = [base + d]
        old = [base]
        # 用真实的 grpo_loss，看 adv>0 时 ratio 超上界才截断
        loss = grpo_loss(new, old, adv, clip=clip)
        # 关键：ratio > 1+clip 时，loss 不再等于 -adv*ratio
        if abs(loss.item() - (-adv * (1 + clip))) < 1e-9 and d > 0.18:
            return d
    return float("nan")
```

解析值：`log(1 + clip) = log(1.2) ≈ 0.1823`。

验收：打印的 δ 与 0.182 相差 < 0.02，并且能说出那句话：
**单次更新 ratio≡1，clip 恒不生效；只有同一批 rollout 被反复更新（>1 个
inner epoch）时，ratio 才可能越过 1±clip。**

这就是本项目 S5 的取舍：8GB 卡上只做 1 次 inner update，clip 是"为了和
GRPO 定义对齐"而保留，实测不会触发。写报告时要把它当已知局限写下，
而不是假装用了完整的 PPO 约束。

## HW5-3 RFT 对照（选做）

```python
from stage5_intervene.rft import build
# 在 dev 子集上采 G=4，统计 keep_rate = 保留样本 / 采样总数
```

好的答案会给出 keep_rate 的量级并补一句：keep_rate 高（比如 >0.3）意味着
"拒绝采样 + SFT"是很强的廉价基线，GRPO 相对它的增益必须显著，否则不该上
GRPO。本项目 S5 的真实结论正是这条——v1 版 SFT 与 GRPO 打平，且 SFT 便宜
40×，所以最终交付的是"负结果 + 一个可信的解释"。

## 验收标准

| 作业 | 通过条件 |
|---|---|
| HW5-1 | 打印 `hw5-1 PASS`；能解释为什么用总体标准差、同分返回 0 |
| HW5-2 | 与 `ln 1.2 ≈ 0.182` 偏差 < 0.02，并写出"单次更新 clip 不生效" |
| HW5-3 | 给出 keep_rate 数值 + 一句对 GRPO 的威胁评估 |
