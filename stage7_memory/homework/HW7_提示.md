# HW7 · 参考答案与验收标准

## HW1 · recall@k 与 MRR

```python
def recall_at_k(ranked, gold, k):
    if k <= 0:
        raise ValueError("k 必须为正")
    return 1.0 if gold in ranked[:k] else 0.0

def mrr(ranked, gold):
    for i, pid in enumerate(ranked, start=1):
        if pid == gold:
            return 1.0 / i
    return 0.0

def average_recall(runs, golds, k):
    if len(runs) != len(golds):
        raise ValueError("runs 与 golds 长度不一致")
    if not runs:
        return 0.0
    return sum(recall_at_k(r, g, k) for r, g in zip(runs, golds)) / len(runs)
```

### 验收

- [ ] `["p3","p1","p3","p9"]` gold=p3 k=3 → 1.0；MRR: p3→1.0, p1→0.5, p9→0.25
- [ ] 空列表 → 0.0；`k<=0` → ValueError
- [ ] 长度不一致 → ValueError

### 为什么 recall@3 只有 recall@1 的两倍

80 页里有大量"同一个模板、不同数值"的柱状图。平均池化后它们几乎共线，
所以 top-1 命中率低；但 top-3 里出现正确页的概率也会被"同族页"抬高，
只是抬不快——**这也正是 recall@1 只有 0.083 的原因**，不是脚本 bug。

---

## HW2 · 选输入策略

```python
from picovlm.stats import mcnemar_exact

def choose_input(a, b, c, n, alpha=0.05):
    triples = [("full", a), ("salient", b), ("random", c)]
    if any(t != n for (_, (_, t)) in triples):
        raise ValueError("三项 n 必须一致")
    ranked = sorted(triples, key=lambda kv: -kv[1][0])
    (name1, (k1, _)), (name2, (k2, _)) = ranked[0], ranked[1]
    # 配对差近似成 b/c 计数（同一批题，一致性未知时取保守估计）
    only1, only2 = max(0, k1 - k2), max(0, k2 - k1)
    p = mcnemar_exact(only1, only2)
    return name1 if p < alpha else "tie"
```

（更严格的做法是保存逐题结果、真正算 b/c；本作业允许用计数近似，但要在
docstring 里写清这是保守近似。）

### 验收

- [ ] `(14,24),(13,24),(7,24)` → `"tie"`（差 1 题，p≈1.0）
- [ ] `(20,24),(8,24),(7,24)` → `"full"`
- [ ] n 不一致 → ValueError
- [ ] 说明为什么不能用"两个 Wilson 区间重叠"来判定（区间是**独立**样本的；
      这里是同一批题，配对信息被丢掉了，会系统性偏保守）

### 常见错法

1. 直接 `max(acc)` 返回 → 永远不 tie，会在 24 条样本上过度解读。
2. 用 `wilson_ci` 两个区间不重叠判断 → 需要的样本量约是配对检验的 2–4 倍。
3. 把 `salient` 的 0.542 写成"接近 full 的 0.583"而不给 CI → 面试会被追问
   "差 4 个点你确定不是噪声？"
