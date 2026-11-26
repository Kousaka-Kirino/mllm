# HW4 提示与验收标准

## HW4-1 Wilson 区间

参考实现：

```python
import math

def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
```

验收：四个点 + n=0，误差 < 1e-9。

**为什么不用 Wald 区间（p̂ ± 1.96√(p̂(1−p̂)/n)）**：k=0 或 k=n 时宽度为 0，
明显荒谬；小样本下还常越界。Wilson 是把二项分布"反解"出来的 score interval。

## HW4-2 逐条 vs 整簇 bootstrap

参考实现：

```python
def compare(n_families=40, per_family=2):
    fams = make_clustered(n_families, per_family)
    flat = [x for f in fams for x in f]
    lo, hi = _ci(flat)
    per_item = hi - lo

    rng = random.Random(7)
    means = []
    for _ in range(2000):
        pick = [fams[rng.randrange(n_families)] for _ in range(n_families)]
        vals = [x for f in pick for x in f]
        means.append(sum(vals) / len(vals))
    means.sort()
    cluster = means[int(0.05 * len(means))] - 0  # 自己按 2.5%/97.5% 取
    lo2, hi2 = means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]
    return per_item, hi2 - lo2
```

验收：`cluster wider? True`。

**结论一句话**：族内样本共享同一个"难度"，把它当独立样本重采样，等于假装
有 80 个独立观测（实际只有 40 个族），于是 CI 变窄、**低估**不确定性。
这就是 2.0 规划里要求"按 family 聚类 bootstrap"的原因。

## HW4-3 错例核对

没有标准答案。好的核对会具体到：这张图的字号是几个像素、模型答了什么、
金标是什么、先验答案是什么——四件事对上才能下判断。

## 验收标准

| 作业 | 通过条件 |
|---|---|
| HW4-1 | 打印 `hw4-1 PASS` |
| HW4-2 | 打印 `cluster wider? True`，且报告里有那句"低估"的结论 |
| HW4-3 | 3 行分析，包含字号/预测/金标/先验四项 |
