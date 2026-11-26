# S7 作业：召回指标与"选哪种输入"

## HW1 · recall@k 与 MRR

文件：`hw1_recall_mrr.py`

实现：

1. `recall_at_k(ranked: list[str], gold: str, k: int) -> float`：gold 在前 k 名里返回 1。
2. `mrr(ranked: list[str], gold: str) -> float`：1/首次命中的名次，未命中 0。
3. `average_recall(runs, golds, k)`：对一批查询求平均。

要点（会考）：
- `ranked` 可能有重复 id（检索器 bug），重复不应重复计数；
- 空列表 + k=5 应返回 0.0，而不是抛异常；
- `k <= 0` 应抛 `ValueError`。

自测里有一条数据：`["p3", "p1", "p3", "p9"]`，gold=`p3`，k=3 → 1.0，MRR=1.0；
gold=`p1`，k=1 → 0.0，MRR=0.5。

## HW2 · 选输入策略

文件：`hw2_choose_input.py`

```python
choose_input(full, salient, random, n) -> str  # "full" | "salient" | "random" | "tie"
```

要求：
- 输入是 `(correct, n)` 两个整数或 `(acc, n)` 元组，自己定并写清楚；
- 用 **配对差** 还是 **独立 Wilson CI**？在这里必须用**同一个 n 的配对**比较
  （同一批题），所以用 `paired_compare` 或 McNemar；
- 若最大差值不显著（p > 0.05），返回 `"tie"`——这正是本实验的情况
  （salient 0.542 vs full 0.583，n=24，不显著）。

自测：`choose_input((14,24),(13,24),(7,24),24) == "tie"`；
`choose_input((20,24),(8,24),(7,24),24) == "full"`。

## 提交

```
S7: <你的名字>
- HW1: 三个函数的实现 + 边界用例
- HW2: 你的判定 + 为什么用配对而非独立 CI
```
