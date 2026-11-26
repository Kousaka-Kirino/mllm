# Stage 4 作业

## HW4-1 · Wilson 区间（必做）

`s hw1_wilson.py`：实现

```python
def wilson(k, n, z=1.96):
    ...
```

公式：

```
p̂ = k/n
denom  = 1 + z²/n
center = (p̂ + z²/(2n)) / denom
half   = z * sqrt(p̂(1-p̂)/n + z²/(4n²)) / denom
```

验收：与 `picovlm.stats.wilson_ci` 在 k/n ∈ {0/20, 1/20, 10/20, 20/20} 上误差 < 1e-9；
`n=0` 时返回 (0,1)；`k=0` 与 `k=n` 时上下界不越界。

## HW4-2 · 为什么不能逐条 bootstrap（必做）

`hw2_cluster_bootstrap.py`：构造"40 个 family，每族 2 条，族内正确性完全相关"
的数据，比较：

- 逐条重采样（n=80）得到的 CI 宽度
- 整簇重采样（抽 40 个 family）得到的 CI 宽度

验收：输出两组宽度并打印 `cluster wider? True`。
报告里写一句：逐条重采样忽略了族内相关，因此**低估**不确定性。

## HW4-3 · 手工核对错例（选做）

挑一条 SFT 的错例，回答三问：字号够不够？问题理解对了吗？先验是不是压过了图？
