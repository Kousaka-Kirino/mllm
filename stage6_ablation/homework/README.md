# S6 作业：把"预算"算给别人看

两个作业都不需要 GPU，但要能自圆其说。

## HW1 · iso-token 与边际收益

文件：`hw1_iso_token.py`

给定这台机器上真实测到的三个点（token 数 → 精度，来自 S4 + S6）：

```python
POINTS = [(64, 0.20), (144, 0.30), (576, 0.38)]
```

实现：

1. `iso_token(points, budget, tol=0.05)`：返回 token 数在 `budget±tol*budget`
   范围内的点（可能为空、可能多个）。
2. `marginal(points)`：对相邻两点返回 `(Δaccuracy / Δtokens) * 1000`，
   即"每多给 1000 个 token，精度涨几个点"。
3. `verdict(points, budget)`：在给定预算下回答"再加 token 值不值"，
   判据自定，但要在 docstring 里写清楚。

自测（`python homework/hw1_iso_token.py`）会检查：单调性、`∞` 处理、
预算落在两个点之间时的行为。

## HW2 · 显存账

文件：`hw2_vram_ledger.py`

用 S2 实测数字（`stage2_model/README.md`）：

- 权重：SigLIP 93M + Qwen 494M + connector 2.3M（bf16）
- 峰值 allocated：2189 MiB（batch=1，seq 15+576）
- 峰值 reserved：2608 MiB

实现 `ledger(batch, seq_text=15, seq_visual=576)`，返回四个数（MiB）：
`weights`、`logits`、`activations`（粗估即可，注明公式）、`fragmentation`。
要求 `sum` 与实测 reserved 的偏差 < 25%，并在 docstring 里解释
`reserved > allocated` 的两个来源（缓存块复用、cuBLAS workspace）。

自测会检查：batch 翻倍时 logits 翻倍、activations 近似翻倍、
fragmentation 不随 batch 线性增长。

## 提交

```
S6: <你的名字>
- HW1: marginal 表（可贴输出）
- HW2: ledger 输出 + 偏差百分比 + 一句"哪一项最难估"
```
