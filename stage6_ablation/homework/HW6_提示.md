# HW6 · 参考答案与验收标准

## HW1 · iso-token 与边际收益

### 参考实现

```python
def iso_token(points, budget, tol=0.05):
    lo, hi = budget * (1 - tol), budget * (1 + tol)
    return sorted((p for p in points if lo <= p.tokens <= hi), key=lambda p: p.tokens)

def marginal(points):
    pts = sorted(points, key=lambda p: p.tokens)
    out = []
    for a, b in zip(pts, pts[1:]):
        dt = b.tokens - a.tokens
        if dt <= 0:
            raise ValueError("token 数必须严格递增")
        out.append((a.name, b.name, (b.accuracy - a.accuracy) / dt * 1000))
    return out

def verdict(points, budget, min_gain=0.05):
    pts = sorted(points, key=lambda p: p.tokens)
    nxt = next((p for p in pts if p.tokens > budget), None)
    if nxt is None:
        return "keep"
    cur = max((p for p in pts if p.tokens <= budget), key=lambda p: p.tokens)
    gain = (nxt.accuracy - cur.accuracy) / (nxt.tokens - cur.tokens) * 1000
    return "add" if gain >= min_gain else "keep"
```

### 验收标准

- [ ] `iso_token(POINTS, 144, tol=0)` 恰好返回 pixelshuffle 一个点
- [ ] `iso_token(POINTS, 1000)` 返回 `[]`（空是合法结果，不是 bug）
- [ ] `marginal` 两项都 > 0，且**第一项 > 第二项**（边际递减）
- [ ] `Δtoken = 0` 抛 `ValueError`
- [ ] docstring 里写清了"预算落在两个采样点之间怎么办"（最常见答案是：用相邻两点线性插值，并标注这是估计）

### 常见错法

1. **用 `abs(p.tokens - budget) < tol`**：`tol=0.05` 会变成"差 0.05 个 token"，
   永远只有精确匹配。要乘 budget。
2. **marginal 用 `(acc_b - acc_a) / acc_a`**：那是相对提升，不是"每 1000 token"。
3. **verdict 直接比 accuracy**：不看 Δtoken 的边际收益，就会建议"从 64 跳到 576"，
   但那 912 个 token 可能根本不值得。

---

## HW2 · 显存账

### 参考实现

```python
def ledger(batch, seq_text=15, seq_visual=576):
    seq = seq_text + seq_visual
    weights = PARAMS_M * 1e6 * 2 / 2**20                      # 1124 MiB
    logits = batch * seq * VOCAB * 4 / 2**20                  # fp32 logits
    # 激活：每层约 3 个 [batch, seq, d] 的中间张量（attn out / mlp in / mlp out）
    # d≈896，bf16；再乘一个 1.6 的经验系数（LayerNorm、残差、临时缓冲）
    act = batch * seq * 896 * 2 * N_LAYERS * 3 * 1.6 / 2**20
    frag = 419.0                                              # 2608 - 2189
    return {"weights": weights, "logits": logits,
            "activations": act, "fragmentation": frag}
```

算出来大约是 1124 + 345 + 760 + 419 ≈ 2648 MiB，与 2608 差 1.5%。

### 验收标准

- [ ] 四项之和与 2608 MiB 偏差 < 25%
- [ ] 权重与 batch 无关；logits 严格随 batch 翻倍
- [ ] 激活随 batch 近似翻倍（系数 1.5–2.5）
- [ ] 碎片不随 batch 线性增长（<1.5×）
- [ ] docstring 写明 `reserved > allocated` 的两个来源：缓存块复用、cuBLAS/cuDNN workspace

### 常见错法

1. **logits 用 bf16**：LM head 默认 fp32 输出（`logits.float()`），差一倍。
   这也是很多小项目 OOM 的真凶——576 个视觉 token 让 `seq` 变大，logits 是
   平方级受害者。
2. **把碎片算进激活**：碎片是分配器行为，不是模型结构；混在一起就没法解释
   "为什么 batch=1 也占 2.6GB"。
3. **忘记 `+1` 的 seq**：生成时序列会再长，评估时 `max_new_tokens=8` 也要算。
