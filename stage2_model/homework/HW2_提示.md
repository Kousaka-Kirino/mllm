# HW2 提示与验收标准

## 一个可用的参考公式

```python
def estimate_peak_mib(params_m, batch, seq_len, hidden,
                      bytes_per_param=2, checkpoint=False, layers=24, heads=14):
    w = params_m * 1e6 * bytes_per_param / MIB
    # 激活：每层保存的隐藏状态（bf16）+ MLP 中间层（约 4x）
    per_tok = hidden * bytes_per_param * (1 + 4) * layers
    act = batch * seq_len * per_tok / MIB
    # 注意力分数矩阵：heads * seq^2 * 2 bytes（fp32 softmax 前）
    attn = batch * heads * seq_len ** 2 * 4 / MIB
    if checkpoint:
        act = act / layers * 2      # 只保留层输入 + 一次重算
    return w + act + attn
```

`589.5M × 2 bytes = 1179 MB ≈ 1124 MiB` 的权重是**确定项**；
误差几乎全在激活项，所以先算权重，再用实测反推激活。

## 参考数值（实测，bf16，bs=1，576 视觉 token，15 文本 token）

| 项 | 实测 |
|---|---|
| 权重 | 1124 MiB（589.5M × 2B） |
| 前向峰值 allocated | 2189 MiB |
| 前向峰值 reserved | 2608 MiB |
| 反推激活 | ≈1065 MiB |

注意 reserved − allocated = 419 MiB，这部分是**分配器碎片 + cuBLAS workspace**，
不在任何公式里。所以"公式算准到 10%"是不现实的目标，**能算到同一量级、
并且知道往哪边偏**才是目标。

## 验收标准

1. `python -m stage2_model.homework.hw2_vram_estimate` 能跑通并打印
   estimate / measured / err 三个数；
2. 你的作业报告里必须回答：
   - 你的估算比实测偏大还是偏小？为什么？
   - `checkpoint=True` 时激活少多少？代价是什么（多一次前向）？
   - 训练时还要加什么（优化器状态、梯度、梯度检查点重算的临时峰值）？
3. 允许误差：**一个数量级内**；超过 3× 要能解释清楚。

## 常见错误

- 忘记 `1e6` / `2**20` 的换算（MB vs MiB），数字差 4.8%；
- 把 `seq_len` 只算文本 token（这里是 15 + 576 = 591）；
- 忽略注意力 `seq²` 项——576 视觉 token 时它不小；
- 用 `params_m` 时把"总参数"和"可训参数"混了（权重按总参数算）。
