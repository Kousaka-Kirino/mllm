# HW3 提示与验收标准

## HW3-1 梯度累积

参考实现：

```python
import torch
torch.manual_seed(0)
m = Tiny().eval()
ids = torch.randint(0, 32, (4, 12))
lab = ids.clone()

def grads(mode):
    m.zero_grad(set_to_none=True)
    if mode == "accum":
        for i in range(4):
            _, loss = m(ids[i:i+1], lab[i:i+1])
            (loss / 4).backward()
    else:
        _, loss = m(ids, lab)
        loss.backward()
    return [p.grad.detach().clone() for p in m.parameters()]

a, b = grads("accum"), grads("batch")
num = sum(((x - y) ** 2).sum() for x, y in zip(a, b)).sqrt()
den = sum((y ** 2).sum() for y in b).sqrt()
rel = float(num / den)
```

**验收**：`rel < 1e-3`。如果 rel 很大，检查：
- 有没有 `m.zero_grad()`（忘记清零会累积上一次的梯度）；
- 除的是 `accum` 还是 `len(batch)`（这里两者都是 4，容易蒙对——把 accum 改成 8 再试）；
- 模型有没有 dropout/BN（eval 模式关掉）。

**为什么这条对 8GB 有意义**：显存只跟单次前向的 batch 有关，梯度累积把"等效
batch"和"显存占用"解耦了。这也是 S6 预算实验的基础。

## HW3-2 LoRA 参数量

参考实现：

```python
def lora_params(d_model=896, d_kv=128, layers=24, r=8):
    per_layer = r * ((d_model + d_model)      # q
                     + (d_model + d_kv)       # k
                     + (d_model + d_kv)       # v
                     + (d_model + d_model))   # o
    return layers * per_layer
```

`24 × 8 × 5632 = 1,081,344` ✅

**常见错误**：
- 把 k/v 也当成 896×896 → 得 1.38M（偏大 27%）；
- 忘了乘 2 个矩阵（A 和 B）→ 得 0.54M；
- 把 alpha 算进参数量（alpha 只是缩放因子，不增加参数）。

## HW3-3 学习率探针

三组 `--max-steps 30` 的 val_loss 一般会呈现"5e-5 太慢 / 2e-4 合适 / 1e-3 后期震荡"。
**注意**：这里的结论只对 318 条 pilot 数据 + 60 步这个尺度成立，不能外推到全量。

## 验收标准

| 作业 | 通过条件 |
|---|---|
| HW3-1 | 打印 `equivalent=True`，rel_err < 1e-3 |
| HW3-2 | 打印 `lora params 1,081,344`，err < 2% |
| HW3-3 | 三组 val_loss 表 + 一句结论（不要求某一组更好） |
