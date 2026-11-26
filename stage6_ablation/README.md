# Stage 6 · 预算与消融：同样的显存，买多少 token、多少精度

> 一句话：**这条 RQ 问的不是"能不能更强"，而是"在 8GB 上，你花的每一毫秒和每一 MiB 买到了什么"。**
> 面试官很爱问"你的模型多大"，能答"可训练参数 1.08M、视觉 token 576、单次 76ms、峰值 1.3GB"
> 的人，比答"大概 0.5B"的人可信得多。

---

## 1. 为什么单独给一个阶段

小模型项目的常见死法是"调不动就换更大的模型"，而真正能讲清楚的是**预算曲线**：
同一块卡上，视觉 token 从 576 降到 64，延迟降一半，精度掉多少？掉的那部分，
是 acuity 不够，还是先验太强？

这一阶段产出的表，会直接进你的报告和简历：

| 连接器 | 视觉 token | 连接器参数 | 单次延迟 | 峰值显存 | 精度 |
|---|---:|---:|---:|---:|---|
| mlp | 576 | 2.30M | 76 ms | 1340 MiB | 见 S4 |
| pixelshuffle | 144 | 2.75M | 41 ms | 1207 MiB | 未训练 → 不报 |
| resampler | 64 | 20.04M | 40 ms | 1219 MiB | 未训练 → 不报 |

**只有 mlp 那一行有精度数字**。另外两行只比成本，因为我们没有重新训练它们——
把它们写成"resampler 精度 0.35"就是编数据。

---

## 2. 三个可直接测量的量

### 2.1 视觉 token 数（`picovlm/budget.py`）

| 连接器 | 机制 | token |
|---|---|---:|
| `mlp` | 每个 patch 投到 LLM 维度，1:1 | 576 |
| `pixelshuffle` | 2×2 邻域拼通道后降维 | 144 |
| `resampler` | 64 个可学 query 做 cross-attention | 64 |

`count_tokens(connector, n_patches, n_queries, factor)` 是纯算术，CPU 就能验。
注意 **token 数不是免费的**：`resampler` 的 64 个 token 要靠 20M 参数的注意力
换，小数据集上它更容易过拟合（这也是为什么 pilot 主臂用 mlp）。

### 2.2 延迟（`stage6_ablation/measure.py::bench_one`）

```
torch.cuda.reset_peak_memory_stats()   # 必须每次重置，否则拿到的是历史峰值
t0 = time.time(); model(batch); dt = time.time() - t0
```

三个坑：

1. **第一次调用会包含 cuDNN autotune / 显存分配**，所以先 warmup 一次再计时；
   本脚本用 `--repeats 3` 取中位数。
2. `torch.cuda.synchronize()` 不能省——GPU 是异步的，不同步就量到的是"提交时间"。
3. Windows 上桌面程序会抢显存，**峰值会随"你开了几个浏览器标签"变化**；
   报告里必须写清楚测量时的空闲显存（本机 7099 MiB）。

### 2.3 峰值显存

`max_memory_allocated()` 只算 PyTorch 分配器里的张量；`max_memory_reserved()`
还包括缓存块。两者差 300–500 MiB 是正常的（碎片 + workspace）。写报告时用
allocated，并注明 "reserved 另计"。

---

## 3. Pareto 前沿与 iso-token 精度（`pareto.py`）

- `pareto_front(points)`：按 token 升序扫，只保留**精度严格更高**的点。
  同 token 的低精度点被支配——这一步会把"换了连接器但没训好"的点自动筛掉。
- `iso_token(points, budget, tol)`：在固定 token 预算下比较不同实现，
  这是"同算力对比"的操作化定义。
- 典型结论形状：`(64, 0.20) (144, 0.30) (576, 0.38)` —— 边际收益递减。
  如果你的曲线是直线，说明 token 还没成为瓶颈，问题在别处（数据/先验）。

---

## 4. 跑一遍

```bash
# 三个连接器的成本（真权重，约 1 分钟）
python -m stage6_ablation.measure --repeats 3

# batch 阶梯：1/2/4 的延迟与显存
python -m stage6_ablation.measure --batch-ladder

# 验收（CPU，读上面的 json）
python -m stage6_ablation.verify_stage6
```

---

## 5. 资源分级 L0–L4（写进报告的"能不能复现"一节）

| 级别 | 设备 | 能跑到哪 | 本机 |
|---|---|---|---|
| L0 | 8GB 笔记本 + 桌面程序占用 | S0–S2、S4（推理） | ✅ 就是这台 |
| L1 | 8GB 空闲 | S3 对齐 + SFT pilot | ✅ |
| L2 | 12GB | 全量 SFT + GRPO G=4 | ⚠️ 需换卡 |
| L3 | 24GB | 多连接器全训练 | ❌ |
| L4 | 80GB | 底座换 3B | ❌ |

写报告时明确"我做到 L1，L2 以上是估算"，比含糊说"能跑"更专业。

---

## 6. 代码导览

| 文件 | 作用 |
|---|---|
| `measure.py` | 三连接器成本 / batch 阶梯，落 `outputs/bench/*.json` |
| `pareto.py` | `Point`、`pareto_front`、`iso_token`、`to_markdown` |
| `verify_stage6.py` | 4 项验收（token 账、前沿、统计、实测 json） |

---

## 7. 作业

见 `homework/README.md`。两道：

- **HW1**：给定 `(tokens, accuracy)` 四个点，实现 `iso_token` 与"边际收益"
  表，回答"再加 100 token 值不值"。
- **HW2**：用 S2 的实测数字算显存账：权重 / 激活 / logits / 碎片，解释为什么
  `reserved - allocated` 是正的。

验收标准在 `homework/HW6_提示.md`，先自己写再看。

---

## 8. 验收标准（本阶段"做完"的定义）

- [ ] `python -m stage6_ablation.verify_stage6` → 4 PASS
- [ ] `outputs/bench/connectors_*.json` 有 3 行，且 `accuracy` 对未训练档位是 `null`
- [ ] 报告里有一张"成本 vs 精度"表，且**没有**把未训练档位的精度写出来
- [ ] 能回答："为什么 mlp 的延迟是 resampler 的两倍，但你可能仍然选 mlp？"
      （答：因为只有 mlp 训过；且 576 token 对细粒度读数更友好——但这需要实验证明）

---

## 9. 延伸

- 换成 `pixelshuffle` 重训一遍对齐，再跑 S4：`delta` 是"少 4 倍 token 的代价"。
- 用 `--batch-ladder` 找"延迟不涨"的最大 batch：通常 batch=2–4 吞吐翻倍而
  延迟只涨 20%，这是推理服务的批处理依据。
- 真要做服务：把 `measure.py` 的 `bench_infer` 换成 vLLM/TensorRT，比较
  prefill/decode 两段——这是另一个项目了，别在这个阶段扩。
