# Stage 2 · 模型骨架

> 这一阶段不训练，只回答一个问题：**"图怎么变成 token 进入语言模型"**，
> 以及"这么做要多少显存、多少时间"。答不清这两个数，后面的训练计划都是许愿。

---

## 1. 为什么用 prefix 布局

做法是把视觉 token 全部放在文本前面：

```
[v1 … vNv] [t1 … tL]
```

备选是 LLaVA-1.5 的"占位符替换"（在 prompt 里放 `<image>`，再把它替换成视觉
embedding）。两者都能用，prefix 的取舍是：

| | prefix（本项目） | 占位符替换 |
|---|---|---|
| 实现复杂度 | 低，一次 `torch.cat` | 要处理 chat template 与特殊 token 边界 |
| 图文交错 | 不支持 | 支持 |
| 位置编码 | 视觉 token 也吃位置 id | 同 |

我们只做"看图答题"，不需要交错，所以选简单的那个。**这不是最优选择，
是"在 8GB 显存 + 单人 16 周"约束下的合理选择**——写进简历时要这么说，
不要写成"我们提出了 prefix 布局"。

---

## 2. 三个数字：token 数、参数量、显存

### 2.1 视觉 token 数由连接器决定

SigLIP-base-patch16-384 把 384×384 切成 24×24=576 个 patch，输出 576 个
768 维向量。连接器决定最终喂给 LLM 多少个 token：

| connector | 做法 | 视觉 token | 连接器参数 |
|---|---|---:|---:|
| `mlp` | 两层 MLP 逐 token 投影 | **576** | 2.3M |
| `resampler` | 64 个可学 query 做 cross-attention（Perceiver 式） | **64** | 20.0M |
| `pixelshuffle` | 2×2 邻域拼通道后降维 | **144** | 2.8M |

token 数直接决定注意力的二次项成本，也决定"模型有多少视觉带宽"。
E6 消融就是扫这三个点，看"同样的训练预算下，token 多是不是真的更好"。

### 2.2 参数量：分清"总参数"和"可训参数"

实测（`--load`）：

```
total 589.5M   = SigLIP 93M + Qwen2.5-0.5B 494M + connector 2.3M
trainable 496.3M  ← 这是"还没上 LoRA"的数字，别引用
```

上了 LoRA（r=8, alpha=16，只挂 q/k/v/o）之后可训参数约 **4–5M**，
这才是简历里该写的"可训练参数量"。报告里必须两者都写清楚：

> 底座 589.5M（SigLIP-base + Qwen2.5-0.5B），冻结视觉塔与 LLM 主干，
> 可训练参数 4.4M（连接器 2.3M + LoRA 2.1M）。

### 2.3 显存账（实测，bf16，单图单问）

| 项 | 数值 |
|---|---|
| 权重 | ≈1.2 GB（589.5M × 2 bytes） |
| 单次前向峰值 allocated | **2189 MiB** |
| 单次前向峰值 reserved | **2608 MiB** |
| 前向延迟（576 token，无 KV cache） | **450 ms** |

训练还要加：激活（随 batch 线性）、优化器状态（AdamW 只作用于可训参数，
4M × 8 bytes ≈ 32MB，可忽略）、梯度检查点（省激活、多一次前向）。
所以 batch=1 + 梯度检查点在 5.5GB 可用显存里是**能跑**的，但余量不宽裕。

---

## 3. 代码导览

```
picovlm/model/
├── vision.py       # FrozenVisionTower：只做前向，参数 requires_grad=False
├── connectors.py   # 三种连接器，统一签名 (B, N, Dv) -> (B, Nv, Dllm)
└── vlm.py          # assemble_inputs（纯函数）+ PicoVLM（可无权重实例化）
stage2_model/
├── smoke.py            # 冒烟：参数量 / token 数 / loss / 延迟 / 显存
├── check_loss_mask.py  # 9 条断言，专治"loss 没 mask 干净"
└── verify_stage2.py    # 验收
```

跑一遍：

```bash
python -m stage2_model.smoke                  # 骨架，秒级
python -m stage2_model.check_loss_mask        # 9 条断言
python -m stage2_model.smoke --load           # 真权重（首次下载约 1.5GB）
python -m stage2_model.verify_stage2 --load
```

---

## 4. 两个真实的坑

### 4.1 dtype 不一致

SigLIP 权重是 fp32，Qwen 用 bf16 加载。第一次跑真权重直接报：

```
RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
```

修法不是在调用处到处 `.to()`，而是在 `PicoVLM.encode_images` 里统一对齐：
视觉输出按连接器的 dtype 转、像素输入按视觉塔的 dtype 转。**这种"在边界处
统一"的写法，比在每个调用点补 `.to()` 可靠得多。**

### 4.2 loss 没 mask 干净（最贵的 bug）

如果 labels 里视觉位/prompt 位不是 `-100`，模型会顺带学"复述问题"。表现是：
loss 曲线很漂亮，答案质量不动。这类 bug 不报错，只会让你以为是"模型太小"。

`check_loss_mask.py` 用一条性质断言把它钉死：

> **改 prompt/padding 的 token id，loss 必须一字不变；改答案 token，loss 必须变。**

这条断言不需要真权重，CPU 上秒级就能跑，建议每次改完数据管线都跑一次。

---

## 5. 作业

### HW1 · 推导 token 数（必做）

`homework/hw1_connector_tokens.py`：实现 `visual_tokens(connector, input_size, patch, n_queries, factor)`，
对三种连接器返回视觉 token 数，并处理非法输入（非平方 patch 数、factor 不整除）。

### HW2 · 自己算一遍显存（必做）

`homework/hw2_vram_estimate.py`：实现 `estimate_peak_mib(params_m, batch, seq_len,
hidden, bytes_per_param=2, checkpoint=False)`，给出"权重 + 激活"的粗略上界，
并和 `smoke --load` 的实测值对比，算误差。

这题没有标准答案——重要的是你能说出"我漏算了什么"（KV cache、临时缓冲、
cuBLAS workspace）。

### HW3 · 换一个连接器看差异（选做）

把 `--connector resampler` 与 `--connector mlp` 各跑一次 `smoke --load`，
记录：视觉 token、延迟、峰值显存。把你的观察写进作业报告。

---

## 6. 验收标准

`python -m stage2_model.verify_stage2` 必须 0 FAIL，重点看：

- `connectors-shapes`：三种连接器的输出 token 数与 budget 表一致（576/64/144）；
- `mask-semantics`：视觉/prompt/padding 全为 -100；
- `loss-only-answer`：改 prompt token 不影响 loss；
- `frozen-policy`：冻结后视觉塔 0 可训参数；
- `real-weights`（加 `--load`）：真权重前向跑通。

---

## 7. 延伸

- LLaVA-1.0 / 1.5 的视觉-语言对齐设计对比（一个线性投影 vs 两层 MLP）；
- Perceiver / Q-Former：为什么用固定数量 query 压缩视觉 token；
- 关于 token 数与能力的关系：*Cambrian-1* 里的"视觉 token 预算"讨论。
