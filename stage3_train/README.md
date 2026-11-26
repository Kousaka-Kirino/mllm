# Stage 3 · 训练（两阶段）

> 这一阶段只做两件事：先让连接器"学会翻译"，再让 LoRA 学会"回答问题"。
> 每一步都要有可复现的数字——loss 曲线不算结果，**可复现的命令 + 存下的权重 + 日志**才算。

---

## 1. 为什么要两阶段

LLaVA 式训练是两阶段：先对齐（只训连接器），再指令微调（训连接器 + LoRA）。
这里做的是同一个思路的缩放版：

| | 阶段 A：对齐 | 阶段 B：SFT |
|---|---|---|
| 可训 | 连接器 2.30M | 连接器 2.30M + LoRA 1.08M = **3.38M** |
| 冻结 | ViT + LLM | ViT + LLM 主干 |
| lr | 1e-3（大） | 2e-4（小） |
| 目标 | 视觉特征 ↔ 词嵌入空间对齐 | 学会答题格式与任务 |

先对齐的理由：视觉特征（768 维）和词嵌入（896 维）不同空间，直接上 LoRA 会让
"空间映射"和"任务学习"两个梯度打架；而 8GB 显存 + 几百条样本，没有余量同时学。

**为什么 lr 差 5 倍**：连接器是从随机初始化开始（要大步走），LoRA 是接在预训练
权重上的低秩增量（大步会破坏预训练能力）。

---

## 2. 显存与速度（实测，RTX 4060 Laptop 8G，bf16）

| 配置 | 峰值 allocated | 每步耗时（bs=1，accum 4） |
|---|---:|---:|
| 对齐（只训连接器） | **3508 MiB** | ≈0.75 s |
| SFT（LoRA + 梯度检查点） | **3769 MiB** | ≈1.2 s |

这比 S2 的前向（2189 MiB）多出来的部分主要是：反向传播的激活 + 梯度 + 优化器状态。
余量到 5.5GB 还有 ~1.7GB，所以 `bs=2` 或 `max_text_tokens` 加到 256 是可能的，
但**先别加**——先把 pilot 跑通，再按 S6 的阶梯做预算实验。

---

## 3. 实测曲线（pilot，train 318 条，val 78 条）

对齐（60 步，lr 1e-3 OneCycle）：

```
step  5  loss 6.3688
step 15  loss 1.1824
step 40  loss 0.8620
step 60  loss 0.6062     val_loss 0.4830
```

SFT（70 步，lr 2e-4，LoRA r=8）：

```
step 10  loss 0.8107
step 40  loss 0.8720
step 60  loss 0.5458     val 生成式准确率 0.75（n=8）
```

两个诚实提醒：

1. **val 只有 8 条**，0.75 的置信区间宽到没意义（Wilson 95% 约 0.41–0.93）。
   要等到 S4 用 80 条 dev_test 才有讨论价值。
2. pilot 的 318 条训练样本里，训练集本身就按 ρ=0.74 偏斜过。**先记住这件事**：
   如果后面模型在"先验冲突"格子上表现差，部分原因可能是这里学出来的。

---

## 4. 代码导览

```
stage3_train/
├── dataset.py     # Sample → (pixel, input_ids, labels)；PROMPT/EOS 契约
├── align.py       # S3-A：冻 ViT+LLM，只训 connector
├── sft.py         # S3-B：连接器 + LoRA
└── verify_stage3.py
configs/align_pilot.yaml / sft_pilot.yaml
```

跑一遍：

```bash
python -m stage3_train.align --dry-run          # CPU，验证管线
python -m stage3_train.align --limit 318        # GPU 真训（约 1 分钟）
python -m stage3_train.sft --dry-run
python -m stage3_train.sft --limit 318 --max-steps 100
python -m stage3_train.verify_stage3 --load
```

---

## 5. 四个真实的坑

### 5.1 `--max-steps` 会把 OneCycle 的学习率表压扁

`total_steps = max_steps`，所以 `--max-steps 12` 时 lr 在 12 步内就衰减到 ~0：
日志里 lr 从 9.9e-4 掉到 1.8e-5。这是**预期行为**，但如果你用它做"快速探针"，
要意识到你看到的是"训练末尾的学习率"，不能拿来判断收敛速度。

### 5.2 替身 LLM 挂不上 LoRA

`--dry-run` 用 `_DummyLLM`（只有 embedding + Linear head），没有 q/k/v/o 投影，
`get_peft_model` 直接抛 `NoMatchingPeftModuleError`。处理方式不是"吞掉异常"，
而是**显式检测并打印警告**，把 LoRA 通路留给真权重验证（`verify_stage3 --load`）。

### 5.3 Windows 下子进程输出是 GBK

`subprocess.run(..., text=True)` 在中文 Windows 上默认用 GBK 解码，子进程打印
中文时报 `UnicodeDecodeError: 'gbk' codec can't decode byte 0xad`。
必须显式 `encoding="utf-8", errors="replace"` + 给子进程 `PYTHONIOENCODING=utf-8`。

### 5.4 答案末尾要监督 EOS

`labels` 里如果漏掉 EOS，模型学会"说答案"但学不会"停"，生成时会续写
"Answer: 42\nQuestion:" 之类的东西。`verify_stage3` 的 `dataset-shapes` 里有一条
断言专门盯这个（`lab[-1] != -100`）。

---

## 6. 作业

### HW3-1 · 梯度累积等价性（必做）

`homework/hw1_grad_accum.py`：证明 `bs=1, accum=4` 与 `bs=4, accum=1` 在
**同一批数据**上的梯度是等价的（相对误差 < 1e-3）。用 `_DummyLLM` + 纯 CPU 就能做，
不需要下模型。

提示：`loss = out["loss"] / accum` 然后 `backward()`；比较 `p.grad` 的 L2 距离。
注意 dropout：先 `model.eval()` 或 seed 固定。

### HW3-2 · 手算 LoRA 参数量（必做）

`homework/hw2_lora_params.py`：给定 `d_model=896`、`layers=24`、
`targets=[q,k,v,o]`、`r=8`，算出 LoRA 参数量，并与实测 **1.08M** 对比。

公式提示：每个投影 `2 * r * d_model`（A 和 B 两个矩阵），q/k/v/o 各一个；
别忘 Qwen2.5 的 `o_proj` 也是 `d_model × d_model`。

### HW3-3 · 学率探针（选做）

固定 30 步，跑 lr ∈ {5e-5, 2e-4, 1e-3} 三组对齐，比较 30 步后的 val_loss。
写一句结论：在这个规模上，lr 大 5 倍会怎样。

---

## 7. 验收标准

`python -m stage3_train.verify_stage3 --load` 必须 0 FAIL：

- `dataset-shapes`：像素在 [-1,1]、prompt 全 -100、EOS 被监督；
- `collate-padding`：pad 位 mask=0、label=-100；
- `prompt-contract`：训练与评测共用同一模板（改了要两边一起改）；
- `freeze-policy`：视觉塔 0 可训、可训参数只有连接器；
- `align-dry-run` / `sft-dry-run`：跑通并存下有限的权重；
- `align-ckpt`：真对齐的 connector.pt 存在且无 NaN；
- `lora-attach`：真权重上 LoRA 参数在 0.1M–50M 之间。

---

## 8. 延伸

- LoRA 原论文（Hu et al., 2021）：为什么低秩增量够用；
- LLaVA-1.5 的两阶段配方与本文的缩放差异；
- 梯度累积 + 梯度检查点的显存-速度权衡（S6 会实测）。
