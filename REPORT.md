# PicoVLM 实验报告（骨架）

> 填数字的规矩：**先写结论，再写区间，最后写样本量**。
> 没有跑到的格子写 `TODO`，不要写"预计"。

## 0. 摘要

一句话：TODO（用 2×2 诊断回答"看不见还是不看图"，并给出三种干预的配对结果）。

## 1. 环境

见 [ENVIRONMENT.md](ENVIRONMENT.md)（自动生成）。
关键：RTX 4060 Laptop 8188MiB、实际可用约 5.5–6.5GB、Python 3.11、torch 2.11+cu128。

## 2. 数据

| 集合 | 条数 | 说明 |
|---|---|---|
| train | TODO | 先验强度 ρ=0.7 的偏斜训练集 |
| dev | TODO | 平衡集，调参用 |
| test | TODO | 2×2 四格 × 2 种先验，冲突对成对出现 |

## 3. 训练

| 阶段 | 可训练参数 | 步数 | 时长 | 峰值显存 | 最终 loss |
|---|---|---|---|---|---|
| 对齐 | TODO | TODO | TODO | TODO | TODO |
| SFT | TODO | TODO | TODO | TODO | TODO |

## 4. RQ1：2×2 诊断

|  | 先验一致 | 先验冲突 |
|---|---|---|
| acuity 易 | TODO | TODO |
| acuity 难 | TODO | TODO |

- 先验效应：TODO
- acuity 效应：TODO
- VDR：TODO（按先验类型拆）

## 5. RQ2：干预

| 臂 | 准确率 | 95% CI | vs SFT（胜/负/平） | p |
|---|---|---|---|---|
| SFT | TODO | TODO | — | — |
| + 分辨率课程 | TODO | TODO | TODO | TODO |
| + 反事实增广 | TODO | TODO | TODO | TODO |
| + 冲突加权 GRPO | TODO | TODO | TODO | TODO |
| RFT（同算力对照） | TODO | TODO | TODO | TODO |

## 6. RQ3：迁移

ChartQA / TextVQA / OOD 风格：TODO。

## 7. 负结果

TODO（必须写：哪个干预没用、为什么、有没有可能是测量问题）。

## 8. 局限

TODO（合成数据的外部效度、单卡单种子、判分口径的边界）。
