# PicoVLM

一个 **5.89 亿参数**的多模态小模型（SigLIP-base + Qwen2.5-0.5B，**只用 8GB 笔记本显卡**训），
以及围绕它的一个受控实验：

> **小模型看图答错，是因为"看不见"，还是因为"不看图"？**

- 看不见 → **acuity**（字形进到模型输入时只剩几个像素）
- 不看图 → **prior**（模型从训练分布里学到的答题偏好）

项目把两者做成可独立操纵的变量，用 **2×2** 设计拆开，再用三种干预去修，
最后诚实地报告"哪些起了作用、哪些没起作用"。

> 硬件：**RTX 4060 Laptop 8GB / 32GB RAM / Windows 11 / 无 WSL**。
> 所有数字都在这一台机器上实测，环境快照见 [ENVIRONMENT.md](ENVIRONMENT.md)，
> 结论与证据的对照见 [stage9_report/CLAIMS.md](stage9_report/CLAIMS.md)。

---

## 三条结论（先看这个）

在 pilot 规模（dev_test 每格 20 族）上，SFT 之后：

| 条件 | 易·一致 | 易·冲突 | 难·一致 | 难·冲突 |
|---|---:|---:|---:|---:|
| 正确图 | **0.950** | **0.000** | **1.000** | 0.100 |
| 错配图（同一张图换成别的样本） | 0.950 | 0.050 | 1.000 | 0.050 |
| 只有问题（灰图） | 0.100 | 0.150 | 0.150 | 0.150 |

1. **换掉图，答案几乎不动。** 80 对里只有 1 对各不相同（正确图独有 1、错配图独有 1，
   McNemar 精确检验 **p = 1.0**）→ 主导信号是先验，不是图像。
2. **冲突格全灭**（0.00 / 0.10），而盲基线在同格恰好是 1.00 / 0.00 —— 模型学会了训练集里的统计规律。
3. **但视觉不是完全没用**：把图换成灰图，四格全部塌到 0.1–0.15。所以它"看了"，只是没有"用来看答案"。

两条独立的旁证（详见 [reports/REPORT_2.0.md](reports/REPORT_2.0.md)）：
- 视觉记忆的页内召回 recall@1 = **0.083**（冻结 SigLIP + 平均池化，失败）
- 局部放大实验：salient 0.542 vs full 0.583，**−0.042**（不显著）→ **acuity 不是当前瓶颈**

一条更狠的探索性证据（E5，[reports/ood_sft_ood.md](reports/ood_sft_ood.md)）：
把**同一批题**的图换个画风（衬线字体 / 全框 / 双轴网格 / 另一套配色）重画，
准确率从 0.512 掉到 **0.000**，97.5% 的输出是**空串**（McNemar p = 9.1e-13）。
237 步的 pilot 学到的是训练分布的表面统计——这条比 2×2 更能说明它有多脆。

这些结论不是**种子噪声**：只换训练种子（17 → 23）重训重评，四格与总体几乎不动
（Δ +0.013，McNemar p = 1.0，[compare_sft_dev_vs_sft_dev_seed23.md](reports/compare_sft_dev_vs_sft_dev_seed23.md)）。

同一套模型换到**大 15.6 倍的 held-out 测试集**（`data/full`，每格 312 族）上复现：

| 格 | acc | Wilson 95% |
|---|---:|---|
| 易·一致 | 0.968 | [0.942, 0.982] |
| 易·冲突 | 0.029 | [0.015, 0.054] |
| 难·一致 | 0.952 | [0.922, 0.971] |
| 难·冲突 | 0.051 | [0.032, 0.082] |

先验效应 **+0.939**（bootstrap [+0.910, +0.966]），acuity 效应 **+0.016**（[−0.014, +0.048]，跨 0）。
结论不变，CI 窄一个量级 → RQ1 不是 80 条样本的偶然。

---

## 一分钟跑通（CPU 即可，不下载权重）

```bash
python -m stage1_data.build --profile pilot --seed 17   # 生成数据（约 1 分钟，636 族）
python -m stage1_data.verify_stage1                     # 14 项数据验收
python -m stage2_model.verify_stage2                    # 连接器 / loss 掩码 / 前向
python -m stage4_eval.run --model prior --suite dev_test # 盲基线答题（0 显存）
python -m stage9_report.collect                         # 汇总 -> reports/REPORT_2.0.md
python -m stage9_report.figures                         # 出图 -> reports/figs/*.png
python -m stage8_system.run_all                         # 全阶段验收矩阵（也可用 verify --profile core）
```

最后一条把每个阶段的 PASS/FAIL/SKIP 汇总到 `outputs/acceptance.md`，
**任何 FAIL 都会让退出码变成 1**，可以直接挂 CI。

需要 GPU 的部分（约 3.5GB 峰值）：

```bash
python -m stage3_train.align --max-steps 60            # 只训 connector，~45s
python -m stage3_train.sft --max-steps 237             # LoRA SFT，~4min
python -m stage4_eval.run --model sft --suite dev_test  # 2×2 评测
python -m picovlm.cli --image data/synth/images/x.png --ask "Which category has the highest value?"
```

---

## 目录（每个 stage = 代码 + 讲解 + 作业 + 答案 + 验收）

| 目录 | 内容 | 关键产出 |
|---|---|---|
| `picovlm/` | 跨阶段共享：契约、判分、统计、连接器、模型组装、CLI | `schema.py` `answers.py` `stats.py` `model/` `cli.py` |
| `stage0_env/` | 环境体检 + 显存/延迟基准 | `ENVIRONMENT.md`、`outputs/bench/` |
| `stage1_data/` | 合成图表引擎：渲染、出题、冲突对、**实测** acuity | `data/synth/*.jsonl`、`datacard.json`、`acuity_thresholds.json` |
| `stage2_model/` | ViT + connector + LLM 组装，loss 掩码不变量 | `stage2_model/check_loss_mask.py` |
| `stage3_train/` | connector 对齐 → LoRA SFT | `outputs/ckpt/{align,sft_pilot}/` |
| `stage4_eval/` | 2×2 评测、按族 bootstrap、McNemar | `outputs/eval/<run>/`、`reports/eval_*.md` |
| `stage5_intervene/` | 课程 / 反事实增广 / GRPO / RFT | `reports/compare_*.md` |
| `stage6_ablation/` | 连接器与 token 预算的显存/延迟权衡 | `reports/budget_*.md` |
| `stage7_memory/` | 视觉检索记忆（多页 RAG + 引用） | `reports/memory_*.md` |
| `stage8_system/` | 一键验收矩阵 | `outputs/acceptance.md` |
| `stage9_report/` | 结果汇总、报告、声明↔证据对照 | `reports/RESULTS.json`、`REPORT_2.0.md`、`CLAIMS.md` |

每个 stage 目录下都有：
- `README.md` —— 这一阶段解决什么问题、代码怎么读、坑在哪
- `homework/README.md` —— 作业题目与提交要求
- `homework/HWx_提示.md` —— 参考答案与验收标准
- `homework/hw*.py` —— 带 TODO 的骨架，`python homework/hwX.py` 自查

---

## 四条不能破的规矩

1. **每个比例都带 95% 区间。** Wilson 区间；样本 20 条也照给，不虚。
2. **每个"变好了"都是同实例配对比较。** 胜/负/平 + McNemar 精确检验，不用两个总数相减；
   效应量的置信区间按 **family 整簇 bootstrap**，因为同一张图的多个问题是相关的。
3. **acuity 是量出来的，不是查表查出来的。** 渲染后按文字 bbox 裁块、数着墨行高，
   记录"原图 / 模型输入"两个尺度；档位按模型输入侧的实测像素预注册分箱。
4. **负结果照写。** GRPO 在 8GB 上组内几乎无梯度、视觉召回失败、放大没用 ——
   这些都在报告里单独成节，且和正结果并列。

## 判分口径

严格（逐字相等）与宽松（数值 5% 容差、布尔同义词、金标作为独立 token 出现）两套同时记。
两者的差值本身是证据：**格式在漂** vs **能力在涨**。

---

## 一句话架构

```
图 1152²  ──resize──▶ 384²  ──SigLIP(冻结, 93M)──▶ 576 patch
                                                      │
                                          connector(2.3M, 只训它)
                                                      ▼
                                     Qwen2.5-0.5B(494M) + LoRA(r=8, 1.08M)
                                                      ▼
                                               "A"（≤8 token）
```

- 可训练参数：connector **2.30M** + LoRA **1.08M** ≈ **3.4M**（总参数 589.5M）
- 单图单问：视觉 576 token + 文本 ~15 token；延迟 ~450ms；峰值显存 **~2.1GB**（前向）/ **3.8GB**（SFT）
- 三个连接器都实现了：`mlp`(576 token) / `resampler`(64) / `pixelshuffle`(144)，用于 iso-token 对比

---

## 文档地图

| 想知道 | 看 |
|---|---|
| 项目怎么一步步做、时间怎么排 | [PicoVLM 项目规划 2.0.md](PicoVLM%20项目规划%202.0.md) |
| 每个命令干什么、要多久、吃多少显存 | [RUNBOOK.md](RUNBOOK.md) |
| 实验结论与数字 | [reports/REPORT_2.0.md](reports/REPORT_2.0.md) |
| 简历条目 ↔ 证据 | [stage9_report/CLAIMS.md](stage9_report/CLAIMS.md) |
| 复试问答 | [stage9_report/deliver/](stage9_report/deliver/) |
| 这台机器的真实环境 | [ENVIRONMENT.md](ENVIRONMENT.md) |

> 历史文档 `PicoVLM_项目规划.md` 是 1.0 版（按"从零训 5 亿参数"设计），
> 已被 2.0 取代，仅作演进记录保留；里面的数字不要引用。
