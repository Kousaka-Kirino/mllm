# PicoVLM

从零训练一个 **5 亿参数**的多模态小模型，用受控实验回答一个问题：

> **小模型看图答错，是因为"看不见"，还是因为"不看图"？**

- 看不见 → **acuity**（字形在模型输入里只剩几像素）
- 不看图 → **prior**（模型学会了用训练分布的统计规律猜答案）

项目把这两件事做成两个可独立操纵的变量，用 2×2 实验拆开，再用三种干预
（分辨率课程 / 反事实增广 / 带冲突加权的 GRPO）去修，最后在 ChartQA/TextVQA
上看能不能迁移。

> 硬件基线：**RTX 4060 Laptop 8GB / 32GB RAM / Windows 11 / 无 WSL**。
> 所有数字都在这台机器上实测，见 [ENVIRONMENT.md](ENVIRONMENT.md)。
>
> 完整规划（16 周时间盒、每阶段验收标准、风险登记、简历映射、复试问答）见
> **[PicoVLM_项目规划.md](PicoVLM_项目规划.md)**。

---

## 一分钟跑通（不下载任何权重）

```bash
python -m stage0_env.check_env          # 环境体检 -> ENVIRONMENT.md + outputs/env_snapshot.json
python -m stage1_data.build             # 生成合成图表数据（默认小规模，约 1 分钟）
python -m stage1_data.verify_stage1     # 13 项数据验收
python -m stage2_model.verify_stage2    # 连接器/labels/前向（CPU）
python -m stage3_train.eval_suite --policy prior --split test
python -m stage4_diag.grid_2x2 --run-id prior-test
python -m stage8_system.run_all         # 一键验收矩阵
```

最后一条会把每个阶段的 PASS/FAIL/SKIP 汇总到 `outputs/acceptance.md`，
**任何一个 FAIL 都会让退出码变成 1**——可以直接挂 CI，也可以在复试现场跑。

## 目录

| 目录 | 内容 | 关键产出 |
|---|---|---|
| `picovlm/` | 跨阶段共享代码：契约、判分、统计、基线、预算算术、模型部件 | `schema.py` `answers.py` `stats.py` `model/` |
| `stage0_env/` | 环境体检与契约冻结 | `ENVIRONMENT.md`、`env_snapshot.json` |
| `stage1_data/` | 合成图表引擎：渲染、出题、冲突对、acuity 阶梯 | `data/synth/*.jsonl`、`datacard.json` |
| `stage2_model/` | ViT + 连接器 + LLM 组装与形状/掩码验收 | `picovlm/model/` |
| `stage3_train/` | 对齐训练、SFT、评测入口 | `outputs/eval/<run>/` |
| `stage4_diag/` | 2×2 诊断、VDR、遮挡忠实度 | `reports/grid_*.md` |
| `stage5_intervene/` | GRPO / RFT / 课程 / 反事实增广 | `reports/compare_*.md` |
| `stage6_budget/` | token 预算与 Pareto 前沿 | `reports/budget_*.md` |
| `stage7_memory/` | 视觉检索记忆（多页 RAG + 引用） | `reports/rag_*.md` |
| `stage8_system/` | CLI、演示、一键验收 | `outputs/acceptance.md` |
| `stage9_deliver/` | 简历条目、复试问答、一页纸 | 交付材料 |

## 三条不能破的规矩

1. **每个比例都带 95% 区间。** Wilson 区间，样本少也不虚。
2. **每个"变好了"都带配对比较。** 同一批题、胜/负/平 + McNemar p 值，不用两个总数相减。
3. **负结果照写。** 如果 GRPO 打不过 RFT、如果 acuity 干预没用，报告里必须有一节写清楚——这比"涨了 3 个点"值钱。

## 判分口径

严格（逐字）与宽松（数值 5% 容差 / 布尔同义词）两套同时记。
两者差值本身是证据：格式在漂 vs 能力在涨。

## 关键设计：acuity 是算出来的，不是"档位"

渲染画布固定 6 英寸、`dpi = 边长/6`，因此

```
字形像素 = 字号pt × font_scale × dpi/72 × (模型输入边长 / 渲染边长)
        = 9pt × font_scale × (384/432) = 8 × font_scale   （默认配置，与渲染边长无关）
```

**渲染分辨率本身不改变模型侧 acuity**——渲染 1024 再缩到 384，和直接渲染 384
在模型眼里等价。真正改变 acuity 的只有两件事：逻辑字号（`font_scale`）和
裁剪放大（S6 的预算机制）。这条恒等式有单测守着（`stage1_data.verify_stage1`）。

## 更多文档

- [PicoVLM_项目规划_v1.0.md](PicoVLM_项目规划_v1.0.md) —— 完整路线、阶段、时间盒、风险
- [RUNBOOK.md](RUNBOOK.md) —— 每个阶段怎么跑、跑多久、占多少显存
- [REPORT.md](REPORT.md) —— 实验报告骨架（随阶段推进填数字）
- [ENVIRONMENT.md](ENVIRONMENT.md) —— 自动生成，不要手改
