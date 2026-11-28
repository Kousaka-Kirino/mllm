# 项目状态（对照 2.0 规划逐条勾）

> 更新方式：每次 commit 后改这里。**状态只有三种**：✅ 有证据 / 🟡 部分 / ⬜ 未做。
> "有证据"指能指到一个可重跑的命令 + 一个落盘文件，不是"代码写了"。

最后更新：2026-11-26

## A. 工程骨架

| 项 | 状态 | 证据 |
|---|---|---|
| 十阶段目录 + 每阶段 README 讲解 | ✅ | `stage*/README.md`，`stage8_system.verify_stage8` 的 `stage-docs` |
| 每阶段 homework 题目 + 骨架 + 答案 | ✅ | `stage*/homework/{README.md,hw*.py,HWx_提示.md}` |
| 一键验收矩阵 | ✅ | `python -m stage8_system.verify --profile full` → **77 PASS / 0 FAIL / 0 SKIP**；`--profile core`（跳过 S2 真权重）→ 76 PASS / 0 FAIL / 1 SKIP；`outputs/acceptance.md` |
| pytest | ✅ | 29 passed |
| 配置外置（不改代码改参数） | ✅ | `configs/*.yaml`（14 份，支持 `base:` 继承），`verify_stage0` 的 `configs-loadable` |
| GitHub 分阶段推送 | ✅ | `origin/main` 上 S0→S9 各一个 commit |

### 各阶段交付物（逐目录核对，`--profile full` 实测）

| 阶段 | 目录 | 主入口 | README | 作业 | 答案 | verify |
|---|---|---|---|---|---|---|
| S0 | `stage0_env/` | `check` / `benchmark` | ✅ | 2 题 | `HW0_提示.md` | 10 PASS |
| S1 | `stage1_data/` | `build` / `validate` | ✅ | 2 题 | `HW1_提示.md` | 14 PASS |
| S2 | `stage2_model/` | `smoke` / `check_loss_mask` | ✅ | 2 题 | `HW2_提示.md` | 7 PASS |
| S3 | `stage3_train/` | `align` / `sft` | ✅ | 2 题 | `HW3_提示.md` | 7 PASS |
| S4 | `stage4_eval/` | `run` / `report` / `ood` | ✅ | 2 题 | `HW4_提示.md` | 9 PASS |
| S5 | `stage5_intervene/` | `train` / `build_rft` / `compare_arms` | ✅ | 2 题 | `HW5_提示.md` | 9 PASS |
| S6 | `stage6_ablation/` | `measure` / `pareto` | ✅ | 2 题 | `HW6_提示.md` | 4 PASS |
| S7 | `stage7_memory/` | `build_index` / `evaluate` | ✅ | 2 题 | `HW7_提示.md` | 6 PASS |
| S8 | `stage8_system/` | `verify` / `run_all` / `cli` | ✅ | 2 题 | `HW8_提示.md` | 6 PASS |
| S9 | `stage9_report/` | `collect` / `figures` | ✅ | 2 题 | `HW9_提示.md` | 5 PASS |

合计 **77 项检查全过**，与 `outputs/acceptance.md` 一致；README 里出现的 32 条
`python -m ...` 命令全部能解析到真实模块（含 `build_rft`、`verify --profile`）。

## B. 数据（S1）

| 项 | 状态 | 证据 |
|---|---|---|
| 渲染器（3 种图 / 字体缩放 / 噪声 / 干扰项） | ✅ | `stage1_data/render.py` |
| **实测**字形像素（不是按字号查表） | ✅ | `stage1_data/measure.py`，`reports/acuity_spotcheck/` |
| acuity 阈值**预注册**并冻结 | ✅ | `data/acuity_thresholds.json`（6.5 / 4.5 / 3.0 px） |
| 四类可操纵先验 + 冲突对 | ✅ | `stage1_data/conflict.py`（position / marginal / numeric / order） |
| family 划分（成对样本同 split） | ✅ | `build.py` 的断言 + `validate.py` 的 `no-family-leak` |
| pilot 数据 | ✅ | 636 family / 636 图 / 30MB，ρ 实测 0.7406 |
| full 数据（≥200 family/格） | ✅ | 已构建并验收：train 19998 / val 2496 / dev_test 1248（312/格）/ locked_test 5000（1250/格），28,742 图 / 约 700MB；`validate --profile full --data-dir data/full` 全 PASS |
| 人工核验记录 | 🟡 | 12 个裁剪图 + 表格（`reports/acuity_spotcheck/README.md`），样本量小 |

## C. 模型（S2）

| 项 | 状态 | 证据 |
|---|---|---|
| 三个连接器（576 / 144 / 64 token） | ✅ | `picovlm/model/connectors.py` |
| loss 掩码不变量（9 条） | ✅ | `stage2_model/check_loss_mask.py` |
| 真权重冒烟（589.5M / 576 vtokens / ~450ms） | ✅ | `stage2_model/smoke.py --load` |
| dtype 一致性（SigLIP fp32 vs Qwen bf16） | ✅ | `encode_images` 里统一 dtype |

## D. 训练（S3）

| 项 | 状态 | 证据 |
|---|---|---|
| connector 对齐（只训 2.30M） | ✅ | 60 步，loss 6.37→0.61，45s，`outputs/ckpt/align/` |
| LoRA SFT（r=8，1.08M） | ✅ | 237 步，loss→0.45，~4min，`outputs/ckpt/sft_pilot/` |
| EOS 监督（答完会停） | ✅ | `dataset.answer_with_eos`；修前格式分恒 0 |
| full 规模训练 | ✅ | 对齐 1,250 步（2,657s，peak 5849MiB）→ `outputs/ckpt/align_full/`；SFT 2,500 步 / 2 epoch（14,713s wall，peak 3770MiB）→ `outputs/ckpt/sft_full/`；dev_test（312/格）评测：0.965/0.580/0.939/0.558，prior_effect **+0.385** [+0.325,+0.444] |

## E. 评测（S4）

| 项 | 状态 | 证据 |
|---|---|---|
| 四条件（correct / mismatched / prior_only / image_only） | ✅ | `stage4_eval/predictor.py` |
| 2×2 四格 + prior/acuity 效应 + VDR | ✅ | `outputs/eval/sft_dev/grid.json` |
| Wilson CI / family 聚类 bootstrap / McNemar | ✅ | `analyse.py`、`reports/eval_*.md` |
| locked_test | ✅ | **已一次性开封**（5,000 条 × 3 条件，耗时 3,150s，留痕 `outputs/eval/final_locked/UNSEALED.json`）：0.947 / 0.577 / 0.942 / 0.584，prior_effect +0.370 [+0.340,+0.400]，acuity_effect +0.005（跨 0），换图 p<1e-300 |
| OOD 风格集 | ✅ | `stage1_data/ood.py` + `stage4_eval/ood.py`：同批 80 题换画风，0.512 → **0.000**，97.5% 空串（p=9.1e-13） |

## F. 干预（S5）

| 项 | 状态 | 证据 |
|---|---|---|
| 分辨率课程 | ✅ 跑满（负结果） | full 基座 1,500 步：0.933/0.551/0.942/0.529，vs 基线 p = 0.0067（但对照同向掉 0.036） |
| 反事实增广（措辞 / 数值） | ✅ 跑满（方向对、幅度不够） | 2,000→3.4k 条 1,500 步：0.923/**0.596**/0.913/**0.577**，vs 对照 p = 0.60 |
| 冲突加权 GRPO | 🟡 探针（负结果） | G=2、200 组：**190 组奖励全同**，只有 10 次有效更新；vs 对照 p = 0.25 |
| plain 对照（同一批样本只多训） | ✅ | `outputs/ckpt/plain_full`，0.929/0.577/0.946/0.538 |
| RFT 对照 | ✅ 代码 | `build_rft` + `rft.py` |
| **E2 反事实测量** | ✅ | `stage5_intervene/invariance.py`：措辞不变性 1.000、数值响应 0.000、仍答旧答案 0.833（n=6，探索性） |
| 三臂配对比较 | ✅ | `reports/RQ2_ARMS.md` + `reports/compare_arm_plain_vs_arm_*.md` |

## G. 预算与记忆（S6/S7）

| 项 | 状态 | 证据 |
|---|---|---|
| 三连接器的 token / 显存 / 延迟 | ✅ | mlp 576/2.30M/76ms/1340MiB；resampler 64/20.04M/40ms/1219MiB；pixelshuffle 144/2.75M/41ms/1207MiB |
| Pareto 前沿 + iso-token | ✅ | `stage6_ablation/pareto.py` |
| 页内召回（recall@k） | ✅ | **0.083** [0.023, 0.258]（失败，已作负结果） |
| 局部放大实验 | ✅ | salient 0.542 vs full 0.583，−0.042（不显著） |

## H. 报告与交付（S8/S9）

| 项 | 状态 | 证据 |
|---|---|---|
| RESULTS.json（所有数字的单一来源） | ✅ | `reports/RESULTS.json` |
| 自动报告（7 节） | ✅ | `reports/REPORT_2.0.md` |
| 声明 ↔ 证据对照（13 条） | ✅ | `stage9_report/CLAIMS.md` |
| 数字可追溯性检查 | ✅ | `numbers-traceable`：32 个三位小数全部可回溯 |
| 交付 CLI | ✅ | `python -m picovlm.cli --image ... --ask ...` |
| 简历条目 / 复试问答 / 一页纸架构 | ✅ | `stage9_report/deliver/0{1,2,3}_*.md` |

## 下一步（按优先级）

1. **full 数据** ✅ 已构建并验收（每格 312 / 1250 族）；
2. **full 规模训练** ✅ 对齐 1,250 步 + SFT 2,500 步（约 4 小时），dev_test 与 locked_test 双重复核；
3. **locked_test 开封** ✅ 2026-11-26 一次性开封，留痕在 `outputs/eval/final_locked/UNSEALED.json`；
4. **干预对照跑满** ✅ 2026-11-28 完成（3 小时预算版）：plain 对照 + 课程 + 反事实 + GRPO
   共用 `sft_full` 基座、同 2,000 条样本、各 1,500 步 → 见 [reports/RQ2_ARMS.md](reports/RQ2_ARMS.md)；
   结论是**三种干预都 n.s. 于"只多训一会儿"的对照**，GRPO 200 组里 190 组奖励相同；
5. （可选）训练种子方差 ✅ 已完成：seed 17 vs 23 → Δ +0.013、p = 1.0
   （[compare_sft_dev_vs_sft_dev_seed23.md](reports/compare_sft_dev_vs_sft_dev_seed23.md)）。

> 1–5 全部完成。RQ2 的结论是负的（干预无净收益），这是**跑满之后**的负结果，
> 可以照实写成"做了同算力对照，三种干预都没赢过对照"——比"设计了矩阵"更硬。
