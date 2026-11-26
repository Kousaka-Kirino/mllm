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
| full 规模训练 | 🟡 | 已用 pilot 模型在 full 的 dev_test（312/格）复现 RQ1；**尚未**在 full 上重训对齐 + SFT |

## E. 评测（S4）

| 项 | 状态 | 证据 |
|---|---|---|
| 四条件（correct / mismatched / prior_only / image_only） | ✅ | `stage4_eval/predictor.py` |
| 2×2 四格 + prior/acuity 效应 + VDR | ✅ | `outputs/eval/sft_dev/grid.json` |
| Wilson CI / family 聚类 bootstrap / McNemar | ✅ | `analyse.py`、`reports/eval_*.md` |
| locked_test | 🟡 | 生成好了（160 族），只在最后评测，尚未"开封" |
| OOD 风格集 | ✅ | `stage1_data/ood.py` + `stage4_eval/ood.py`：同批 80 题换画风，0.512 → **0.000**，97.5% 空串（p=9.1e-13） |

## F. 干预（S5）

| 项 | 状态 | 证据 |
|---|---|---|
| 分辨率课程 | ✅ 代码 / 🟡 效果 | `--method curriculum`，pilot 下未跑满 |
| 反事实增广（措辞 / 数值） | ✅ 代码 | 20 → 60 条，`counterfactual.py` |
| 冲突加权 GRPO | 🟡 | G=2–4 探针：**5/6 组优势恒 0**（负结果） |
| RFT 对照 | ✅ 代码 | `build_rft` + `rft.py` |
| **E2 反事实测量** | ✅ | `stage5_intervene/invariance.py`：措辞不变性 1.000、数值响应 0.000、仍答旧答案 0.833（n=6，探索性） |
| 三臂配对比较 | ✅ | `compare_arms.py` → `reports/compare_*.md` |

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
2. **full 规模训练**（对齐 + SFT 各数小时）→ 目前只有 pilot 模型在 full 测试集上的复现；
3. **干预对照跑满**（课程 / 反事实 / RFT / GRPO 各同算力）→ RQ2 的配对表；
4. **locked_test 开封一次** → 最终数字；
5. （可选）训练种子方差 ✅ 已完成：seed 17 vs 23 → Δ +0.013、p = 1.0
   （[compare_sft_dev_vs_sft_dev_seed23.md](reports/compare_sft_dev_vs_sft_dev_seed23.md)）。

> 第 1–4 步是"把 pilot 结论升级成主结论"的唯一路径；
> 不做也可以交付（诚实的 pilot 结论 + 完整骨架），但简历上要写清"pilot 规模"。
