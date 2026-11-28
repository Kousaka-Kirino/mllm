# RQ2 三臂对比 · 3 小时执行计划

> 起点：`outputs/ckpt/sft_full`（full 规模重训，locked_test 主结果 0.947/0.577/0.942/0.584）
> 环境：`PICOVLM_DATA=data/full`，单卡 4060 8GB，`HF_HUB_OFFLINE=1`

## 为什么"分开训"能塞进 3 小时

三臂不必各训一个 epoch。它们**共享同一个基座、同一批样本、同一个步数预算**，
只变"喂什么数据"这一个因子——这正是受控实验的要求，也让每臂压到 20–30 分钟。

| 臂 | 数据/信号 | 步数 | 预计 |
|---|---|---|---|
| A 课程 | 前 40% 步数只给易/中样本，之后混合 | 1,500 | 20–30 min |
| B 反事实 | 原样本 + 措辞变体（金标不变）+ 数值变体（金标变） | 1,500 | 25–35 min |
| C 冲突加权 GRPO | 在线采样，冲突格奖励 ×2 | 200（G=2） | 15–25 min |
| 基线 | `sft_full` 本身（已有 dev_test 评测） | — | 0 |

三臂用同一个 `--shuffle-seed 17` 和 `--limit 2000`，样本集合完全相同。

## 时间表（2026-11-28）

| 时段 | 内容 | 产出 |
|---|---|---|
| 17:05–17:15 | 给 `stage5_intervene/train.py` 加 `--init/--config/--shuffle-seed`，跑 10 步冒烟 | 冒烟日志 |
| 17:15–17:45 | 臂 A 课程，1,500 步 | `outputs/ckpt/curriculum_full` |
| 17:45–18:20 | 臂 B 反事实，1,500 步 | `outputs/ckpt/counterfactual_full` |
| 18:20–18:45 | 臂 C GRPO，200 步 G=2 | `outputs/ckpt/grpo_full` |
| 18:45–19:20 | 三臂 + 基线在 dev_test 上评测（correct/mismatched 两条件） | 4 个 run |
| 19:20–19:50 | `compare_arms` 配对比较 → 报告 → commit & push | `reports/rq2_*.md` |

## 验收口径（和主实验一致）

- 主指标：dev_test 2×2 四格准确率 + `prior_effect`（family 聚类 bootstrap CI）。
- 臂间比较：同实例配对，McNemar 精确检验，报不一致对数。
- 任何一臂打不过基线，就作为负结果写进报告——**不重跑、不挑种子**。

## 降级方案

- 若某臂超时：把步数砍到 1,000（其它不变），在报告里注明预算不同。
- 若 GRPO 冒烟显示整组同分率过高（>80%）：改跑 `--group-size 4`，或只报探针结果。
- 若显存吃紧：`--batch-size 1` 已是默认；关掉桌面程序可多出 ~1GB。
