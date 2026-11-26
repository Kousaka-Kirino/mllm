# 简历条目 ↔ 可核查数字

> 用法：简历里每写一句话，右边必须能指到一个文件。复试时被追问"这个数怎么来的"，
> 直接打开对应文件。**没有来源的句子不要写**。

| # | 简历/答辩声明 | 数字 | 来源文件 |
|---|---|---|---|
| 1 | 自建可控图表数据集，family 级无泄漏划分 | 636 样本 / 636 family（train 318 / val 78 / dev_test 80 / locked_test 160） | `data/datacard.json` |
| 2 | 先验强度可调（ρ），实测偏差 < 0.05 | 请求 0.7 / 实测 0.7406 | `data/datacard.json` |
| 3 | acuity 用**实测**字形像素而非解析式，阈值预注册 | 中位 6.0 px，IQR 4.0，范围 [3.0, 9.0] | `data/acuity_thresholds.json`、`data/datacard.json` |
| 4 | 0.5B 级多模态模型：SigLIP-base + Qwen2.5-0.5B + 连接器 | 总 589.5M，连接器 2.30M，视觉 token 576 | `stage2_model/README.md`、`outputs/bench/connectors_*.json` |
| 5 | 单次前向成本可复现 | 76 ms，峰值 1340 MiB（batch=1, seq 591） | `outputs/bench/connectors_*.json` |
| 6 | connector 对齐 + LoRA SFT 两阶段训练 | 对齐 60 步 loss 6.37→0.61；SFT 237 步 loss→0.45 | `outputs/train/*/log.jsonl` |
| 7 | **RQ1 主结果**：先验一致时接近满分，冲突格几乎全灭 | 0.950 / 0.000 / 1.000 / 0.100（易一致/易冲突/难一致/难冲突） | `reports/eval_sft_dev.md`、`outputs/eval/sft_dev/records.json` |
| 8 | 先验效应量（family 聚类 bootstrap 2000 次） | +0.950，95% CI [+0.833, +1.000] | `outputs/eval/sft_dev/grid.json` |
| 9 | 换图不改变答案 → 不看图（不是看不见） | correct 0.512 vs mismatched 0.512；80 对各 1 对翻转，McNemar p = 1.0 | `outputs/eval/sft_dev/predictions.jsonl` |
| 10 | 但视觉输入仍被使用（不是纯先验） | prior_only 掉到 0.138；随机裁块 0.292 | `outputs/eval/sft_dev/records.json`、`reports/memory_*.md` |
| 11 | 放大"字最密"的块 → 不提升，说明 acuity 不是瓶颈 | full 0.583 / salient 0.542（Δ −0.042，不显著） | `reports/memory_*.md` |
| 12 | 页内检索（冻结 SigLIP + 平均池化）不足 | recall@1 0.083 [0.023, 0.258] | `reports/memory_*.md` |
| 13 | GRPO 组内优势在 pilot 上信号稀疏（诚实负结果） | 6 组中 5 组整组同分 → 优势为 0 | `outputs/grpo/*.json` |
| 14 | E2 反事实：改问句不变、改数值不响应（探索性，n=6） | 不变性 1.000 [0.610,1.000]；响应 0.000；仍答旧答案 0.833 | `outputs/eval/invariance_sft.json` |
| 15 | E5 画风迁移：同批题换画风后崩溃（探索性） | 0.512 → 0.000，97.5% 输出空串，McNemar p=9.1e-13 | `reports/ood_sft_ood.md`、`outputs/eval/sft_ood/ood.json` |
| 16 | 训练种子方差远小于效应（排除"碰巧"） | seed 17 vs 23：Δ +0.013，1 对 0 翻转，McNemar p = 1.0 | `reports/compare_sft_dev_vs_sft_dev_seed23.md` |
| 17 | 大测试集复现（15.6×，每格 312 族） | 0.968 / 0.029 / 0.952 / 0.051；prior_effect +0.939 [0.910,0.966]；acuity_effect +0.016 [−0.014,0.048] | `outputs/eval/sft_full_dev/grid.json`、`reports/eval_sft_full_dev.md` |
| 18 | 一键验收 | 见 `outputs/acceptance.md` 合计行 | `outputs/acceptance.md` |

## 三条写简历的规矩

1. **只写可复现的数**：每个数都能用仓库里的命令重跑出来（数据由种子重建）。
2. **负结果照写**：第 11、12、13、15 条比"我们超过了 baseline"更能说明你会做实验。
3. **区分"我做的"和"借用的"**：SigLIP/Qwen 是预训练权重，你训的是连接器 + LoRA；
   写"可训练参数 3.38M"而不是"我训了 0.5B"（3.38M = LoRA 1.08M + connector 2.30M，
   SFT 阶段实际更新的参数量；总参 589.5M 里 586M 是冻结的）。
