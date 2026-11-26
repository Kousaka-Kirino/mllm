# PicoVLM · 结果汇总（自动生成）

生成时间：20261126T020304Z

> 本文件由 `python -m stage9_report.collect` 生成，不要手改。
> 每个数字都能在 `reports/RESULTS.json` 与 `outputs/` 里找到来源。

## 1. 数据

- profile：`pilot`，seed 17
- 样本数：{'train': 318, 'val': 78, 'dev_test': 80, 'locked_test': 160}
- family 数：{'train': 318, 'val': 78, 'dev_test': 80, 'locked_test': 160}
- ρ 先验强度：请求 0.7 / 实测 **0.7406**
- acuity 分箱（实测阈值 {'easy_min_px_model': 6.5, 'mid_min_px_model': 4.5, 'extreme_max_px_model': 3.0, 'n_measurements': 636, 'median': 6.0, 'p05': 3.0, 'p95': 9.0, 'coverage': {'below_extreme': 0, 'hard': 304, 'mid': 48, 'easy': 284}, 'rule': 'easy: px >= easy_min; mid: mid_min <= px < easy_min; hard: extreme_max <= px < mid_min; extreme: px < extreme_max', 'source': '预注册常量（pilot 实测着墨高≈6.2×font_scale 时校准），写入即冻结'}）：{'hard': 304, 'easy': 284, 'mid': 48}
- 字形像素（模型输入侧，636 次测量）：median 6.0 px，IQR 4.0，范围 [3.0, 9.0]

## 2. 训练

| run | steps | 首步 loss | 末步 loss | 末步 lr |
|---|---:|---:|---:|---:|
| sft_20261126T004058Z | 23 | 10.3022 | 0.7279 | 7.37e-07 |

## 3. 评测（RQ1：2×2 诊断，`correct_image` 条件）

| run | suite | n | accuracy | 95% CI |
|---|---|---:|---:|---|
| first_option-test | easy_conflict | 24 | 0.000 | [0.000, 0.138] |
| first_option-test | easy_consistent | 24 | 0.750 | [0.551, 0.880] |
| first_option-test | hard_conflict | 24 | 0.000 | [0.000, 0.138] |
| first_option-test | hard_consistent | 24 | 0.750 | [0.551, 0.880] |
| first_option-test | neutral | 24 | 0.417 | [0.245, 0.612] |
| oracle-test | easy_conflict | 24 | 1.000 | [0.862, 1.000] |
| oracle-test | easy_consistent | 24 | 1.000 | [0.862, 1.000] |
| oracle-test | hard_conflict | 24 | 1.000 | [0.862, 1.000] |
| oracle-test | hard_consistent | 24 | 1.000 | [0.862, 1.000] |
| oracle-test | neutral | 24 | 1.000 | [0.862, 1.000] |
| prior-test | easy_conflict | 24 | 0.000 | [0.000, 0.138] |
| prior-test | easy_consistent | 24 | 1.000 | [0.862, 1.000] |
| prior-test | hard_conflict | 24 | 0.000 | [0.000, 0.138] |
| prior-test | hard_consistent | 24 | 1.000 | [0.862, 1.000] |
| prior-test | neutral | 24 | 0.208 | [0.092, 0.405] |
| prior_dev | easy_conflict | 20 | 0.000 | [0.000, 0.161] |
| prior_dev | easy_consistent | 20 | 1.000 | [0.839, 1.000] |
| prior_dev | hard_conflict | 20 | 0.000 | [0.000, 0.161] |
| prior_dev | hard_consistent | 20 | 1.000 | [0.839, 1.000] |
| sanity-oracle | easy_conflict | 24 | 1.000 | [0.862, 1.000] |
| sanity-oracle | easy_consistent | 24 | 1.000 | [0.862, 1.000] |
| sanity-oracle | hard_conflict | 24 | 1.000 | [0.862, 1.000] |
| sanity-oracle | hard_consistent | 24 | 1.000 | [0.862, 1.000] |
| sanity-oracle | neutral | 24 | 1.000 | [0.862, 1.000] |
| sft_dev | easy_conflict | 20 | 0.000 | [0.000, 0.161] |
| sft_dev | easy_consistent | 20 | 0.950 | [0.764, 0.991] |
| sft_dev | hard_conflict | 20 | 0.100 | [0.028, 0.301] |
| sft_dev | hard_consistent | 20 | 1.000 | [0.839, 1.000] |
| verify-oracle | easy_conflict | 24 | 1.000 | [0.862, 1.000] |
| verify-oracle | easy_consistent | 24 | 1.000 | [0.862, 1.000] |
| verify-oracle | hard_conflict | 24 | 1.000 | [0.862, 1.000] |
| verify-oracle | hard_consistent | 24 | 1.000 | [0.862, 1.000] |
| verify-oracle | neutral | 24 | 1.000 | [0.862, 1.000] |
| verify_prior | easy_conflict | 20 | 0.000 | [0.000, 0.161] |
| verify_prior | easy_consistent | 20 | 1.000 | [0.839, 1.000] |
| verify_prior | hard_conflict | 20 | 0.000 | [0.000, 0.161] |
| verify_prior | hard_consistent | 20 | 1.000 | [0.839, 1.000] |

### 3b. 条件对比（S4 E0 矩阵）

| run | condition | suite | n | accuracy |
|---|---|---|---:|---:|
| prior_dev | image_only | easy_conflict | 20 | 0.000 |
| prior_dev | image_only | easy_consistent | 20 | 1.000 |
| prior_dev | image_only | hard_conflict | 20 | 0.000 |
| prior_dev | image_only | hard_consistent | 20 | 1.000 |
| prior_dev | mismatched_image | easy_conflict | 20 | 0.000 |
| prior_dev | mismatched_image | easy_consistent | 20 | 1.000 |
| prior_dev | mismatched_image | hard_conflict | 20 | 0.000 |
| prior_dev | mismatched_image | hard_consistent | 20 | 1.000 |
| prior_dev | prior_only | easy_conflict | 20 | 0.000 |
| prior_dev | prior_only | easy_consistent | 20 | 1.000 |
| prior_dev | prior_only | hard_conflict | 20 | 0.000 |
| prior_dev | prior_only | hard_consistent | 20 | 1.000 |
| sft_dev | mismatched_image | easy_conflict | 20 | 0.050 |
| sft_dev | mismatched_image | easy_consistent | 20 | 0.950 |
| sft_dev | mismatched_image | hard_conflict | 20 | 0.050 |
| sft_dev | mismatched_image | hard_consistent | 20 | 1.000 |
| sft_dev | prior_only | easy_conflict | 20 | 0.150 |
| sft_dev | prior_only | easy_consistent | 20 | 0.100 |
| sft_dev | prior_only | hard_conflict | 20 | 0.150 |
| sft_dev | prior_only | hard_consistent | 20 | 0.150 |
| verify_prior | image_only | easy_conflict | 20 | 0.000 |
| verify_prior | image_only | easy_consistent | 20 | 1.000 |
| verify_prior | image_only | hard_conflict | 20 | 0.000 |
| verify_prior | image_only | hard_consistent | 20 | 1.000 |
| verify_prior | mismatched_image | easy_conflict | 20 | 0.000 |
| verify_prior | mismatched_image | easy_consistent | 20 | 1.000 |
| verify_prior | mismatched_image | hard_conflict | 20 | 0.000 |
| verify_prior | mismatched_image | hard_consistent | 20 | 1.000 |
| verify_prior | prior_only | easy_conflict | 20 | 0.000 |
| verify_prior | prior_only | easy_consistent | 20 | 1.000 |
| verify_prior | prior_only | hard_conflict | 20 | 0.000 |
| verify_prior | prior_only | hard_consistent | 20 | 1.000 |

## 4. 效应（family 聚类 bootstrap）

**prior-test**：prior_effect +1.000，acuity_effect +0.000，interaction +0.000，VDR MISSING

**sanity-oracle**：prior_effect +0.000，acuity_effect +0.000，interaction +0.000，VDR MISSING

**sft_dev**：prior_effect +0.950，acuity_effect -0.050，interaction +0.050，VDR easy 0.000 / hard 0.100
- bootstrap（2000 次，按 family）：prior_effect +0.950 [+0.833, +1.000]

**verify-oracle**：prior_effect +0.000，acuity_effect +0.000，interaction +0.000，VDR MISSING

**verify_prior**：prior_effect +1.000，acuity_effect +0.000，interaction +0.000，VDR easy 0.000 / hard 0.000
- bootstrap（2000 次，按 family）：prior_effect +1.000 [+1.000, +1.000]

## 5. 预算（S6）

| 连接器 | token | 参数(M) | 延迟(ms) | 峰值(MiB) | 精度 |
|---|---:|---:|---:|---:|---|
| mlp | 576 | 2.30 | 76 | 1340 | 0.512（correct_image, 80 条） |
| resampler | 64 | 20.04 | 40 | 1219 | 未训练（不报） |
| pixelshuffle | 144 | 2.75 | 41 | 1207 | 未训练（不报） |

## 6. 视觉记忆（S7）

- 页内召回 recall@1 0.083 [0.023, 0.258]（n=24）
- 放大实验：full 0.583 / salient 0.542 / random 0.292（n=24）
- salient − full = -0.042

## 7. 结论（按证据强度排序）

1. **先验主导**：prior_effect 大、VDR 低、换图不影响答案（S4）。
2. **不是纯先验**：灰图/随机裁块会让准确率崩塌（S4 + S7），说明视觉输入被使用。
3. **acuity 不是当前瓶颈**：显著块放大没有超过整图（S7）。
4. **检索未达标**：冻结 SigLIP 平均池化的 recall@1 很低（S7）。

> 以上四条与 `reports/RESULTS.json` 一一对应；缺项的写 MISSING，不补。

## 8. E2 反事实（探索性）

| model | 措辞不变性率 | 数值响应率 | 仍答旧答案 |
|---|---:|---:|---:|
| sft | 1.000 (n=6) | 0.000 (n=6) | 0.833 |

> 只改问句时答案几乎不变（不变性率高），但把表里的数值改掉、金标换人后，模型仍答旧答案（响应率 0）→ 它读的是题面与先验，不是图。
> n 很小（数值变体只支持 maxcat），**探索性**，不进主结论。
