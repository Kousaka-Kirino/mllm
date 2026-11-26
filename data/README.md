# 数据：不进仓库，可重建

```bash
# pilot（默认，几千张图以内，用来验证管线）
python -m stage1_data.build --profile pilot --seed 17

# full（主分析用，全部写到 data/full/ 下，不碰 pilot）
python -m stage1_data.build --profile full --seed 17 --out data/full
```

`--out` 决定**所有**产物落在哪：`<out>/synth/*.jsonl`、`<out>/synth/images/*.png`、
`<out>/datacard.json`、`<out>/hashes.json`。只有 `data/acuity_thresholds.json`
是全局共享的——full 构建默认复用它（冻结阈值），否则分箱会随数据分布漂移，
跨规模结果不可比；确需重算时加 `--recalibrate`。

## pilot 产物（`data/`）

| 路径 | 内容 | 大小 |
|---|---|---|
| `data/synth/{train,val,dev_test,locked_test}.jsonl` | 样本清单（含 family_id、meta.table、实测字形像素） | train 318 / val 78 / dev_test 80 / locked_test 160 |
| `data/synth/images/` | 渲染出的 PNG（按内容 sha1 去重） | 636 张，约 16 MB |
| `data/datacard.json` | 数据卡：计数、family 数、ρ 实测、像素分布 | 约 20 KB |
| `data/hashes.json` | 产物 sha256 清单（复现校验用） | 约 5 KB |

## full 产物（`data/full/`）

| 路径 | 内容 |
|---|---|
| `data/full/synth/*.jsonl` | train 19998 / val 2496 / dev_test 1248 / locked_test 5000 |
| `data/full/synth/images/` | 约 28.7k 张，约 780 MB |
| `data/full/datacard.json` | 计数、ρ 实测（0.7002）、像素分布 |

> full 与 pilot 用同一套种子与阈值，**唯一样本量不同**；两者的 `family_id` 可能重合，
> 所以不要把 pilot 训练出来的模型直接拿到 full 的 locked_test 上评估。

这些文件被 `.gitignore` 排除，因为**它们由代码 + 种子确定性地生成**。
`python -m stage1_data.validate` 会重算哈希并与 `data/hashes.json` 比对，
所以"数据不在仓库里"不等于"数据不可复现"。

## 两条设计约定（写论文/答辩时必须讲清）

1. **2×2 是"组间"设计**：易/难 × 一致/冲突 四格用的是**不同的图**（每格各自的族）。
   所以四格之间只能做组间比较（Wilson CI、按 family 聚类的 bootstrap），
   **不能**对四格做配对检验；配对只发生在"同一样本的不同条件"上
   （correct_image vs mismatched_image、seed 17 vs seed 23）。
   `validate` 的 `pair-integrity` 检查就是在钉这条：一个 `pair_id` 只留一条、且不跨 split。
2. **先验标签必须能从数据重算**：`meta.prior_role` 由 `answer == prior_answer` 直接算出，
   不信生成器自报。full 构建时抓到过 6 条"标了 consistent 但答案与先验不等"的 marginal 样本，
   现在这类样本会被自动改标为 conflict（构建日志会打印修正条数）。
   判分、分格、报告都以重算结果为准。
