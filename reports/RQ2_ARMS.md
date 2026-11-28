# RQ2 · 三种干预的同算力对比（探索性，3 小时预算版）

> 生成时间：2026-11-28。所有数字可用文末命令重跑；配置见 `PLAN_RQ2_3H.md`。

## 1. 设计（受控在哪）

| 因子 | 取值 | 控制方式 |
|---|---|---|
| 基座 | `outputs/ckpt/sft_full`（full 规模重训） | 四臂共用同一份连接器 + LoRA |
| 样本 | train split 前 2,000 条，`--shuffle-seed 17` | 四臂完全同一批样本 |
| 步数 | 1,500 步（bs=1），GRPO 200 组（G=2） | 每臂同预算 |
| 学习率 | 5e-5（GRPO 1e-5） | 与 SFT 阶段一致 |
| 评测 | dev_test 1,248 条 × 2 条件 | 同实例配对 McNemar |

四臂分别是：**plain**（对照，只多训不多做）、**curriculum**（前 40% 步只看易/中样本）、
**counterfactual**（原样本 + 措辞变体 + 数值变体，约 3.4k 条，1,500 步）、
**grpo**（在线采样，冲突格奖励 ×2）。

> 没有 plain 这一臂，前面三臂的差异就无法归因到"干预"还是"多训了 1,500 步"。

## 2. 结果（dev_test，312 族/格，correct_image）

| run | 易·一致 | 易·冲突 | 难·一致 | 难·冲突 |
|---|---:|---:|---:|---:|
| 基线 `sft_fulltrain_dev` | 0.965 | 0.580 | 0.939 | 0.558 |
| plain（对照） | 0.929 | 0.577 | 0.946 | 0.538 |
| curriculum | 0.933 | 0.551 | 0.942 | 0.529 |
| counterfactual | 0.923 | 0.596 | 0.913 | 0.577 |
| grpo（10 次有效更新） | 0.965 | 0.571 | 0.939 | 0.554 |

## 3. 配对比较（同实例，McNemar 精确检验）

| 对比 | Δ(A−B) | 只有 A 对 | 只有 B 对 | p |
|---|---:|---:|---:|---:|
| 基线 vs curriculum | +0.022 | 60 | 33 | **0.0067** |
| 基线 vs counterfactual | +0.008 | 63 | 53 | 0.40 |
| 基线 vs grpo | +0.003 | 6 | 2 | 0.29 |
| 对照 vs curriculum | +0.009 | 31 | 20 | 0.16 |
| 对照 vs counterfactual | −0.005 | 42 | 48 | 0.60 |
| 对照 vs grpo | −0.010 | 40 | 52 | 0.25 |

## 4. 读法（诚实版）

1. **三种干预都没有跑赢"什么都不做、只多训一会儿"**。与对照相比三臂全部 n.s.
   （p = 0.16 / 0.60 / 0.25）。真要下结论只能说："在 2,000 条样本、1,500 步的预算下，
   没有任何一种干预带来可测的提升。"
2. **curriculum 相对基线显著变差**（p = 0.0067），但对照也在同一方向掉了 0.036——
   所以这个"显著"主要是**小样本上多训 1,500 步导致的遗忘**，不是课程的功劳/锅。
   课程训练日志里还有两次 loss 尖峰（step 1000: 1.79，step 1100: 2.38），正是
   "先易后难"切换的那一刻。
3. **counterfactual 是唯一让"冲突格上升、一致格下降"的臂**（+0.016 / +0.019 vs
   −0.042 / −0.026）。方向对，但总体 n.s.：它像是在先验与视觉之间**重新分配**了权重，
   没有净收益。这与"冲突加权 GRPO 应该提高 VDR"的假设一致，但幅度不够。
4. **GRPO 在这套设定下几乎不产生梯度**：200 组里 190 组奖励完全相同
   （`zero_adv_groups=190`），只有 10 次更新，49.4 秒就结束了。
   G=2 时组内基线把信号抹掉，这是比"GRPO 打不过 SFT"更具体的一条负结果。

## 5. 局限

- 只训 1,500 步 / 2,000 条，远小于主实验的 2,500 步 / 19,998 条；不能外推到 full 规模。
- 只用一个种子；训练方差本身在 ±0.013 量级（见报告 §10），三臂的差异都在这个量级附近。
- GRPO 的 G=2 太小组化；G=8 需要 4 倍采样时间，本次预算不允许。

## 6. 重跑

```bash
export PICOVLM_DATA=data/full
python -m stage5_intervene.train --method plain          --steps 1500 --limit 2000 --init sft_full --config sft_full.yaml --shuffle-seed 17 --out outputs/ckpt/plain_full
python -m stage5_intervene.train --method curriculum     --steps 1500 --limit 2000 --init sft_full --config sft_full.yaml --shuffle-seed 17 --out outputs/ckpt/curriculum_full
python -m stage5_intervene.train --method counterfactual --steps 1500 --limit 2000 --init sft_full --config sft_full.yaml --shuffle-seed 17 --out outputs/ckpt/counterfactual_full
python -m stage5_intervene.train --method grpo --steps 200 --limit 400 --group-size 2 --conflict-weight 2.0 --init sft_full --config sft_full.yaml --shuffle-seed 17 --out outputs/ckpt/grpo_full

python -m stage4_eval.run --model sft --suite dev_test --conditions correct_image,mismatched_image \
  --connector outputs/ckpt/<arm>_full/connector.pt --adapter outputs/ckpt/<arm>_full/lora --run-id arm_<arm>
python -m stage5_intervene.compare_arms --a arm_plain --b arm_<arm>
python -m stage5_intervene.table_arms sft_fulltrain_dev arm_plain arm_curriculum arm_counterfactual arm_grpo
```
