# Stage 5 · 三种干预（受控实验）

> 这一阶段回答 RQ2：**在同样的算力预算下，哪种干预最能提升"冲突格"的准确率？**
> 三种干预只在"喂什么数据 / 给什么信号"上不同，模型、优化器、评测全部钉死。

---

## 1. 三个臂

| 臂 | 干预 | 一句话 | 关键风险 |
|---|---|---|---|
| `curriculum` | 先易后难采样（前 40% 只用易样本） | 让模型先学会"看清"，再学"看清小字" | 易样本服从未见过难样本会退化 |
| `counterfactual` | 原样本 + 措辞变体（金标不变）+ 数值变体（金标改变） | 逼模型每次重新读图，堵住"背先验"捷径 | 数值变体会引入渲染噪声 |
| `grpo` | 组内相对奖励 + 只对答案 token 反传 | 用"在同一条数据上谁更好"当信号 | 整组同分时完全没信号（见 §4） |

对照组：`rft`（拒绝采样微调）——只留采样到的正确答案当 SFT 目标。
**如果不跑 RFT，就没法说 GRPO 有效**：RFT 通常更便宜、更稳。

---

## 2. 反事实数据怎么造（`counterfactual.py`）

- **措辞变体**：`"Which category has the highest value?" → "Which category is the largest?"`
  图与金标都不动。模型若对措辞敏感，说明它在背题面。
- **数值变体**：改表里最大值那一列，让 argmax 换人，**重新渲染**，金标跟着变
  （pilot 实测：`A → F`）。模型若仍答旧答案，说明它没重新读图。
- 两种变体都保留 `family_id`，划分时整族进同一 split（2.0 §4.4）。

---

## 3. GRPO 的四个坑（VLM 版）

1. **loss 只算答案 token**。视觉 576 位 + prompt 位必须 `-100`，
   否则模型会把"复述问题"当成低 loss 的捷径。
2. **同一组 rollout 共用一份 `pixel_values`**。代码里 `expand(G, ...)`；
   否则显存按 G 倍涨，8GB 直接 OOM。
3. **组内优势在整组同分时全 0**。这不是 bug，是信号缺失——必须计数并报告
   `zero_adv_groups`，否则你会以为"训练在跑但没学会"是超参问题。
4. **格式分与正确分必须分开报**。`total = correct + 0.2×format`。
   如果只报 total，格式崩了会被正确率的提升掩盖。

### 关于 PPO clip：一个诚实的说明

每个 rollout 只做**一次**参数更新时，新旧策略完全相同，`ratio ≡ 1`，
clip 永远不生效——此时 GRPO 就是"带组内基线的 REINFORCE"。
clip 的价值在于"同一批 rollout 多 epoch 复用"，而 8GB 下我们**不能**复用
（存不下 logp_old）。代码保留 clip 并写了测试，是为了让这个取舍可见，
而不是假装在用 PPO。

---

## 4. pilot 实测（GRPO，4 步 × G=4，12 条训练样本）

```
step   0 r=[0.0, 0.0, 0.0, 1.0] adv=[-0.58, -0.58, -0.58, 1.73] loss=-7.862
step   1 整组同分（r=[0,0,0,0]）跳过
step   2 整组同分（r=[1,1,1,1]）跳过
step   3 整组同分（r=[1,1,1,1]）跳过
step   4 整组同分（r=[1,1,1,1]）跳过
step   5 整组同分（r=[1,1,1,1]）跳过
→ steps=1, zero_adv_groups=5, reward_mean=0.25
```

**6 个 rollout 组里有 5 组完全没有梯度信号**。原因很朴素：模型对同一张图
的 4 次采样要么全对、要么全错。这个数字比任何"GRPO 提升了 X%"都更有信息量——
它说明在这个规模上，**奖励的方差主要来自"题目难度"而不是"回答质量"**，
而 GRPO 的组内基线恰好把难度差异减掉了，剩下几乎全是噪声。

这正是 PandaVLA 的结论在 VLM 上的对应物：
**同数据 SFT 就够的场景里，RL 的增量信号很薄。**

---

## 5. 代码

```
stage5_intervene/
├── counterfactual.py  # 措辞/数值变体
├── grpo.py            # reward / advantages / logprob / clip / rollout
├── rft.py             # 拒绝采样微调的数据构造
├── train.py           # 三种臂的统一入口（--method）
├── compare_arms.py    # 逐格配对 + McNemar
└── verify_stage5.py
```

```bash
python -m stage5_intervene.train --method curriculum --steps 40
python -m stage5_intervene.train --method counterfactual --steps 40
python -m stage5_intervene.train --method grpo --steps 10 --group-size 2
python -m stage5_intervene.compare_arms --a sft_dev --b grpo_dev
python -m stage5_intervene.verify_stage5
```

---

## 6. 作业

### HW5-1 · 手推组内优势（必做）

`homework/hw1_advantage.py`：实现 `advantages(rewards)`，并证明
`[0,0,0,1]` 的优势是 `[-0.577, -0.577, -0.577, 1.732]`（保留 3 位）。

### HW5-2 · clip 什么时候生效（必做）

`homework/hw2_clip_when.py`：构造 `logp_new = logp_old + δ`，扫描
δ ∈ [0, 3]，记录 clip 生效的 δ 阈值，并解释为什么"单次更新 + 小 lr"下
它几乎不生效。

### HW5-3 · RFT 对照（选做）

用 `rft.py` 在 40 条样本上采 G=4，统计 keep_rate。如果 keep_rate > 50%，
说明"拒绝采样 + SFT"是很便宜的基线——写一句它对 GRPO 的威胁。

---

## 7. 验收标准

`python -m stage5_intervene.verify_stage5` 必须 0 FAIL：

- `reward-decomposition`：total/correct/format 三项对得上，冲突加权正确；
- `group-advantages`：同分全 0、`[0,1]` 得 ±1、单元素 0；
- `clip-behavior`：ratio=1 时 loss 等于 `-adv·logp`，大偏差时 clip 生效；
- `counterfactual-wording`：金标/图/族不变，问句变；
- `counterfactual-numeric`：金标变、图变、族不变（≥8/10）；
- `curriculum-order`：头 40% 全是易/中样本，且样本集合不变；
- `grpo-probe`：2 步真跑（整组同分跳过属正常）；
- `arms-compare`：A vs A 生成报告且含 McNemar。

---

## 8. 延伸

- GRPO 原论文（Shao et al., 2024）的组内归一化；
- RFT / rejection sampling finetuning 与 best-of-n 的关系；
- 为什么"组内同分"在难任务上更常见（奖励饱和/地板效应）。
