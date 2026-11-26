# Stage 5 作业

## HW5-1 · 组内优势（必做）

`hw1_advantage.py`：实现 `advantages(rewards, eps=1e-6)`。

期望：`[0,0,0,1]` → `[-0.577,-0.577,-0.577,1.732]`。

**关键**：std 用总体标准差（÷n）。用样本标准差（÷(n−1)）会得到 −0.5/1.5，
和 GRPO 原文不一致。

## HW5-2 · clip 的生效阈值（必做）

`hw2_clip_when.py`：扫描 δ，找出 clip 首次生效处，与解析值 `ln(1+clip)` 对照。

结论应该是一句话：**adv>0 时，只有 `ratio > 1+clip`（即 δ > ln 1.2 ≈ 0.182）
clip 才截断**。单次更新 ratio≡1，所以 clip 永远不生效——它只在"同一批 rollout
反复更新"时才有意义。

## HW5-3 · RFT 对照（选做）

```bash
python -c "from stage5_intervene.rft import build; ..."
```

在 40 条样本上采 G=4，统计 `keep_rate`。keep_rate 越高，"拒绝采样 + SFT"
这个便宜基线就越强——写一句它对 GRPO 的威胁。

## 验收标准

| 作业 | 通过条件 |
|---|---|
| HW5-1 | 打印 `hw5-1 PASS` |
| HW5-2 | 扫描值 ≈ 0.182，并写出那句 clip 结论 |
| HW5-3 | 报告 keep_rate 与一句判断 |
