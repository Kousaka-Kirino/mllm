# 阶段 0 作业

> 题目正文在 [../README.md](../README.md) 第 9 节。这里只放**提交要求**和**骨架文件**。

## 提交要求

每题交两样东西：

1. **代码**：能跑通的 `.py` 文件（骨架里标了 `TODO` 的地方补完）；
2. **结论**：一句话，必须带数字。例如
   - 作业 1：`估算峰值 2.6 GiB，实测 2.21 GiB，差 18%，主要差在 logits 上`
   - 作业 2：`关桌面后 idle free 从 5.5GB 回到 7.1GB，训练预算按 7GB 规划`

> **不要只贴代码。** 没有数字的作业等于没做。

## 文件

| 题 | 文件 | 说明 |
|---|---|---|
| 1 | `hw1_vram_budget.py` | 纯计算 + 和 `outputs/bench/*.json` 对照 |
| 2 | `hw2_idle_free.py` | 采样 idle free，判断真实训练预算 |
| 3 | `hw3_thresholds.md` | 文字作答：写你自己的 `configs/env.yaml` 门槛 |

## 检查自己

```powershell
$env:PYTHONIOENCODING="utf-8"
python stage0_env/homework/hw1_vram_budget.py
python stage0_env/homework/hw2_idle_free.py
```

两个脚本都能跑通、并且打印出的结论对得上
[HW0_提示.md](HW0_提示.md) 里的"验收标准"，就算过。
