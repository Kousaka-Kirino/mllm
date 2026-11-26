# 09 · 报告：把实验变成"可核查的结论"

这一阶段不训练、不评测，只做一件容易被低估的事：**把散落在 outputs/ 里的
数字，变成一份别人敢引用的报告**。

判断标准很朴素：**报告里任何一个数字，你都应该能在 30 秒内指到它的来源文件**。
做不到，就说明这个数字是"回忆"出来的，不是"算"出来的。

---

## 1. 三层结构

| 层 | 文件 | 谁看 | 稳定性 |
|---|---|---|---|
| 原始证据 | `outputs/eval/<run>/records.json`、`outputs/bench/*.json` | 你自己复核 | 每次跑都会变 |
| 汇总 | `reports/RESULTS.json` | 脚本 / CI / 作业 | 每次 collect 重写 |
| 叙事 | `reports/REPORT_2.0.md`、`CLAIMS.md` | 人（老师 / 面试官） | 手写，但数字必须来自汇总 |

`collect.py` 的职责就是**只搬运，不发明**：它把 `records.json` 里的
accuracy/CI、`bench` 里的显存/延迟、`memory` 里的 recall 读出来，
拼成一个大 json；`to_markdown` 再把它渲染成 7 节报告。

> **派生数字也要落盘。** 第一次写的时候，报告里出现了 `0.500`（SFT 在
> correct_image 上的 80 条平均），但 RESULTS.json 里只有分格的数，
> `verify_stage9` 的 `numbers-traceable` 立刻 FAIL。修法不是放宽检查，
> 而是把派生值写进 `main` 段。**能被查到的数才配写进报告。**

---

## 2. 13 条声明 ↔ 来源

`CLAIMS.md` 是简历条目与证据的一一对应表。它的作用不是"好看"，而是：

- 写简历时**只从这里抄**，不从记忆里抄；
- 每改一次实验，先更新 CLAIMS.md 的数字，再改简历；
- 面试官问"这个 95% 怎么来的"，你能直接翻到 `outputs/eval/sft_dev/records.json`。

声明分三档：
- **强**（有对照 + CI + 可复现）：一致格 95%、冲突格 0%、prior_effect +0.95 [0.83,1.00]
- **中**（单次实验，方向明确但未做统计检验）：GRPO 零优势组 5/6
- **弱 / 负**（失败结论）：页内召回 0.083、放大不提升 -0.042

**负结论必须写**，而且要和正结论并列。这不是谦虚，是因为"我们试了 X，
X 不行，原因是 Y"比"我们做到了 Z"更能说明你懂这个系统。

---

## 3. 代码导览

| 文件 | 作用 | 关键函数 |
|---|---|---|
| `collect.py` | 汇总 + 渲染 | `collect_eval`、`collect_bench`、`collect_memory`、`build_results`、`to_markdown` |
| `CLAIMS.md` | 声明对照表 | — |
| `verify_stage9.py` | 本阶段验收 | `results-json`、`report-sections`、`numbers-traceable`、`claims-table` |

`numbers-traceable` 的实现只有十几行，但它是整个项目里**性价比最高**的检查：
它挡住了"报告写了一个不存在的数"这一类最难发现的错误。

---

## 4. 作业

1. **HW1**：给 `collect.py` 加"与上一次结果对比"（Δaccuracy），
   并标出"变化超过 2 倍 Wilson 半宽"的项。
2. **HW2**：写一个 `make_slides.py`，从 RESULTS.json 生成 6 页幻灯片的
   markdown 大纲（只生成文字，不生成 pptx）。

题目与答案：`homework/README.md`、`homework/HW9_提示.md`。

---

## 5. 验收标准

```
python -m stage9_report.verify_stage9
```
- `results-json`：RESULTS.json 存在，含 data/train/eval/bench/memory/main
- `report-sections`：报告 7 个小节齐全
- `numbers-traceable`：报告里每个三位小数都能在 RESULTS.json 里找到
- `claims-table`：≥10 条声明且每条带来源

四项全 PASS。
