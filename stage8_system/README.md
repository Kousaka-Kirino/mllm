# 08 · 系统：把十个阶段串成一条命令

到这一阶段，前面九个目录各自能跑，但它们之间**只是"都放在同一个文件夹里"**，
还不是一个系统。系统的标志只有一个：**一条命令能跑完并给出一致的结论**。

这一阶段要解决三件事：

1. **一键矩阵**：`python -m stage8_system.run_all` 依次跑 S0→S9 的验收，
   汇总成 `outputs/acceptance.md`（人看）+ `acceptance.json`（机器看）。
2. **交付 CLI**：`python -m picovlm.cli --image x.png --ask "..."`，
   让复试老师（或你自己）不用改代码就能问一张图。
3. **不制造循环依赖**：run_all 会跑各阶段的 verify，所以 S8 的 verify **不能**
   要求"acceptance 必须已存在且全绿"，否则永远跑不起来。这是本项目里
   唯一一个"自指"陷阱。

---

## 1. 为什么需要 run_all（而不是九个脚本）

手工跑九个 verify 的问题是：**你会忘记跑其中一个**，而忘记的那个阶段
恰恰是最容易悄悄坏掉的（通常是改了共享包 `picovlm/` 之后）。

`run_all` 的契约：

| 契约 | 具体行为 |
|---|---|
| 顺序 | 严格 S0→S9，前面的失败不阻止后面的继续跑（要看到全貌） |
| 隔离 | 每个阶段一个子进程，一个阶段崩了不会污染下一个的内存/CUDA 状态 |
| 解析 | 从子进程输出里用正则数 `[PASS]` / `[FAIL]` / `[SKIP]` |
| 裁决 | 有任何 FAIL → 退出码 1；只有 SKIP → 退出码 0 但报告里标黄 |
| 留档 | 写 markdown（给人）+ json（给脚本/CI） |

### 一个真实的坑：SKIP 不是 PASS

S2 的 `real-weights` 检查在没加 `--load` 时会 SKIP（不下载 1.5GB 权重）。
如果 run_all 把 SKIP 算成 PASS，你会得到"55 项全绿"的假象。
所以 `acceptance.json` 里三项分开记，报告里也分开写。

---

## 2. 交付 CLI：三个模型档位

`picovlm/cli.py` 故意做得很小，因为它要能在**任何机器上**演示：

| `--model` | 载入什么 | 用途 | 显存 |
|---|---|---|---|
| `prior` | 什么都不载 | 盲基线：只看问题给答案 | 0 |
| `align` | connector 权重 | 只对齐过连接器的模型 | ~1.6GB |
| `sft` | connector + LoRA | 最终模型 | ~1.6GB |

`prior` 档是为了**现场对比**：先让模型不看图答一次，再看图答一次。
这个对比是整篇报告的核心叙事（"它到底看没看图"），CLI 让它一句话就能演示。

> 演示脚本：
> ```
> python -m picovlm.cli --image data/synth/images/xxx.png --ask "Which category has the highest value?" --model prior
> python -m picovlm.cli --image data/synth/images/xxx.png --ask "Which category has the highest value?" --model sft
> ```

---

## 3. 代码导览

| 文件 | 作用 | 关键函数 |
|---|---|---|
| `run_all.py` | 一键矩阵 | `STAGES`（10 项）、`run_stage`、`main` |
| `cli.py` | 交付 CLI | `main` |
| `verify_stage8.py` | 本阶段验收 | 6 项检查 |

`STAGES` 是一个 `(阶段名, 模块路径, 中文说明)` 的列表。**它是这个项目的
"总目录"**：想知道"这个项目有哪些证据"，看这十行就够了。

---

## 4. 作业

1. **HW1**：给 `run_all` 加 `--only S3,S4` 参数，只跑指定阶段。
2. **HW2**：实现"证据链检查"——报告里每条声明都必须能追到 `outputs/` 下
   某个文件的某个数字（S9 已经做了简化版，这里做通用的）。

题目与答案分别在 `homework/README.md`、`homework/HW8_提示.md`。

---

## 5. 验收标准

```
python -m stage8_system.verify_stage8
```
- `stages-wired`：≥10 个阶段、顺序正确、末位是 S9、全部可导入
- `cli-works`：`picovlm.cli --model prior` 能跑、输出含 `[prior]` 与"盲基线"
- `acceptance-if-present`：有 acceptance 时合计与明细一致
- `docs-present`：README / RUNBOOK / REPORT / CLAIMS 齐全
- `stage-docs`：十个阶段目录都有 README
- `readme`：根 README 含 run_all / acuity / 2×2 / stage9_report

六项全 PASS 才算这一阶段完成。
