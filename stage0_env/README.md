# 阶段 0：环境、契约与硬件基准

> 目标：把"我这台机器上到底能跑什么"变成一份**可复核的事实文件**，而不是一句"应该能跑"。
>
> 产出：`ENVIRONMENT.md`、`outputs/env_snapshot.json`、`outputs/bench/bench_*.csv`、`configs/*.yaml`。
>
> 耗时：首次约 15 分钟（含下载底座 1.2GB），之后 1 分钟。

---

## 1. 为什么这一阶段不能跳过

本项目的所有结论都建立在"8GB 显存能跑多大的模型、多长的序列"之上。
如果显存账算错，后面会发生两种事故，而且都不会报错：

- **你以为在训练，其实在交换内存**。显存不够时 PyTorch 会退到共享内存（WDDM），
  训练不 OOM、但慢 10 倍，你会以为是"模型太大"。
- **你以为跑的是同一个实验**。换了 torch 版本或底座 revision，
  `bf16` 的数值行为、tokenizer 的切分都可能变，两次结果就不可比。

所以 S0 的产物不是代码，是**约束**：哪些数字之后不能再变、哪些每次实验都要重测。

## 2. 要钉死的三件事

| 要钉死的 | 在哪 | 变了会怎样 |
|---|---|---|
| Python / torch / transformers / peft 版本 | `ENVIRONMENT.md` | 训练曲线不可复现 |
| 底座权重与 revision | `configs/model.yaml` | 换 revision 等于换模型，指标不可比 |
| 样本 schema 与配置约定 | `picovlm/schema.py`、`picovlm/config.py` | 数据/训练/评测对不上字段 |

这三样是**冻结接口**。后面任何阶段想改，必须先在 README 里写清楚"为什么改、旧结果怎么处理"，
而不是悄悄改掉。

## 3. 显存的三个数字，别混用

```
total      = 显卡标称（8188 MiB）      ← 宣传数字，没有意义
free       = 此刻还能借多少（idle 7099 MiB） ← 桌面程序随时会变
allocated  = 你的张量真正占了（训练时看这个）
reserved   = PyTorch 缓存池占了（allocated + 碎片）
```

判断"会不会 OOM"要看 **allocated 峰值**，不是 free。
判断"是不是快到极限"要看 **reserved 是否远大于 allocated**（碎片多，说明该降 batch）。

本项目在 4060 Laptop 上的真实情况：桌面程序（浏览器、聊天软件、IDE）会长期占用约 7.8GB 中的大部分，
**训练前必须关掉**，否则 idle free 会掉到 5.5GB 以下，S3 的 LoRA SFT 直接 OOM。

## 4. 代码导览

| 文件 | 作用 | 关键函数 |
|---|---|---|
| `stage0_env/check_env.py` | 体检实现：python/内存/磁盘/包/GPU/网络/种子 | `collect(net)`、`to_markdown(checks, snap)` |
| `stage0_env/check.py` | 2.0 规划的 CLI 入口，控制输出路径与退出码 | `run(output, net, write_md)`、`main()` |
| `stage0_env/benchmark.py` | 硬件基准：加载 + 单图前向，重复测、取中位数 | `bench_load`、`bench_infer`、`summarize` |
| `stage0_env/verify_stage0.py` | 阶段验收（本阶段的"考试"） | `main()` |
| `configs/env.yaml` | 门槛值：显存/磁盘/必需包，可改但要有理由 | — |
| `configs/model.yaml` | 底座规格，2.0 的冻结项 | — |

## 5. 跑一遍

```powershell
$env:PYTHONIOENCODING="utf-8"
conda activate pandavla

# 1) 体检（写 ENVIRONMENT.md，不要手改这个文件）
python -m stage0_env.check

# 2) 硬件基准（首次会下载约 1.2GB 权重；无网用 --dry-run 只验测量框架）
python -m stage0_env.benchmark --suite load,infer --repeats 3

# 3) 阶段验收
python -m stage0_env.verify_stage0
```

期望看到类似：

```
[PASS] gpu    NVIDIA GeForce RTX 4060 Laptop GPU 8188MiB, free 7099MiB
| load  | 2 | 62.4  | 2140 | 2600 |
| infer | 2 | 118.7 | 2210 | 2680 |
```

`load` 的峰值是"两个底座都在显存里"的占用，`infer` 是单图前向的增量——
这两个数决定了 S3 的 batch 上限，**必须来自真机，不能抄别人的博客**。

## 6. 怎么读一次基准才算"可信"

一次测量不算数。规则：

1. **重复 ≥3 次**，报中位数（第一次含冷启动，会被中位数自动吸收）；
2. **记录当时的 idle free**（`bench_*.json` 里的 `idle_free_mib`），
   同样配置在不同桌面负载下差 2GB 是正常的；
3. **区分 load 与 infer**：load 是常驻占用，infer 是峰值增量；
4. 失败也要落盘（OOM 是哪一步、reserved 长什么样），负结果同样进报告。

## 7. 分级门槛（本阶段只到 L0）

| 等级 | 门槛 | 本阶段 |
|---|---|---|
| L0 环境成立 | CUDA 可用；底座能加载；单图前向无 NaN/OOM | ✅ 本阶段必须过 |
| L1 最小闭环 | connector 反传、LoRA 训练、保存/恢复 | S3 |
| L2 核心可行 | 主线单配置连续 500 步无 OOM/NaN | S3 |
| L3 完整主线 | 锁定测试 + 两种干预 | S4/S5 |
| L4 扩展 | GRPO/RFT、OOD、消融 | S5–S7 |

## 8. 我在这台机器上踩过的坑（真实记录）

| 坑 | 现象 | 解法 |
|---|---|---|
| `torch.cuda.get_device_properties` 没有 `major_minor` | `AttributeError`（torch 2.11） | 用 `props.major, props.minor` |
| 网络探测用 HEAD | hf-mirror 返回 `URLError` | 改 GET，`--no-net` 可跳过 |
| `nvidia-smi` 不在 PATH | 驱动版本取不到 | 先试 `C:\Windows\System32\nvidia-smi.exe`，再退回注册表 `DriverVersion` |
| PowerShell 里命令含 `format` | 被安全钩子拒绝 | 避开该词，用 `Select-Object` 代替 `Format-Table` |
| 中文输出乱码 | 控制台 GBK | 先 `$env:PYTHONIOENCODING="utf-8"` |
| 桌面程序占显存 | free 从 7.1GB 掉到 5.5GB | 训练前关闭；基准里记录 idle free |

把这些写进文档不是自嘲，是**让别人能复现你的失败**——这比只写成功路径有用得多。

## 9. 作业

题目在 [`homework/README.md`](homework/README.md)，答案与验收标准在
[`homework/HW0_提示.md`](homework/HW0_提示.md)。三道题都不长，但都要**跑出数字**：

1. 用实测值算一次显存账：给 `batch=2, seq=576+32` 估峰值，再和 benchmark 对照；
2. 对比"关桌面程序"前后的 idle free，判断你这台机器的真实训练预算；
3. 写一条你自己的 `configs/env.yaml` 门槛，并说明为什么是这个数。

## 10. 验收标准

`python -m stage0_env.verify_stage0` 会检查：

- 快照 JSON 存在且字段完整（python/内存/磁盘/包/GPU/网络）；
- GPU 型号与 `configs/env.yaml` 的 `name_contains` 一致，显存 ≥ `min_total_mib`；
- 必需包全在，可选包缺失要记录而不是 FAIL；
- `ENVIRONMENT.md` 是自动生成的（含"自动生成"标记）；
- 硬件基准 CSV 存在，且至少覆盖 `load` 与 `infer`；
- `configs/*.yaml` 能被 `load_config` 解析（`base:` 链无环）；
- 种子区间冻结文件存在。

全部 PASS 才允许进 S1。

## 11. 延伸

- 想理解 WDDM 共享显存：看任务管理器"性能 → GPU → 专用/共享"两列；
- 想比较底座：`configs/model.yaml` 改 `vision.model` 后重跑 benchmark，别只看参数量；
- 想压显存：优先降 `input_size` 与 `max_text_tokens`，其次 `grad_checkpoint`，最后才是换底座。
