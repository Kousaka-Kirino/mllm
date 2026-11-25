# RUNBOOK

> 每个阶段：怎么跑、跑多久、占多少显存、产物在哪。
> 时间是 **RTX 4060 Laptop 8GB** 上的估算，实测后替换。

## 0. 一次性准备

```bash
conda create -n picovlm python=3.11 -y
conda activate picovlm
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
set HF_ENDPOINT=https://hf-mirror.com     # 国内下载
```

## S0 环境

```bash
python -m stage0_env.check_env --no-net   # 不测网络，快
python -m stage0_env.verify_stage0
```

产物：`ENVIRONMENT.md`、`outputs/env_snapshot.json`。耗时 < 5s（不含网络）。

## S1 数据

```bash
python -m stage1_data.build               # 小规模 312 条，约 40s，9MB
python -m stage1_data.build --full        # 全量约 5k 图，6–8 分钟，250MB
python -m stage1_data.verify_stage1
```

磁盘：小规模 9MB / 全量约 250MB。**先跑小规模把管线跑通**。

## S2 模型

```bash
python -m stage2_model.verify_stage2          # CPU，30s
python -m stage2_model.verify_stage2 --load   # 下载 1.5GB 权重后测真模型
```

## S3 训练

```bash
python -m stage3_train.train_align --dry-run   # 形状检查，10s
python -m stage3_train.train_align --limit 50000   # 约 2h
python -m stage3_train.train_sft --epochs 2        # 约 6–9h/epoch
python -m stage3_train.eval_suite --policy prior --split test   # 先跑基线
```

显存：对齐约 2.5–3GB，SFT 约 3.5–4.5GB（batch=2、梯度检查点、bf16）。
**训练前关掉抖音/飞书/浏览器标签页**——它们能吃掉 1–2GB。

## S4 诊断

```bash
python -m stage4_diag.grid_2x2 --run-id <run>
```

产物：`reports/grid_<run>.md`、`outputs/eval/<run>/grid.json`。

## S5 干预

```bash
python -m stage5_intervene.compare_arms --a sft-test --b grpo-test
```

## S6 预算

```bash
python -m stage6_budget.verify_stage6
```

## S7 记忆

```bash
python -m stage7_memory.verify_stage7
```

## S8 一键验收

```bash
python -m stage8_system.run_all
```

产物：`outputs/acceptance.md`。任何 FAIL → 退出码 1。
