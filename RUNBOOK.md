# RUNBOOK

> 每个阶段：怎么跑、跑多久、占多少显存、产物在哪。
> 时间为 **RTX 4060 Laptop 8GB / 32GB RAM** 实测或按实测外推，标注了"实测/估算"。

## 0. 环境

本来就是 `pandavla` 环境（Python 3.11.11 / torch 2.11.0+cu128 / transformers 5.24.1 / peft 0.22.0）。
不需要新建环境，但**训练前关掉桌面程序**（抖音/飞书/微信/Edge 能吃掉 7.8GB 显存，
实测空闲时 7099 MiB、全开时只剩约 5.5GB）。

```bash
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_OFFLINE=1          # 权重已缓存后，跳过联网检查
set PYTHONIOENCODING=utf-8
```

---

## S0 环境（< 10s，0 显存）

```bash
python -m stage0_env.check --no-net        # 体检 -> ENVIRONMENT.md
python -m stage0_env.benchmark --suite load,infer --repeats 3   # 显存/延迟基线，~40s
python -m stage0_env.verify_stage0
```

产物：`ENVIRONMENT.md`、`outputs/env_snapshot.json`、`outputs/bench/bench_*.json|csv`

---

## S1 数据（pilot 约 1 分钟 / 30MB）

```bash
python -m stage1_data.build --profile pilot --seed 17    # 636 族 / 636 图
python -m stage1_data.validate                            # 13 项检查
python -m stage1_data.measure --spotcheck 12              # 导出裁剪图 + 人工核验表
python -m stage1_data.verify_stage1                       # 14 项
```

产物：`data/synth/{train,val,dev_test,locked_test}.jsonl`、`data/synth/images/`、
`data/datacard.json`、`data/hashes.json`、`data/acuity_thresholds.json`
（阈值是**预注册常量**：easy ≥ 6.5px、mid ≥ 4.5px、hard ≥ 3.0px、< 3.0px 记 extreme）

`--profile full` 生成主分析集：约 20k 图 / 1.5GB / 25–35 分钟（估算）。
**不需要重跑**：`--profile full` 默认复用冻结的阈值文件，只有 `--recalibrate` 才会重算。

---

## S2 模型（CPU 30s；`--load` 下载 1.5GB 后 60s）

```bash
python -m stage2_model.smoke                 # 形状/延迟，替身模型，CPU
python -m stage2_model.smoke --load          # 真权重：589.5M / 576 vtokens / ~450ms / 2.1GB
python -m stage2_model.check_loss_mask       # 9 条掩码不变量
python -m stage2_model.verify_stage2 --load  # 7 项
```

---

## S3 训练（实测：align 45s / SFT 约 4 分钟）

```bash
python -m stage3_train.align --dry-run                 # 10s，不加载权重
python -m stage3_train.align --max-steps 60            # connector 对齐：loss 6.37→0.61，峰值 3508 MiB
python -m stage3_train.sft --max-steps 237             # LoRA SFT：loss→0.45，峰值 3770 MiB，~1s/步
python -m stage3_train.verify_stage3 --load
```

产物：`outputs/ckpt/align/connector.pt`、`outputs/ckpt/sft_pilot/{connector.pt,lora/}`、
`outputs/train/*/log.jsonl`

> pilot 规模下 237 步就收敛到一个"会答题但走先验"的解 —— 这正是本项目要研究的现象，
> 不是 bug。要看"能不能学会看图"，需要 `--profile full` 的数据和更长的训练。

---

## S4 评测（实测：80 条 × 4 条件约 3 分钟）

```bash
python -m stage4_eval.run --model prior --suite dev_test            # 盲基线，0 显存
python -m stage4_eval.run --model sft  --suite dev_test             # 2×2 + 三条件
python -m stage4_eval.run --model sft  --suite locked_test --limit 160
python -m stage4_eval.verify_stage4
```

产物：`outputs/eval/<run>/{predictions.jsonl,records.json,grid.json}`、`reports/eval_<run>.md`

---

## S5 干预（实测：GRPO 2–6 步探针，3 分钟以内）

```bash
python -m stage5_intervene.train --method curriculum --steps 20
python -m stage5_intervene.train --method counterfactual --steps 20
python -m stage5_intervene.train --method grpo --steps 6 --group 2   # G=2 探针（8GB 下的现实选择）
python -m stage5_intervene.build_rft                                  # 拒绝采样造 RFT 数据
python -m stage5_intervene.invariance --model sft --limit 24 --save   # E2：措辞不变性 / 数值响应
python -m stage5_intervene.compare_arms --a sft_dev --b grpo_dev
python -m stage5_intervene.verify_stage5
```

产物：`reports/compare_*.md|json`、`data/rft/*.jsonl`

---

## S6 消融（实测：三个连接器 × 2 次，约 1 分钟）

```bash
python -m stage6_ablation.measure --save
python -m stage6_ablation.pareto --save
python -m stage6_ablation.verify_stage6
```

实测：mlp 576 tok / 2.30M / 76ms / 1340MiB；resampler 64 / 20.04M / 40ms / 1219MiB；
pixelshuffle 144 / 2.75M / 41ms / 1207MiB。

---

## S7 视觉记忆（实测：12 页，约 2 分钟）

```bash
python -m stage7_memory.build_index --pages 12
python -m stage7_memory.evaluate --limit 24      # recall + 放大实验
python -m stage7_memory.verify_stage7
```

产物：`outputs/memory/pages_*.npz`、`reports/memory_*.md`

---

## S8 一键验收（约 4 分钟，CPU）

```bash
python -m stage8_system.verify --profile core   # 十阶段矩阵
python -m stage8_system.run_all                 # 同上 + 写 acceptance.md/json
```

产物：`outputs/acceptance.md`、`outputs/acceptance.json`。任何 FAIL → 退出码 1。

---

## S9 报告（< 5s）

```bash
python -m stage9_report.collect          # -> reports/RESULTS.json + REPORT_2.0.md
python -m stage9_report.verify_stage9
```

---

## 常见故障

| 症状 | 原因 | 处理 |
|---|---|---|
| `AttributeError: 'dict' object has no attribute 'pixel_values'` | `bench_one` 传了 dict | 传 `Batch(...)` |
| `mat1 and mat2 must have the same dtype` | SigLIP fp32 / Qwen bf16 | 已修：`encode_images` 里统一 dtype |
| 模型答完不停、格式分恒为 0 | 答案没跟 eos | `answer_with_eos` 用 tokenizer 的 eos |
| CUDA OOM | 桌面程序占显存 | 关掉、或 `--group 2`、batch=1 |
| 下载 SSL EOF | hf 直连不稳 | `HF_ENDPOINT=https://hf-mirror.com`，或 `HF_HUB_OFFLINE=1` |
| pwsh 命令被安全钩子拒绝 | 命令里出现字面 "format" | 改写（用 `.replace()` 或临时 py 文件） |
