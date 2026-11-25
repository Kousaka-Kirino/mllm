# ENVIRONMENT

> ⚠️ 本文件由 `python -m stage0_env.check_env` 自动生成，**不要手改**。
> 手改会和 `outputs/env_snapshot.json` 对不上，verify_stage0 会报 FAIL。

生成时间：2026-11-25T13:37:10+00:00　PicoVLM v0.1.0

## 1. 机器

- 系统：Windows 10 (AMD64)
- CPU：20 核　内存：31.7 GB
- 磁盘：32.4 GB 可用 / 651.6 GB

## 2. GPU

- NVIDIA GeForce RTX 4060 Laptop GPU，总显存 8188 MiB，当前可用 **7099 MiB**
- CUDA 12.8，驱动 572.83，capability [8, 9]

> 可用显存远小于总显存时，训练前先关掉浏览器/聊天软件；本项目按「可用 5.5–6.5 GB」设计。

## 3. 依赖

| 包 | 版本 |
|---|---|
| torch | 2.11.0+cu128 |
| transformers | 5.24.1 |
| peft | 0.22.0 |
| accelerate | 1.16.0 |
| safetensors | 0.8.0 |
| numpy | 2.2.4 |
| PIL | — |
| matplotlib | 3.10.1 |
| pytest | 9.2.0 |
| datasets | — |
| trl | — |
| bitsandbytes | — |
| gradio | — |

## 4. 网络

- https://hf-mirror.com → 200
- https://www.modelscope.cn → 200

## 5. 种子区间（冻结）

| 用途 | 区间起点 |
|---|---|
| data | 10000 |
| split | 20000 |
| bootstrap | 30000 |
| train | 40000 |
| grpo | 50000 |
| eval | 60000 |

## 6. 体检结论

| 项 | 状态 | 细节 |
|---|---|---|
| python | PASS | 3.11.11 |
| ram | PASS | 31.7 GB |
| disk | PASS | 32.4 GB free |
| core-packages | PASS | ok |
| optional-packages | SKIP | absent: datasets, trl, bitsandbytes |
| gpu | PASS | NVIDIA GeForce RTX 4060 Laptop GPU 8188MiB, free 7099MiB |
| network | PASS | {"https://hf-mirror.com": 200, "https://www.modelscope.cn": 200} |
| seeds | PASS | 6 bands frozen |
