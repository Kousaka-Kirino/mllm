# Stage 3 作业

两题必做，一题选做。全部在 CPU 上可完成（不需要下模型权重）。

## HW3-1 · 梯度累积等价性（必做）

文件：`hw1_grad_accum.py`

实现 `grad_accum_equivalent(...)`：用 `_DummyLLM` + 固定 seed，比较

- `bs=1, accum=4`（4 次前向、每次 loss/4 反传）
- `bs=4, accum=1`（1 次前向、loss 直接反传）

同一批 4 条数据上，两种方式的 `p.grad` 相对误差 < 1e-3 返回 True。

这是"8GB 显存能不能用大 batch"的理论基础：显存只跟**单次前向的 batch** 有关，
累积只是把梯度加起来。

## HW3-2 · 手算 LoRA 参数量（必做）

文件：`hw2_lora_params.py`

实现 `lora_params(d_model=896, layers=24, targets=4, r=8)`，返回参数量。
参考实测 **1.08M**（Qwen2.5-0.5B，r=8，alpha=16，q/k/v/o）。

写清楚你的公式：每个被挂的投影 `2 × r × d_model`，共 `layers × targets` 个。

## HW3-3 · 学习率探针（选做）

```bash
python -m stage3_train.align --limit 128 --max-steps 30 --lr 5e-5
python -m stage3_train.align --limit 128 --max-steps 30 --lr 2e-4
python -m stage3_train.align --limit 128 --max-steps 30 --lr 1e-3
```

记录三组 val_loss，写一句结论。注意 5.1 的坑：`--max-steps` 会压扁 lr 表，
所以三组要**用同一个 max-steps** 才有可比性。
