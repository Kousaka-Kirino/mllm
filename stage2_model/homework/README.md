# Stage 2 作业

三题，前两题必做，第三题选做。做完把文件留在本目录即可。

## HW1 · 视觉 token 数（必做）

文件：`hw1_connector_tokens.py`

实现 `visual_tokens(connector, input_size=384, patch=16, n_queries=64, factor=2)`
和 `check_budget()`，要求：

- `mlp` 返回 patch 数（`(input_size // patch) ** 2`）；
- `resampler` 返回 `n_queries`；
- `pixelshuffle` 返回 `patches // factor**2`，且 `patches % factor**2 != 0` 时抛 `ValueError`；
- `input_size % patch != 0` 时抛 `ValueError`；
- `check_budget()` 用 `picovlm.budget.count_tokens` 对照，三种连接器都必须一致。

验收：`python -m stage2_model.homework.hw1_connector_tokens` 打印
`hw1 PASS`，并自证 576 / 64 / 144。

提示：不要硬编码 576——你未来会换 336、448 的输入尺寸，硬编码会静默算错。

## HW2 · 显存估算（必做）

文件：`hw2_vram_estimate.py`

实现 `estimate_peak_mib(params_m, batch, seq_len, hidden, bytes_per_param=2,
checkpoint=False)`，产出"权重 + 激活"的粗略上界（MiB），再实现
`compare_with_measured(estimate_mib, measured_mib)` 给出相对误差。

要求：

- 权重 = `params_m * 1e6 * bytes_per_param / 2**20`；
- 激活按"每层每 token 的隐藏状态 + 注意力中间量"给一个可辩护的公式
  （公式写在你自己的注释里，没有唯一正确答案）；
- `checkpoint=True` 时激活量应显著下降（说明为什么）；
- 误差 > 40% 时打印"你的公式漏了什么"的提示。

验收：跑 `python -m stage2_model.homework.hw2_vram_estimate`，
把估算值与实测 `2189 MiB`（alloc，单图单问，576 视觉 token）对比，
在你的作业报告里写清楚**漏算项**（KV cache、cuBLAS workspace、临时 buffer、
碎片）。

## HW3 · 换连接器看差异（选做）

```bash
python -m stage2_model.smoke --load --connector mlp
python -m stage2_model.smoke --load --connector resampler
python -m stage2_model.smoke --load --connector pixelshuffle
```

记录三组：视觉 token / 延迟(ms) / 峰值显存(MiB)，填一张表。
思考：token 从 576 降到 64，延迟为什么不按比例降？（答：Qwen 的 decode 部分
是 15 个 token，视觉部分省下的算力占比有限；且小 token 数的 attention
已经不是瓶颈。）
