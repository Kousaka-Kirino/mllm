# HW1 / HW2 参考答案与验收标准

> 先自己写，卡住再看。这里的答案不是唯一写法，重点在**边界条件**。

---

## HW1 · family 划分

### 关键思路

1. **先洗牌再切**：用 `np.random.default_rng(seed)` 对 family 列表做 permutation。
   不要用 `random.shuffle`（不可复现的坑之一）。
2. **最大余数法**：`n * ratio` 向下取整后，把剩下的名额依次给"小数部分最大"的
   split，保证总数守恒且比例误差最小。
3. **不要按行切**：按 family 切完再回填样本，而不是先切样本再按 family 分组。

### 参考答案

```python
import numpy as np

NAMES = ("train", "val", "dev_test", "locked_test")


def split_by_family(families, ratios, seed=17):
    fams = list(families)
    n = len(fams)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    fams = [fams[i] for i in order]

    raw = np.asarray(ratios, dtype=float)
    raw = raw / raw.sum()
    quota = np.floor(raw * n).astype(int)
    # 最大余数法：按小数部分从大到小补齐
    rem = n - quota.sum()
    frac = raw * n - quota
    for i in np.argsort(-frac)[:rem]:
        quota[i] += 1

    out, start = {}, 0
    for name, k in zip(NAMES, quota):
        out[name] = fams[start:start + k]
        start += k
    return out
```

### 验收断言

```python
fams = [f"fam-{i:04d}" for i in range(120)]
a = split_by_family(fams, (0.8, 0.1, 0.05, 0.05), seed=17)

# 1. 不重不漏
flat = [x for v in a.values() for x in v]
assert sorted(flat) == sorted(fams)

# 2. 比例误差 ≤ 0.02
for name, r in zip(NAMES, (0.8, 0.1, 0.05, 0.05)):
    assert abs(len(a[name]) / len(fams) - r) <= 0.02, (name, len(a[name]))

# 3. 可复现
b = split_by_family(fams, (0.8, 0.1, 0.05, 0.05), seed=17)
assert a == b

# 4. 边界
assert split_by_family([], (0.8, 0.1, 0.05, 0.05)) == {k: [] for k in NAMES}
one = split_by_family(["only"], (0.8, 0.1, 0.05, 0.05))
assert sum(len(v) for v in one.values()) == 1
```

### 易错点

- `int(n * 0.05)` 对 n=120 是 6，四个加起来 120，正好；但 n=7 时
  `0.8*7=5.6→5`、`0.1*7=0.7→0`，不加余数法会丢 2 个 family。
- 传进来的 `families` 可能不唯一：应该在函数入口 `assert len(set(fams)) == n`，
  或者明确写成"调用方保证唯一"（本项目采用后者，因为 build 时已经保证）。
- 返回顺序：`dev_test` 在 `locked_test` 前面，别把后两个搞反——它们用途不同
  （一个用来选配置，一个只在最后跑一次）。

---

## HW2 · 着墨高度

### 关键思路

1. 先按阈值二值化：`ink = patch < thr`；
2. 按行求 `ink.any(axis=1)` 得到"这一行有没有墨"；
3. 数**最长连续 True 段**（不是所有 True 的和——文字块里可能夹着空隙，
   比如 "=" 或两行数字）。

### 参考答案

```python
import numpy as np


def ink_height(patch, thr=200):
    if patch.size == 0:
        return 0
    rows = (np.asarray(patch) < thr).any(axis=1)
    best = cur = 0
    for flag in rows:
        cur = cur + 1 if flag else 0
        best = max(best, cur)
    return int(best)


def scale_to_model(px_render, render_size, input_size=384):
    return float(px_render) * float(input_size) / float(render_size)
```

### 验收断言

```python
p = np.full((5, 4), 255, np.uint8); p[1:4, :] = 10
assert ink_height(p) == 3
assert ink_height(p, thr=5) == 0          # 阈值太严，什么都没有
assert ink_height(np.full((3, 3), 255, np.uint8)) == 0   # 全白
assert ink_height(np.zeros((0, 4), np.uint8)) == 0        # 空块

assert abs(scale_to_model(27, 1152) - 9.0) < 1e-9
assert abs(scale_to_model(9, 1152, 384) - 3.0) < 1e-9
```

### 你会观察到的现象（写进报告）

- 同一字号，阈值 200 → 180，着墨高度可能 +1px；小字号更容易被阈值影响。
  这就是为什么正式数据里阈值必须固定并写进 datacard。
- 用 `any(axis=1)` 而不是 `sum(axis=1) > W/2`：数字笔画是竖线，横向像素少，
  用"半行以上有墨"会把数字判成空。

---

## HW3 · 抽查（选做）

推荐做法：从 `data/synth/locked_test.jsonl` 里抽 10 条，打开 `image_path`，
先自己回答，再对照 `answer`。把答错的记为"我看错了"，把标注错的记为
"数据 bug"——后者才是要修的东西。WILSON 置信区间在这么小的样本上没有意义，
所以这题不算统计，只算质检。
