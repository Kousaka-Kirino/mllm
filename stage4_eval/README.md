# Stage 4 · 诊断评测（2×2）

> 这一阶段的项目意义最大：**它把"模型小所以不行"拆成"看不见"和"不看"两种可测的失败。**
> 其余阶段都是在为它服务，或者从它出发做干预。

---

## 1. 四格设计

|  | prior 一致 | prior 冲突 |
|---|---|---|
| **acuity 易** | A | B |
| **acuity 难** | C | D |

- **一致格**：图里的答案与先验（位置/边际/数值/顺序偏置）一致；
- **冲突格**：图里的答案与先验相反；
- 盲基线（只给问题、不给图）在一致格能拿高分，在冲突格必然接近 0。

三个可解释的量：

| 量 | 定义 | 说明什么 |
|---|---|---|
| `prior_effect` | A − B | 同样的图，只把先验翻成冲突，掉多少分 → **先验依赖强度** |
| `acuity_effect` | A − C | 先验一致时把字号调小，掉多少分 → **视觉分辨率瓶颈** |
| `interaction` | (A−B) − (C−D) | 两个因素是否相乘（小字 + 冲突是否更糟） |
| `VDR` | B、D 的准确率 | **看图依赖率**：图能翻掉先验，才叫真的在看图 |

---

## 2. 四种输入条件（E0）

| condition | 图 | 问题 | 测什么 |
|---|---|---|---|
| `correct_image` | 正确图 | 真实问题 | 完整能力 |
| `mismatched_image` | 别族的图 | 真实问题 | 换图后答案变不变 |
| `prior_only` | 纯灰图 | 真实问题 | 纯先验（盲基线） |
| `image_only` | 正确图 | 泛化问句 | 不看问题 |

`mismatched_image` 是整个评测里最关键的一列：**如果换一张图答案不变，模型就没在看图。**
它比"冲突格准确率低"更直接，因为它不依赖"模型是否理解冲突"这种解释。

---

## 3. pilot 实测（第一次 SFT，70 步：历史读数）

> 这是**第一次跑通管线时**的读数（70 步、未加 EOS 监督之前），保留下来是为了让你看见
> "模型从半成品到成品"的变化。**现在的正式读数是下面 §3.1 那张表。**

| condition | 易·一致 | 易·冲突 | 难·一致 | 难·冲突 | overall |
|---|---:|---:|---:|---:|---:|
| `prior`（规则盲基线） | 1.000 | 0.000 | 1.000 | 0.000 | 0.500 |
| `correct_image` | **0.750** | **0.100** | **0.750** | **0.050** | 0.325 |
| `mismatched_image` | 0.750 | 0.150 | 0.750 | 0.050 | 0.325 |
| `prior_only`（灰图） | 0.200 | 0.200 | 0.100 | 0.050 | 0.138 |

Wilson 95%（correct_image）：0.750 → [0.53, 0.89]；0.100 → [0.03, 0.30]。
**每格 20 条，区间宽到只能当"方向性信号"，不能当结论。**

### 3.1 正式读数（237 步、EOS 监督，`run-id = sft_dev`）

| condition | 易·一致 | 易·冲突 | 难·一致 | 难·冲突 | overall |
|---|---:|---:|---:|---:|---:|
| `correct_image` | **0.950** | **0.000** | **1.000** | **0.100** | 0.512 |
| `mismatched_image` | 0.950 | 0.050 | 1.000 | 0.050 | 0.512 |
| `prior_only`（灰图） | 0.100 | 0.150 | 0.150 | 0.150 | 0.138 |

correct 与 mismatched 逐样本配对：80 对里各只有 1 对翻转，**McNemar 精确检验 p = 1.0**。
EOS 监督带来的变化是"一致格从 0.75 涨到 0.95/1.00"，而不是"开始看图"——
冲突格几乎没动（0.10 / 0.05 仍然是噪声水平）。

### 怎么读这张表（诚实版）

1. **`mismatched_image` 与 `correct_image` 几乎逐格相同**（正式读数 0.950/0.950、0.512/0.512，
   McNemar p = 1.0）。换掉图，答案几乎不变 → 模型的决定主要来自问题与先验，不是图像内容。
2. `prior_only` 掉到 0.1–0.15：把图换成灰图后，连"一致格"都答不对了。
   这说明模型**不是完全忽略视觉输入**，而是把图当作"确认先验的线索"：
   有图时它按"图与先验是否吻合"的隐式模式作答，一旦图变成分布外的灰图就崩。
3. `acuity_effect = −0.050`（CI [−0.167, 0.000]）：易/难两档**没有正向差异**——
   难格甚至略高。在 237 步、实测字形像素 3–9px 的范围里，模型还没到"能分辨字号"的阶段；
   这是 S5 分辨率课程要回答的问题，不是 S4 能下结论的。

> 这三条写进简历时，必须带"pilot、每格 20 条、CI 很宽"的限定语。
> 把它们写成"我们发现模型不看图"就是造假——复试老师一问样本量就会露。

### E5：换画风（`stage4_eval/ood.py`）

同一批题、同一张表的数值，只把图重画成"训练时没见过"的样子
（衬线字体、四边全框、双轴虚线网格、斜刻度、另一套配色；粒度不变，
实测字形像素与 in-domain 差 ≤ 1px）。所以这是**同实例配对**，可以直接 McNemar。

| | acc | 
|---|---:|
| in-domain `sft_dev` | 0.512 |
| OOD `sft_ood` | **0.000** |

80 对里只有 in-domain 对 41 条、OOD 对 0 条（p = 9.1e-13），
而且 **97.5% 的 OOD 输出是空串**——不是"看错图"，是"根本不出话"。

> 这条的教训：pilot 的 237 步、3.38M 可训练参数，学到的是**训练分布的表面统计**
> （字体、配色、坐标轴样式），而不是"这组柱子里哪根最高"。
> 想让它稳，需要的不是更大模型，是更多样式覆盖——这正是 S5 反事实增广的动机。

---

## 4. 代码与命令

```bash
python -m stage4_eval.run --model prior --suite dev_test \
    --conditions correct_image,mismatched_image,prior_only,image_only --run-id prior_dev
python -m stage4_eval.run --model sft --suite dev_test \
    --conditions correct_image,prior_only,mismatched_image --run-id sft_dev
python -m stage4_eval.report --run-id sft_dev --baseline prior_dev
python -m stage1_data.ood                                # E5：同批题换画风
python -m stage4_eval.run --model sft --suite ood_style --run-id sft_ood
python -m stage4_eval.ood --in-domain sft_dev --ood sft_ood
python -m stage4_eval.verify_stage4
```

产物：

```
outputs/eval/<run_id>/predictions.jsonl   # 每条预测：raw/norm/两套判分/condition/family
outputs/eval/<run_id>/records.json        # 按 (condition, 格) 聚合：n/acc/Wilson/族数
reports/eval_<run_id>.md                  # 四格 + 效应量 + bootstrap + McNemar
```

---

## 5. 统计口径（别写错）

- **单格**用 Wilson CI（小样本下比正态近似靠谱）；
- **效应量**用 family 整簇 bootstrap（同一族的样本不独立，不能逐条重采样）；
- **同实例比较**（例如干预前后）用 McNemar 精确检验，并报不一致对数
  （`a_only` / `b_only`）——只报 p 值不报不一致对数是不完整的；
- 主测试集（`locked_test`）只用于**预注册的主终点**，不要拿它调参；
- 判分用两套：严格（逐字）与宽松（数值容差 5%）。报告主表用宽松，但两套都存。

---

## 6. 作业

### HW4-1 · 自己实现 Wilson 区间（必做）

`homework/hw1_wilson.py`：实现 `wilson(k, n, z=1.96)`，并与 `picovlm.stats.wilson_ci` 对上；
边界检查 `k=0`、`k=n`、`n=0`。

### HW4-2 · 论证"为什么不能逐条 bootstrap"（必做）

`homework/hw2_cluster_bootstrap.py`：构造 40 个 family、每族 2 条高度相关的样本
（同族两条正确性相同），分别做"逐条重采样"和"整簇重采样"，比较两者 CI 宽度。
在报告里写一句结论：哪种会**低估**不确定性，为什么。

### HW4-3 · 手工核对一条预测（选做）

从 `predictions.jsonl` 里挑一条 `correct_image` 的错例，回到图上看：
是字号看不清、问题理解错、还是先验压过图。写 3 行分析。

---

## 7. 验收标准

`python -m stage4_eval.verify_stage4` 必须 0 FAIL：

- `prior-eval`：80×4=320 条预测，四种 condition 齐全；
- `two-by-two-direction`：盲基线的 `prior_effect` 必须 ≈ +1.0（构造自洽性检查）；
- `records-ci`：`ci_low ≤ acc ≤ ci_high`，且族数 > 0；
- `family-bootstrap`：效应量的 CI 上下界不倒置；
- `mcnemar`：自己比自己 p=1、不一致对数 0；
- `report-written`：报告里有 `prior_effect` 与 Wilson；
- `sft-run`：真模型评测跑过（没有则 SKIP）。

---

## 8. 延伸

- Wilson (1927) score interval 的原始推导；
- McNemar 精确检验（二项检验而非卡方近似）为什么在小样本更稳；
- 聚类 bootstrap（cluster bootstrap）与 GEE 的关系。
