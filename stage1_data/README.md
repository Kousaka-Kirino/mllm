# Stage 1 · 数据引擎

> 这一阶段的成品是"一份能拿来做因果结论的数据"，不是"一堆图片"。
> 如果这一关做错，后面所有漂亮的数字都是假的。

---

## 1. 为什么数据阶段最难

多模态项目里，模型代码通常只占 20% 的工作量，剩下 80% 在数据。原因很朴素：
**你观察到的是"模型答对了/答错了"，但你想知道的是"为什么"**。如果数据里
"答案"和"某个捷径"高度相关，模型答对可能是因为它真的看了图，也可能只是
背下了捷径——而你的测试集如果同分布，就永远分不出来。

所以 S1 的目标不是造更多图，而是造出**能被区分**的图：

- 有一组题，**只有看图才能答对**（图像充分条件）；
- 有一组题，**不看图按先验也能答对**（先验充分条件）；
- 这两组题在图片风格、题型、答案分布上尽量一致，**只差"先验是否成立"**。

这就是本项目的 2×2 诊断格（acuity × prior）的数据基础。

---

## 2. 三个必须钉死的东西

### 2.1 字形像素要"实测"，不要"解析算"

v1 里我们用过解析式：9pt × font_scale × 384/432 = 8×font_scale。它只对
"字号盒"成立，而模型真正能分辨的是**着墨高度**（数字的实心笔画有多高）。
实测下来，同一字号盒的实际着墨高度只有名义值的约 0.75 倍，而且不同字体的
数字/字母高度也不一样。

2.0 的做法（`stage1_data/measure.py`）：

1. `fig.findobj(Text)` 拿到每个刻度/标注文字的 bbox；
2. 把 bbox 映射回图像像素坐标，裁出文字块；
3. 二值化（`INK_THRESHOLD=200`），数"有墨水的行数" → 着墨高度；
4. 同时记录**渲染尺度**和**模型输入尺度**（×384/size）两个数字。

`reports/acuity_spotcheck/` 里会导出裁剪图 + 表格，供人工抽查——
这一步不能省，因为测量函数本身也可能错。

### 2.2 分箱阈值是"预注册"的，不是"看数据切的"

`stage1_data/build.py` 顶部有三个常量：

```python
EASY_MIN_PX = 6.5      # 模型输入侧着墨高 ≥ 6.5px：看得见
MID_MIN_PX  = 4.5
EXTREME_MAX_PX = 3.0   # < 3px：低于可测范围，单独一档不进主比较
```

它们写进 `data/acuity_thresholds.json` 后**冻结**，full 构建会复用同一份
（`--recalibrate` 才能改）。这样"easy/hard"是绝对定义，不会因为换了数据集
边界就漂移，跨规模的数字才可比。

### 2.3 family 划分：防的不是"图片重复"，是"表格重复"

同一条底层数值表，可以问出 6 种问题、改 4 种先验、换 8 档字号。如果这些变体
分散到 train 和 test，测试集就成了"见过的表换个问法"，成绩虚高。

所以每个样本都带 `family_id = hash(表种子 + 形状 + 数值)`，划分时保证：

- 一个 family 只出现在一个 split（`validate` 会逐条查）；
- `dev_test` / `locked_test` 每个 2×2 格**至少 200 个独立 family**（full 档）；
- 成对的反事实（同一张图两种问法）整体落在同一 split。

---

## 3. 数据长什么样

四种图表风格：单系列柱状（`bar`）、多系列分组柱（`grouped_bar`）、折线（`line`），
真值都来自一个 `np.ndarray` 数值表，所以答案永远可复算。

六种题型：

| qtype | 问法 | 答案类型 | 用到的先验 |
|---|---|---|---|
| `value` | A 类别的值是多少 | number | 无（中性题） |
| `compare` | A 比 B 高吗 | bool | marginal（yes/no 分布） |
| `count` | 有几根柱子超过 50 | number | numeric（爱答 2） |
| `maxcat` | 哪个类别最高 | label | position（首/末位置） |
| `order` | A 和 B 谁更高（两种问法） | label | order（先提到的选项） |
| `maxseries` | 哪个系列在 A 上最高 | label | 无（中性题） |

训练集按 **ρ（先验强度）** 人为偏斜：`ρ=0.7` 表示"有先验的题里 70% 的先验
答案正好正确"。测试集四格精确平衡，所以 ρ 是**可扫描的实验变量**，不是宿命。

---

## 4. 代码导览

```
stage1_data/
├── render.py      # ChartSpec → PIL 图。6in × dpi 的坑在这里被固定
├── make.py        # 六种题型的生成器；共用同一张底层表
├── conflict.py    # 四类先验的成对构造（consistent / conflict）
├── measure.py     # 实测字形像素 + 人工核验材料
├── acuity.py      # 字号阶梯（S4/S5 用）
├── build.py       # 生成 4 个 split + datacard + hashes（唯一入口）
├── validate.py    # 13 条数据断言
└── verify_stage1.py  # 验收：validate 全过 + 渲染确定性 + 配置一致
```

跑一遍：

```bash
python -m stage1_data.build --profile pilot --seed 17      # 1-2 分钟
python -m stage1_data.validate --profile pilot
python -m stage1_data.measure --spotcheck 12               # 导出核验图
python -m stage1_data.verify_stage1 --profile pilot
```

`--profile full` 大约 5000 张图、十几分钟，正式跑实验前必须做一次。

---

## 5. 怎么读产物

`data/datacard.json` 是一份"数据身份证"，重点看四个字段：

- `rho_measured`：实测先验强度，和目标差超过 0.05 就要查生成器；
- `families`：每个 split 的独立表数量——这是**有效样本量**，不是行数；
- `glyph_px_model`：实测像素的中位数/IQR，确认真的覆盖了 easy↔hard；
- `acuity_thresholds`：冻结的阈值，后面所有报告都引用这一份。

`data/hashes.json` 记录每个产物的 sha256；改了任何生成逻辑，哈希就会变，
避免"用旧数据跑新代码"这种最难查的错。

---

## 6. 作业

### HW1 · 自己实现 family 划分（必做）

打开 `homework/hw1_family_split.py`，实现 `split_by_family(families, ratios, seed)`：
给定 family 列表和目标比例，返回 `{split: [family...]}`，要求

1. 同一 family 只出现一次；
2. 各 split 的比例误差 ≤ 2%；
3. 同 seed 可复现。

答案与验收标准见 `homework/HW1_提示.md`。

### HW2 · 自己写一个"字形高度"测量（必做）

打开 `homework/hw2_acuity_probe.py`，实现 `ink_height(patch, thr=200)`：
输入一个灰度文字块（numpy 数组），返回"最长的连续有墨行数"。
再实现 `scale_to_model(px_render, render_size, input_size=384)`。

这题的意义：你会发现"着墨高度"对阈值敏感——阈值调 10 个灰度值，
小字号的测量结果能差 1px。所以正式数据里必须固定阈值并记录。

### HW3 · 抽查 10 条样本（选做，但推荐）

随机抽 10 条 test 样本，打开图片，人工回答，再和标注对比。
把不一致的记进 `homework/HW3_核对.md`。这是最便宜的数据质检。

---

## 7. 验收标准

`python -m stage1_data.verify_stage1` 必须 **0 FAIL**，其中这几条最关键：

- `answers-recomputable`：答案能由底层表重算；
- `family-no-leak`：没有 family 跨 split；
- `test-balanced`：2×2 四格均衡且族数达标；
- `acuity-measured` + `acuity-geometry`：像素是实测的、随字号单调；
- `thresholds-frozen`：阈值文件与 datacard 一致。

> 注意：pilot 档的族数（每格 20/40）**不够下结论**，只够验证流程。
> 正式结论必须用 `--profile full` 重新构建，并在报告里注明 profile。

---

## 8. 延伸阅读

- 为什么"合成图表"能研究真实问题：PlotQA / DVQA 就是用底层数据表生成
  QA 对，可控性和可复算性是它们相对真实图表数据集的最大优势。
- 关于捷径学习（shortcut learning）的系统讨论：Geirhos et al., *Shortcut
  Learning in Deep Neural Networks*, 2020。
