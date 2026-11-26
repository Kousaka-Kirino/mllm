# S9 答案与验收标准

## HW1：Δ + 噪声判据

```python
def load_runs(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    ev = d.get("eval", {})
    items = ev.items() if isinstance(ev, dict) else [(str(r.get("run")), r) for r in ev]
    out = {}
    for run, r in items:
        if not isinstance(r, dict):
            continue
        for suite, m in (r.get("suites") or r.get("records") or {}).items():
            if isinstance(m, dict) and m.get("accuracy") is not None:
                out[(run, suite)] = m
    return out

def delta_table(prev, cur):
    rows = []
    for key in set(prev) & set(cur):
        a, b = prev[key], cur[key]
        p, c = float(a["accuracy"]), float(b["accuracy"])
        ph = (float(a.get("ci_high", p)) - float(a.get("ci_low", p))) / 2
        ch = (float(b.get("ci_high", c)) - float(b.get("ci_low", c))) / 2
        rows.append({"run": key[0], "suite": key[1], "prev": p, "cur": c,
                     "delta": c - p, "significant": abs(c - p) > ph + ch})
    return sorted(rows, key=lambda r: -abs(r["delta"]))
```

要点：
- **不要用 Δ > 0 当"提升"**。80 条样本的 Wilson 半宽约 ±0.10，Δ 小于这个数就是噪声。
- `prev_hw + cur_hw` 是乐观下界（两区间独立时更接近 `sqrt(ph²+ch²)`），
  但作为"粗判"够用；要严格就上 McNemar（S4 已经会）。
- 同一份文件跟自己比：Δ 恒为 0，`0 > ph+ch` 为假 → 全部不显著。这就是验收用例。

验收：输出为空或全 0 Δ、无"显著"标记；换上一份真实旧结果后能看出差异项。

## HW2：6 页大纲

```python
def outline(res):
    ev = res.get("eval", {})
    def cell(suite):
        r = ev.get("sft_dev") or {}
        return (r.get("suites") or {}).get(suite, {}).get("accuracy")
    mem = res.get("memory", {})
    return [
        {"title": "问题：它到底看图了吗",
         "bullets": ["acuity 与 prior 两种失败模式", "用 2×2 设计把二者拆开"]},
        {"title": "方法：0.5B 底座 + connector 对齐 + LoRA",
         "bullets": [f"视觉 token {res.get('bench', [{}])[0].get('tokens', 576)}",
                     "只训 2.3M connector + 1.08M LoRA"]},
        {"title": "主结果：一致格高、冲突格零",
         "bullets": [f"易·一致 {cell('easy_consistent'):.3f}",
                     f"难·一致 {cell('hard_consistent'):.3f}",
                     f"易·冲突 {cell('easy_conflict'):.3f}",
                     f"难·冲突 {cell('hard_conflict'):.3f}"]},
        {"title": "干预：GRPO 在 8GB 上组内几乎无信号",
         "bullets": ["零优势组 5/6", "原因：组内同分"]},
        {"title": "负结果：视觉记忆与放大都没救回来",
         "bullets": [f"recall@1 {mem.get('recall_at_1', 0):.3f}",
                     "salient − full = -0.042"]},
        {"title": "结论",
         "bullets": ["先验主导，视觉是可被绕过的旁路",
                     "acuity 不是瓶颈 → 分辨率课程不值得做",
                     "没做到：8GB 上跑不动有效 GRPO（诚实负结果）"]},
    ]
```

要点：第 6 页必须有一条"没做到"。幻灯片最容易被删掉的就是负结果，
而面试官问得最多的恰恰是它。

验收：6 个 `## `、第 3 页含四格数字、第 6 页含"没做到"。
