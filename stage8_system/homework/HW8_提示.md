# S8 答案与验收标准

## HW1：`--only S3,S4`

```python
def filter_stages(stages, only):
    if only is None:
        return list(stages)
    want = {s.strip().upper() for s in only.split(",") if s.strip()}
    unknown = want - {s[0] for s in stages}
    if unknown:
        raise ValueError(f"未知阶段：{sorted(unknown)}；可选 {[s[0] for s in stages]}")
    out = [s for s in stages if s[0] in want]
    if not out:
        raise ValueError("过滤后没有阶段可跑")
    return out
```

要点：
- `upper()` 处理 `s3`；`strip` 处理 `S3, S4` 里的空格。
- **保留原顺序**，不按用户输入顺序——否则 S9 可能跑在 S3 前面，读到旧结果。
- 未知名字必须报错：静默忽略会让你以为"跑了 S9"，其实什么都没跑。

验收：`python -m stage8_system.run_all --only S4` 只打印 S4 的检查；
`--only S99` 退出码非 0 且消息里含 `S99`。

## HW2：通用证据链

```python
def load_numbers(paths):
    known = set()
    def walk(o):
        if isinstance(o, dict):
            for v in o.values(): walk(v)
        elif isinstance(o, list):
            for v in o: walk(v)
        elif isinstance(o, (int, float)) and not isinstance(o, bool):
            known.add(f"{abs(float(o)):.3f}")
    for p in paths:
        walk(json.loads(Path(p).read_text(encoding="utf-8")))
    return known

def trace(md_text, known):
    found = {x.lstrip("-") for x in NUM.findall(md_text)}
    return found & known, found - known

def check_report(md_path, evidence_paths):
    known = load_numbers(evidence_paths)
    bad, _ = trace(Path(md_path).read_text(encoding="utf-8"), known)
    # 注意 trace 返回 (ok, bad)
    require(not bad, f"不可追溯：{sorted(bad)}")
    print(f"PASS: {len(known)} 个证据数，报告全部可追溯")
    return True
```

三个要点（也是三个坑）：

1. **`bool` 是 `int` 的子类**。不排除的话 `True` 会变成 `1.000`，混进证据集。
2. **`abs()`**。报告里写 `-0.042`，json 里是 `-0.042`，但 `f"{v:.3f}"` 已经带负号；
   统一取 `abs` 后比对，才不会因为符号写法不一致而误报。
3. **这是"必要性"检查，不是"充分性"检查**。数字对得上，不代表这句话的
   因果结论对——它只防"手写了一个不存在的数"。所以它必须和 CLAIMS.md 的
   "数字 ↔ 文件"对照表一起用。

验收：`python homework/hw2_evidence_chain.py` → `PASS`；把报告里某个 0.xxx
改成 0.999 后再跑 → 报出 `0.999` 且退出码非 0。
