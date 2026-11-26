"""S0 验收：环境快照与 ENVIRONMENT.md 的一致性、契约常量冻结。

这一关只回答一个问题：**这台机器的状态被如实记录了吗？**
任何"我记得是 24GB 内存"的口头描述都会在这里被数字打回。
"""

from __future__ import annotations

import json
import sys

from picovlm.config import OUTPUTS, ROOT
from picovlm.report import require, run


def snapshot() -> dict:
    path = OUTPUTS / "env_snapshot.json"
    if not path.exists():
        raise AssertionError("outputs/env_snapshot.json 不存在，先跑 python -m stage0_env.check_env")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    def snapshot_readable() -> str:
        d = snapshot()
        need = {"python", "platform", "gpu", "packages", "checks", "disk_gb"}
        missing = need - set(d)
        require(not missing, f"快照缺字段 {missing}")
        return f"{len(d['checks'])} 条检查"

    def no_fail() -> str:
        d = snapshot()
        fails = [c for c in d["checks"] if c["status"] == "FAIL"]
        require(not fails, f"有 {len(fails)} 项 FAIL：{[c['name'] for c in fails]}")
        return "没有 FAIL 项"

    def env_md_matches() -> str:
        md = (ROOT / "ENVIRONMENT.md")
        require(md.exists(), "ENVIRONMENT.md 不存在")
        text = md.read_text(encoding="utf-8")
        require("自动生成" in text, "缺少'自动生成'声明（会被误手改）")
        d = snapshot()
        require(str(d["ram_gb"])[:4] in text, "内存数字与快照不一致")
        return f"{len(text)} 字符，与快照同源"

    def gpu_recorded() -> str:
        g = snapshot()["gpu"]
        if not g:
            return "无 GPU（快照已如实记录）"
        require(int(g.get("total_mib") or 0) > 0, "显存没记下来")
        return f"{g.get('name')} {g.get('total_mib')}MiB (free {g.get('free_mib')}MiB)"

    def disk_recorded() -> str:
        d = snapshot()["disk_gb"]
        require(isinstance(d, dict) and d, "磁盘信息缺失")
        return " / ".join(f"{k}:{v}GB" for k, v in list(d.items())[:3])

    def schema_frozen() -> str:
        from picovlm.schema import ANSWER_KINDS, PRIOR_TYPES, SPLITS
        require(SPLITS == ("train", "dev", "test"), "split 被改动")
        require(ANSWER_KINDS == ("number", "text", "bool", "label"), "answer_kind 被改动")
        require(PRIOR_TYPES == ("none", "position", "marginal", "numeric", "order"),
                "prior 类型被改动")
        return "SPLITS / ANSWER_KINDS / PRIOR_TYPES 与 v1.0 契约一致"

    def package_clean_import() -> str:
        import subprocess
        code = ("import sys; import picovlm; "
                "assert 'torch' not in sys.modules, 'picovlm 顶层不应 import torch'")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, encoding="utf-8", cwd=str(ROOT))
        require(r.returncode == 0, f"顶层引入了 torch：{r.stderr[-300:]}")
        return "picovlm 顶层无 torch 依赖"

    def configs_loadable() -> str:
        """2.0 的规矩：实验参数只从 configs/ 来，所以每份配置都必须能 load 出来。"""
        from picovlm.config import CONFIGS, load_config
        files = sorted(CONFIGS.glob("*.yaml"))
        require(len(files) >= 8, f"配置太少：{len(files)} 份")
        for f in files:
            cfg = load_config(f.name)
            require(isinstance(cfg, dict) and cfg, f"{f.name} 解析为空")
        return f"{len(files)} 份配置全部可解析（含 base 链）"

    def benchmark_recorded() -> str:
        """硬件基准是 L0 的硬证据：必须有 load 和 infer 两个 suite 的实测行。"""
        import glob
        files = sorted(glob.glob(str(OUTPUTS / "bench" / "bench_*.json")))
        require(files, "没有 outputs/bench/bench_*.json，先跑 python -m stage0_env.benchmark")
        data = json.loads(open(files[-1], encoding="utf-8").read())
        rows = data.get("rows", [])
        suites = {r["suite"] for r in rows}
        require({"load", "infer"} <= suites, f"bench 缺 suite：{suites}")
        require(any(r["peak_alloc_mib"] > 0 for r in rows),
                "全是 dry-run 数据（peak=0），要跑一次真实基准")
        return f"{len(files)} 次基准，最新 {len(rows)} 行，dry_run={data.get('dry_run')}"

    def homework_present() -> str:
        need = ["README.md", "HW0_提示.md", "hw1_vram_budget.py", "hw2_idle_free.py"]
        base = ROOT / "stage0_env" / "homework"
        missing = [n for n in need if not (base / n).exists()]
        require(not missing, f"作业缺文件 {missing}")
        return f"{len(need)} 个作业文件齐全"

    return run([
        ("snapshot-readable", snapshot_readable),
        ("no-fail-checks", no_fail),
        ("env-md-matches", env_md_matches),
        ("gpu-recorded", gpu_recorded),
        ("disk-recorded", disk_recorded),
        ("schema-frozen", schema_frozen),
        ("configs-loadable", configs_loadable),
        ("benchmark-recorded", benchmark_recorded),
        ("homework-present", homework_present),
        ("package-import", package_clean_import),
    ], title="stage0_env")


if __name__ == "__main__":
    sys.exit(main())
