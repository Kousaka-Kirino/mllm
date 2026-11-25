"""S8 验收：CLI、README、交付材料骨架、run_all 接线是否正确。

注意：**不检查 acceptance.md 是否已存在**——那是 `run_all` 自己的产物，
由它跑完后写；在这里检查会形成"先有鸡还是先有蛋"。存在时顺带校验一致性。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

from picovlm.config import OUTPUTS, ROOT
from picovlm.report import Skip, require, run


def main() -> int:
    def stages_wired() -> str:
        from stage8_system.run_all import STAGES
        require(len(STAGES) >= 9, f"阶段数不足：{len(STAGES)}")
        names = [s[0] for s in STAGES]
        require(names == sorted(names) and names[0] == "S0", "阶段顺序不对")
        for _, module, _ in STAGES:
            require(importlib.util.find_spec(module) is not None, f"{module} 找不到")
        return f"{len(STAGES)} 个阶段全部可导入"

    def cli_works() -> str:
        proc = subprocess.run(
            [sys.executable, "-m", "stage8_system.cli", "--image",
             "data/images/synth/synth-test-position-00001.png",
             "--ask", "Which category is largest?", "--policy", "first_option"],
            capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT))
        out = (proc.stdout or "") + (proc.stderr or "")
        require(proc.returncode == 0, f"CLI 退出码 {proc.returncode}\n{out[-400:]}")
        require("A:" in out, "CLI 没有输出答案")
        return "CLI 输出正常"

    def acceptance_if_present() -> str:
        md, js = OUTPUTS / "acceptance.md", OUTPUTS / "acceptance.json"
        if not md.exists() or not js.exists():
            raise Skip("首次运行，acceptance 还没有生成（run_all 会写）")
        d = json.loads(js.read_text(encoding="utf-8"))
        rows = d["rows"]
        backs = sum(r["pass"] for r in rows)
        require(backs == d["total"]["pass"], "合计与明细不符")
        return f"{len(rows)} 个阶段 / {backs} 项检查"

    def deliverable_skeleton() -> str:
        d = ROOT / "stage9_deliver"
        files = sorted(d.glob("*.md")) if d.exists() else []
        require(len(files) >= 3, f"交付材料模板不足：{[f.name for f in files]}")
        return " / ".join(f.name for f in files[:4])

    def readme() -> str:
        md = ROOT / "README.md"
        require(md.exists(), "根 README.md 缺失")
        text = md.read_text(encoding="utf-8")
        for kw in ("run_all", "acuity", "2×2"):
            require(kw in text, f"README 缺少关键词 {kw!r}")
        return f"{len(text)} 字符"

    return run([
        ("stages-wired", stages_wired),
        ("cli-works", cli_works),
        ("acceptance-if-present", acceptance_if_present),
        ("deliverable-skeleton", deliverable_skeleton),
        ("readme", readme),
    ], title="stage8_system")


if __name__ == "__main__":
    sys.exit(main())
