"""S8 验收：一键矩阵、CLI、交付文档齐不齐。

有一点刻意不做：**不检查 acceptance.md 是否"全绿"**——那会让 verify 依赖
run_all 的产物，形成先有鸡还是先有蛋。这里只检查"接线正确 + 产物结构合法"。

CLI 用 `--model prior`（盲基线，不加载 1.5GB 权重、不占显存），
这样 S8 在任何机器上都能跑；真模型的端到端在 S4 里测。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

from picovlm.config import OUTPUTS, ROOT
from picovlm.report import Skip, require, run


def stages_wired() -> str:
    from stage8_system.run_all import STAGES
    require(len(STAGES) >= 10, f"阶段数不足：{len(STAGES)}")
    names = [s[0] for s in STAGES]
    require(names == sorted(names) and names[0] == "S0", "阶段顺序不对")
    require(names[-1] == "S9", "最后一个阶段应是 S9")
    for _, module, _ in STAGES:
        require(importlib.util.find_spec(module) is not None, f"{module} 找不到")
    return f"{len(STAGES)} 个阶段全部可导入"


def cli_works() -> str:
    imgs = sorted((ROOT / "data" / "synth" / "images").glob("*.png"))
    if not imgs:
        raise Skip("没有图片（先跑 stage1_data.build）")
    import os
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"}
    proc = subprocess.run(
        [sys.executable, "-m", "picovlm.cli", "--image", str(imgs[0]),
         "--ask", "Which category has the highest value?", "--model", "prior"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), env=env)
    out = (proc.stdout or "") + (proc.stderr or "")
    require(proc.returncode == 0, f"CLI 退出码 {proc.returncode}\n{out[-400:]}")
    require("[prior]" in out, "CLI 没有输出答案")
    require("盲基线" in out, "CLI 应提示这是盲基线")
    return "CLI（prior 模式）输出正常"


def accept_if_present() -> str:
    md, js = OUTPUTS / "acceptance.md", OUTPUTS / "acceptance.json"
    if not md.exists() or not js.exists():
        raise Skip("首次运行，acceptance 还没有生成（run_all 会写）")
    d = json.loads(js.read_text(encoding="utf-8"))
    rows = d["rows"]
    require(sum(r["pass"] for r in rows) == d["total"]["pass"], "合计与明细不符")
    require(len(rows) >= 9, "矩阵阶段数不足")
    return f"{len(rows)} 个阶段 / {d['total']['pass']} 项 PASS"


def docs_present() -> str:
    need = ["README.md", "RUNBOOK.md", "REPORT.md"]
    missing = [f for f in need if not (ROOT / f).exists()]
    require(not missing, f"缺顶层文档：{missing}")
    claims = ROOT / "stage9_report" / "CLAIMS.md"
    require(claims.exists(), "缺 stage9_report/CLAIMS.md")
    return "README / RUNBOOK / REPORT / CLAIMS 齐全"


def stage_docs() -> str:
    """每个 stage 都必须有 README 讲解；homework 目录允许后面补。"""
    need = ["stage0_env", "stage1_data", "stage2_model", "stage3_train",
            "stage4_eval", "stage5_intervene", "stage6_ablation",
            "stage7_memory", "stage8_system", "stage9_report"]
    missing = [d for d in need if not (ROOT / d / "README.md").exists()]
    require(not missing, f"缺阶段 README：{missing}")
    return f"{len(need)} 个阶段 README 齐全"


def readme_updated() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    for kw in ("run_all", "acuity", "2×2", "stage9_report"):
        require(kw in text, f"README 缺少关键词 {kw!r}")
    return f"{len(text)} 字符"


def main() -> int:
    return run([
        ("stages-wired", stages_wired),
        ("cli-works", cli_works),
        ("acceptance-if-present", accept_if_present),
        ("docs-present", docs_present),
        ("stage-docs", stage_docs),
        ("readme", readme_updated),
    ], title="S8 验收（2.0）")


if __name__ == "__main__":
    sys.exit(main())
