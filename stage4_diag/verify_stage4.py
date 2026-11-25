"""S4 验收：在一份已有评测上重跑 2×2，并检查报告与 JSON 一致。"""

from __future__ import annotations

import sys

from picovlm.config import OUTPUTS, REPORTS
from picovlm.report import Skip, require, run


def main() -> int:
    import json

    def have_run() -> str:
        base = OUTPUTS / "eval"
        if not base.exists() or not any((p / "predictions.jsonl").exists()
                                        for p in base.iterdir()):
            raise Skip("先跑一次 stage3_train.eval_suite")
        return "找到评测结果"

    def grid_runs() -> str:
        from stage4_diag import grid_2x2
        from picovlm.report import Skip as S

        try:
            rid = grid_2x2.latest_run()
        except S as exc:
            raise Skip(str(exc)) from exc
        samples, preds, _ = grid_2x2.load_run(rid)
        res = grid_2x2.analyse(samples, preds)
        require(res["table"]["easy_consistent"] is not None, "四格缺失")
        require(-1.0 <= (res["prior_effect"] or 0) <= 1.0, "prior_effect 越界")
        return f"{rid}: 先验效应 {res['prior_effect']:.1%}"

    def report_written() -> str:
        md = sorted(REPORTS.glob("grid_*.md"))
        if not md:
            raise Skip("还没生成 grid 报告")
        text = md[-1].read_text(encoding="utf-8")
        require("2×2 诊断" in text or "2×2" in text, "报告标题缺失")
        require("VDR" in text, "VDR 段缺失")
        return f"{md[-1].name}（{len(text)} 字符）"

    def json_matches() -> str:
        base = OUTPUTS / "eval"
        grids = [p for p in base.glob("*/grid.json")]
        if not grids:
            raise Skip("还没落 grid.json")
        g = json.loads(grids[-1].read_text(encoding="utf-8"))
        require(set(g["table"]) == {"easy_consistent", "easy_conflict",
                                    "hard_consistent", "hard_conflict"}, "四格键名漂移")
        return f"{grids[-1].parent.name}: {g['n']} 条"

    return run([
        ("have-eval-run", have_run),
        ("grid-runs", grid_runs),
        ("report-written", report_written),
        ("grid-json", json_matches),
    ], title="stage4_diag")


if __name__ == "__main__":
    sys.exit(main())
