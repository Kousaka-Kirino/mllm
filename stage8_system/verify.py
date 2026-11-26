"""S8 · 分级验收（规划 2.0 的 `python -m stage8_system.verify --profile core`）。

`run_all` 只回答"现在能不能过"；这里回答"用哪一档成本去过"，因为在复试现场
（或 CI 里）你不可能每次都等 S2 去下 1.5GB 权重、等 S4 去跑 240 条推理。

- `core`：默认档。跳过需要真权重的 S2（它自己会 SKIP），其余全跑。
- `full`：给 S2 补 `--load`，连真实权重一起验；最慢，也最可信。
- `--only 1 4`：只跑指定阶段，改代码时用。

两个档都会把结果写进 `outputs/acceptance.md` / `acceptance.json`（与 run_all 同格式），
退出码在任一 FAIL 时为 1。
"""

from __future__ import annotations

import argparse
import sys
import time

from picovlm.config import OUTPUTS
from picovlm.io_utils import write_json

from stage8_system.run_all import STAGES, run_stage

#: 每个 profile 给特定阶段追加的参数。
PROFILE_ARGS: dict[str, dict[str, list[str]]] = {
    "core": {},
    "full": {"S2": ["--load"]},
}

PROFILES = tuple(PROFILE_ARGS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="core", choices=list(PROFILES))
    ap.add_argument("--only", nargs="*", default=None, help="只跑这些阶段号，如 1 4")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    wanted = {f"S{x}" for x in args.only} if args.only else {s[0] for s in STAGES}
    extra_by_stage = PROFILE_ARGS[args.profile]
    rows, t0 = [], time.time()
    for stage, module, title in STAGES:
        if stage not in wanted:
            continue
        p, f, s, out = run_stage(module, extra_by_stage.get(stage))
        rows.append({"stage": stage, "module": module, "title": title,
                     "pass": p, "fail": f, "skip": s})
        print(f"{stage} {title:<12} PASS {p:2d} / FAIL {f} / SKIP {s}")
        if f and not args.quiet:
            tail = "\n".join(out.strip().splitlines()[-12:])
            print(f"  ↳ {module}\n" + "\n".join("    " + l for l in tail.splitlines()))

    total = {k: sum(r[k] for r in rows) for k in ("pass", "fail", "skip")}
    md = ["# 验收矩阵（profile: %s）" % args.profile, "",
          f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}，耗时 {time.time() - t0:.1f}s", "",
          "| 阶段 | 内容 | PASS | FAIL | SKIP |", "|---|---|---|---|---|"]
    md += [f"| {r['stage']} | {r['title']} | {r['pass']} | {r['fail']} | {r['skip']} |"
           for r in rows]
    md += ["", f"**合计：{total['pass']} PASS / {total['fail']} FAIL / "
               f"{total['skip']} SKIP**", ""]

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    # --only 是"改代码时快速验一个阶段"，不能把全量矩阵覆盖成一行——
    # verify_stage8 的 acceptance-if-present 会检查矩阵阶段数，覆盖了就会 FAIL。
    name = "acceptance_partial" if args.only else "acceptance"
    (OUTPUTS / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    write_json(OUTPUTS / f"{name}.json",
               {"profile": args.profile, "rows": rows, "total": total})
    print("\n".join(md[-2:]))
    print(f"-> {OUTPUTS / (name + '.md')}")
    return 1 if total["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
