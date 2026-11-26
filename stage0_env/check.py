"""S0 环境体检的 CLI 入口（2.0 规划里的 `python -m stage0_env.check`）。

真正的检查逻辑在 `check_env.collect()`；这里只负责参数、落盘位置和退出码。
这样"检查什么"和"结果写到哪"分开，后面加新的检查项不用动 CLI。
"""

from __future__ import annotations

import argparse
import sys

from picovlm import io_utils as io
from picovlm.config import OUTPUTS, ROOT

from .check_env import Check, collect, to_markdown


def run(output: str = "outputs/env_snapshot.json", net: bool = True,
        write_md: bool = True) -> tuple[list[Check], dict]:
    checks, snap = collect(net=net)
    snap = {**snap, "checks": [c.__dict__ for c in checks]}
    io.write_json(ROOT / output, snap)
    if write_md:
        (ROOT / "ENVIRONMENT.md").write_text(
            to_markdown(checks, snap), encoding="utf-8", newline="\n")
    return checks, snap


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="stage0_env.check",
                                 description="环境与硬件体检，写 ENVIRONMENT.md")
    ap.add_argument("--output", default=str(OUTPUTS / "env_snapshot.json"),
                    help="快照 JSON 路径（相对项目根）")
    ap.add_argument("--no-net", action="store_true", help="跳过网络探测")
    ap.add_argument("--no-md", action="store_true", help="不重写 ENVIRONMENT.md")
    args = ap.parse_args(argv)

    checks, _ = run(output=args.output, net=not args.no_net,
                    write_md=not args.no_md)
    for c in checks:
        print(f"[{c.status:<4}] {c.name:<16} {c.detail}")
    fails = [c for c in checks if c.status == "FAIL"]
    print(f"\n[check] {len(checks) - len(fails)}/{len(checks)} 通过，FAIL {len(fails)}")
    if fails:
        print("[check] 先修环境，再进 S0 的门槛：", ", ".join(c.name for c in fails))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
