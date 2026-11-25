"""统一的 verify 小框架：让每个阶段的验收输出长得一样。

    from picovlm.report import run, Skip
    run([("items-ok", check_items), ...])

规则：检查函数返回 str 作为详情；抛 `Skip` 记为 SKIP；抛别的异常记为 FAIL。
任何一个 FAIL，进程退出码为 1（`run_all` 靠这个聚合）。
"""

from __future__ import annotations

import traceback
from typing import Callable

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class Skip(Exception):
    """环境不具备，跳过（不算失败）。"""


def run(checks: list[tuple[str, Callable[[], str]]], title: str = "") -> int:
    rows: list[tuple[str, str, str]] = []
    for name, fn in checks:
        try:
            detail = fn() or ""
            rows.append((name, PASS, str(detail)))
        except Skip as exc:
            rows.append((name, SKIP, str(exc)))
        except Exception as exc:  # noqa: BLE001
            rows.append((name, FAIL, f"{type(exc).__name__}: {exc}"))
            if isinstance(exc, AssertionError):
                continue
            traceback.print_exc(limit=2)
    if title:
        print(f"== {title} ==")
    for name, status, detail in rows:
        print(f"[{status:<4}] {name:<26} {detail}")
    n_fail = sum(1 for _, s, _ in rows if s == FAIL)
    n_skip = sum(1 for _, s, _ in rows if s == SKIP)
    n_pass = len(rows) - n_fail - n_skip
    print(f"-- PASS {n_pass} / FAIL {n_fail} / SKIP {n_skip} --")
    return 1 if n_fail else 0


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def approx(a: float, b: float, tol: float, name: str = "") -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{name or 'value'} 期望 {b}±{tol}，实际 {a}")
