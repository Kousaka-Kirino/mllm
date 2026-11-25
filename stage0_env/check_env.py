"""S0：环境体检。`python -m stage0_env.check_env [--no-net]`

产出：
    ENVIRONMENT.md               人读的环境基线（**自动生成，不要手改**）
    outputs/env_snapshot.json    机读快照，verify_stage0 用它

为什么要有这一步：8GB 显存的项目，成败一半在环境。把"显卡被桌面程序占了 6GB"
这种事写进交付材料，比在报告里写"训练很慢"有用得多。
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from picovlm import __version__
from picovlm import io_utils as io
from picovlm.config import OUTPUTS, ROOT
from picovlm.seeds import SEED_BANDS

PACKAGES = ("torch", "transformers", "peft", "accelerate", "safetensors",
            "numpy", "PIL", "matplotlib", "pytest", "datasets", "trl",
            "bitsandbytes", "gradio")
MIRRORS = ("https://hf-mirror.com", "https://www.modelscope.cn")


@dataclass
class Check:
    name: str
    status: str            # PASS | WARN | SKIP
    detail: str = ""
    data: dict = field(default_factory=dict)


def _ram_gb() -> float | None:
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return round(st.ullTotalPhys / 2**30, 1)
    except Exception:
        pass
    return None


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:
        return None


def collect(net: bool = True) -> tuple[list[Check], dict]:
    checks: list[Check] = []
    snap: dict = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "picovlm_version": __version__}

    checks.append(Check("python", "PASS" if sys.version_info >= (3, 10) else "WARN",
                        platform.python_version()))
    snap["python"] = platform.python_version()
    snap["platform"] = f"{platform.system()} {platform.release()} ({platform.machine()})"
    snap["cpu_count"] = os.cpu_count()
    ram = _ram_gb()
    snap["ram_gb"] = ram
    checks.append(Check("ram", "PASS" if (ram or 0) >= 16 else "WARN", f"{ram} GB"))

    usage = shutil.disk_usage(ROOT)
    snap["disk_gb"] = {"total": round(usage.total / 2**30, 1),
                       "free": round(usage.free / 2**30, 1)}
    checks.append(Check("disk", "PASS" if usage.free / 2**30 >= 20 else "WARN",
                        f"{usage.free / 2**30:.1f} GB free"))

    pkgs = {p: _pkg_version(p) for p in PACKAGES}
    snap["packages"] = pkgs
    missing = [p for p in ("torch", "transformers", "peft", "accelerate") if not pkgs.get(p)]
    checks.append(Check("core-packages", "FAIL" if missing else "PASS",
                        "missing: " + ", ".join(missing) if missing else "ok"))
    checks.append(Check("optional-packages", "SKIP",
                        "absent: " + ", ".join(p for p in ("datasets", "trl", "bitsandbytes")
                                               if not pkgs.get(p))))

    gpu: dict = {}
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            props = torch.cuda.get_device_properties(0)
            gpu = {"name": props.name, "total_mib": round(total / 2**20),
                   "free_mib": round(free / 2**20),
                   "capability": [int(props.major), int(props.minor)],
                   "cuda": torch.version.cuda, "driver": _driver()}
            status = "PASS" if gpu["free_mib"] >= 6000 else "WARN"
            checks.append(Check("gpu", status,
                                f"{gpu['name']} {gpu['total_mib']}MiB, free {gpu['free_mib']}MiB"))
        else:
            checks.append(Check("gpu", "FAIL", "torch 装了但 cuda 不可用"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("gpu", "SKIP", f"torch 不可用：{exc}"))
    snap["gpu"] = gpu

    if net:
        net_rows = {}
        for url in MIRRORS:
            try:
                with urllib.request.urlopen(url, timeout=5) as resp:
                    net_rows[url] = resp.status
            except Exception as exc:  # noqa: BLE001
                net_rows[url] = f"ERR {type(exc).__name__}"
        snap["network"] = net_rows
        ok = any(v == 200 for v in net_rows.values())
        checks.append(Check("network", "PASS" if ok else "WARN", json.dumps(net_rows)))
    else:
        checks.append(Check("network", "SKIP", "--no-net"))

    snap["seed_bands"] = SEED_BANDS
    checks.append(Check("seeds", "PASS", f"{len(SEED_BANDS)} bands frozen"))
    return checks, snap


def _driver() -> str | None:
    """驱动版本。先问 nvidia-smi（要用全路径：它不一定在 PATH 里），
    再退回注册表（WDDM 机器上 nvidia-smi 常被策略挡掉）。"""
    exe = Path(r"C:\Windows\System32\nvidia-smi.exe")
    for cmd in ([str(exe)] if exe.exists() else []) + ["nvidia-smi"]:
        try:
            out = subprocess.run([cmd, "--query-gpu=driver_version",
                                  "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=8)
        except Exception:
            continue
        if out.returncode == 0:
            line = (out.stdout or "").strip().splitlines()
            if line and line[0].strip():
                return line[0].strip()
    try:
        import winreg
        base = r"SYSTEM\CurrentControlSet\Control\Class" \
               r"\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            for i in range(16):
                try:
                    with winreg.OpenKey(root, f"{i:04d}") as sub:
                        val, _ = winreg.QueryValueEx(sub, "DriverVersion")
                        if val:
                            return str(val)
                except OSError:
                    continue
    except Exception:
        pass
    return None


def to_markdown(checks: list[Check], snap: dict) -> str:
    lines = [
        "# ENVIRONMENT",
        "",
        "> ⚠️ 本文件由 `python -m stage0_env.check_env` 自动生成，**不要手改**。",
        "> 手改会和 `outputs/env_snapshot.json` 对不上，verify_stage0 会报 FAIL。",
        "",
        f"生成时间：{snap['generated_at']}　PicoVLM v{snap['picovlm_version']}",
        "",
        "## 1. 机器",
        "",
        f"- 系统：{snap.get('platform')}",
        f"- CPU：{snap.get('cpu_count')} 核　内存：{snap.get('ram_gb')} GB",
        f"- 磁盘：{snap['disk_gb']['free']} GB 可用 / {snap['disk_gb']['total']} GB",
        "",
        "## 2. GPU",
        "",
    ]
    gpu = snap.get("gpu") or {}
    if gpu:
        lines += [f"- {gpu['name']}，总显存 {gpu['total_mib']} MiB，当前可用 **{gpu['free_mib']} MiB**",
                  f"- CUDA {gpu.get('cuda')}，驱动 {gpu.get('driver')}，capability {gpu.get('capability')}",
                  "", "> 可用显存远小于总显存时，训练前先关掉浏览器/聊天软件；"
                  "本项目按「可用 5.5–6.5 GB」设计。"]
    else:
        lines.append("- 未检测到可用 GPU（CPU 模式只能跑数据与单元测试）")
    lines += ["", "## 3. 依赖", "", "| 包 | 版本 |", "|---|---|"]
    for name, ver in snap["packages"].items():
        lines.append(f"| {name} | {ver or '—'} |")
    lines += ["", "## 4. 网络", ""]
    for url, status in (snap.get("network") or {}).items():
        lines.append(f"- {url} → {status}")
    lines += ["", "## 5. 种子区间（冻结）", "", "| 用途 | 区间起点 |", "|---|---|"]
    for k, v in snap["seed_bands"].items():
        lines.append(f"| {k} | {v} |")
    lines += ["", "## 6. 体检结论", "", "| 项 | 状态 | 细节 |", "|---|---|---|"]
    for c in checks:
        lines.append(f"| {c.name} | {c.status} | {c.detail} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-net", action="store_true")
    args = ap.parse_args(argv)
    checks, snap = collect(net=not args.no_net)
    io.write_json(OUTPUTS / "env_snapshot.json", {**snap, "checks": [c.__dict__ for c in checks]})
    md = to_markdown(checks, snap)
    (ROOT / "ENVIRONMENT.md").write_text(md, encoding="utf-8", newline="\n")
    for c in checks:
        print(f"[{c.status:<4}] {c.name:<16} {c.detail}")
    if any(c.status == "FAIL" for c in checks):
        print("\n[check_env] 有 FAIL 项，先修环境再往下走。")
        return 1
    print(f"\n[check_env] OK → ENVIRONMENT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
