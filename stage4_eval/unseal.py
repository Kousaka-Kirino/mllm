"""S4 · locked_test 开封（预注册纪律的可执行版本）。

为什么需要单独一个入口：`locked_test` 的价值全在"**只开一次**"。如果它和 dev_test
一样随手就能跑，人就会忍不住边调边看，于是它退化成第二个 dev_test，最后报告里的
"held-out" 三个字就是假的。

这个模块做三件事：

1. 默认拒绝第二次开封（存在 `UNSEALED.json` 就退出）；
2. 跑完之后把"开封记录"落盘：时间、命令行、git 提交、预测文件 sha256；
3. 打印一句话结论，方便直接抄进报告。

    python -m stage4_eval.unseal --model sft \
        --connector outputs/ckpt/sft_full/connector.pt \
        --adapter outputs/ckpt/sft_full/lora \
        --conditions correct_image,mismatched_image,prior_only \
        --run-id final_locked

真正的评测逻辑仍在 `stage4_eval.run` 里——这里只负责"仪式"和防重复。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from picovlm.config import OUTPUTS

from stage4_eval import run as R


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10
                              ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sft", choices=["prior", "align", "sft"])
    ap.add_argument("--conditions", default="correct_image,mismatched_image,prior_only")
    ap.add_argument("--connector", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--run-id", default="final_locked")
    ap.add_argument("--reason", default="", help="为什么现在开（写进记录）")
    ap.add_argument("--force", action="store_true", help="覆盖已有开封记录（慎用）")
    args = ap.parse_args(argv)

    out_dir = OUTPUTS / "eval" / args.run_id
    stamp = out_dir / "UNSEALED.json"
    if stamp.exists() and not args.force:
        old = json.loads(stamp.read_text(encoding="utf-8"))
        print(f"[unseal] 拒绝：{stamp} 已存在\n"
              f"  上次开封 {old.get('unsealed_at')}（{old.get('reason') or '未写理由'}）\n"
              f"  locked_test 只应开一次；确需重开请加 --force（并在报告里说明原因）。")
        return 1

    argv_run = ["--model", args.model, "--suite", "locked_test",
                "--conditions", args.conditions, "--run-id", args.run_id]
    if args.connector:
        argv_run += ["--connector", args.connector]
    if args.adapter:
        argv_run += ["--adapter", args.adapter]
    if args.limit:
        argv_run += ["--limit", str(args.limit)]

    t0 = time.time()
    rc = R.main(argv_run)
    if rc != 0:
        return rc

    preds = out_dir / "predictions.jsonl"
    rec = {
        "unsealed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reason": args.reason,
        "git_head": _git_head(),
        "argv": sys.argv[1:] if argv is None else list(argv),
        "run_id": args.run_id,
        "suite": "locked_test",
        "conditions": args.conditions,
        "predictions_sha256": _sha256(preds) if preds.exists() else None,
        "elapsed_s": round(time.time() - t0, 1),
    }
    stamp.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[unseal] 已记录 -> {stamp}\n"
          f"  这句话可以抄进报告：locked_test 于 {rec['unsealed_at']} "
          f"在 commit {rec['git_head']} 上开封一次，结果见 {out_dir.name}/records.json。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
