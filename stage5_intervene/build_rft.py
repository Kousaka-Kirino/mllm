"""S5 · `build_rft` 命令行入口（规划 2.0 里写的 `python -m stage5_intervene.build_rft`）。

为什么单独一个入口：RFT 不是训练器，而是**数据构造器**。它把同一批 rollout 里
采样正确的答案挑出来，写成 jsonl，之后用 S3 的 SFT 脚本再训一遍。把它做成独立
命令，才能保证"GRPO vs RFT"两臂吃的是同一批 prompt、同一份采样预算。

    python -m stage5_intervene.build_rft --model sft --limit 40 --group-size 4

产物：`outputs/rft/rft_<stamp>.jsonl`（sample_id / answer / pred / family_id ...）
与同目录 `meta.json`（rollouts / kept / keep_rate）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from picovlm.config import DATA, OUTPUTS

from stage4_eval.predictor import load_pico
from stage5_intervene import rft


def load_split(split: str, limit: int | None, root: Path = DATA):
    from picovlm.io_utils import read_jsonl
    from picovlm.schema import Sample

    path = root / "synth" / f"{split}.jsonl"
    if not path.exists():
        raise SystemExit(f"找不到 {path}，先跑 stage1_data.build")
    rows = read_jsonl(path)
    samples = [Sample(**r) for r in rows]
    if limit:
        samples = samples[:limit]
    return samples


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sft", choices=["align", "sft"])
    ap.add_argument("--split", default="dev_test")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--connector", default=None, help="连接器 ckpt，默认用 model 对应的")
    ap.add_argument("--adapter", default=None, help="LoRA 目录，默认 outputs/ckpt/sft_pilot/lora")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    samples = load_split(args.split, args.limit)
    model, tok = load_pico(args.model, connector_ckpt=args.connector,
                           adapter=args.adapter)
    stats = rft.build(model, tok, samples, group_size=args.group_size,
                      out=Path(args.out) if args.out else None)
    print(f"[rft] {stats['kept']}/{stats['rollouts']} 条被保留，"
          f"keep_rate={stats['keep_rate']:.3f} -> {stats['path']}")
    if stats["kept"] == 0:
        # keep_rate=0 本身是一个有效（且重要）的实验结果：说明这个 checkpoint
        # 在当前 prompt 上一条都没答对，RFT 这条路走不通。不当成命令失败。
        print("[rft] 注意：一条都没留下。这本身就是结果——要么模型还没学会答题格式"
              "（查 S3 的 EOS 后缀），要么该 split 太难（冲突格占比高）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
