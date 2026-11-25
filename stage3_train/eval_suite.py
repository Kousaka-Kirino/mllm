"""S3 评测入口：把"某个答题者"跑到某个 split 上，落 predictions + records。

现在支持两类答题者：
- `--policy oracle|prior|constant|random|first_option`：零显存基线（先跑这个）
- `--model outputs/ckpt/xxx`：真 VLM（等 S3 训完再用）

用法：
    python -m stage3_train.eval_suite --policy prior --split test
    python -m stage3_train.eval_suite --policy oracle --split test --run-id sanity
"""

from __future__ import annotations

import argparse
import random
import sys

from picovlm import io_utils as io
from picovlm import policies
from picovlm.config import DATA, OUTPUTS, REPORTS, EvalConfig
from picovlm.eval_utils import predict, print_table, records, save
from picovlm.report import Skip
from picovlm.schema import Sample


def load_split(split: str) -> list[Sample]:
    path = DATA / "synth" / f"{split}.jsonl"
    if not path.exists():
        raise Skip(f"{path} 不存在，先跑 python -m stage1_data.build")
    return [Sample.from_dict(r) for r in io.read_jsonl(path)]


def with_train_stats(samples: list[Sample]) -> list[Sample]:
    """给 constant/random 基线补上"训练集多数答案"和候选集。"""
    train = load_split("train")
    stats = policies.fit_train_stats(train)
    return policies.attach_stats(samples, stats)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="prior",
                    choices=sorted(policies.POLICIES))
    ap.add_argument("--split", default="test", choices=("train", "dev", "test"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--run-id", default="")
    ap.add_argument("--seed", type=int, default=EvalConfig().seed)
    args = ap.parse_args(argv)

    try:
        samples = with_train_stats(load_split(args.split))
    except Skip as exc:
        print(f"[SKIP] {exc}")
        return 0
    if args.limit:
        samples = samples[: args.limit]

    run_id = args.run_id or f"{args.policy}-{args.split}"
    rng = random.Random(args.seed)
    fn = policies.get_policy(args.policy)

    preds = predict(lambda s: fn(s, rng), samples,
                    model_id=f"policy:{args.policy}", run_id=run_id)
    recs = records(preds, samples, run_id, f"policy:{args.policy}")
    paths = save(preds, recs, run_id)

    print(f"run_id={run_id}  n={len(preds)}  -> {paths['records']}")
    print_table(recs)

    # 顺手把 2×2 诊断也生成出来：每次评测都自带报告，S4 不用再单独触发
    from stage4_diag.grid_2x2 import analyse, to_markdown
    grid = analyse({s.id: s for s in samples}, list(preds))
    out = io.ensure_dir(OUTPUTS / "eval" / run_id)
    io.write_json(out / "grid.json", grid)
    (REPORTS / f"grid_{run_id}.md").write_text(to_markdown(grid), encoding="utf-8")
    print(f"grid -> {REPORTS / f'grid_{run_id}.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
