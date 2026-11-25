"""S8：单题推理 CLI（演示 / 手工抽查用）。

    python -m stage8_system.cli --image data/images/synth/xxx.png --ask "哪个类别最大？"
    python -m stage8_system.cli --image ... --ask ... --policy oracle   # 不加载模型

真模型路径需要 S3 的 checkpoint；没训完时用 --policy 走基线，保证 CLI 一直可用。
"""

from __future__ import annotations

import argparse
import sys

from picovlm import policies
from picovlm.schema import Sample


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--ask", required=True)
    ap.add_argument("--policy", default="prior", choices=sorted(policies.POLICIES))
    ap.add_argument("--ckpt", default="", help="S3 训完的 outputs/ckpt/sft")
    args = ap.parse_args(argv)

    sample = Sample(id="cli", source="cli", split="test", image_path=args.image,
                    image_size=384, question=args.ask, answer="(cli)")

    if args.ckpt:
        from picovlm.config import ModelConfig
        from picovlm.model import PicoVLM
        print("[model] 加载 checkpoint（骨架版尚未接 generate，见 S3 README）")
        _ = PicoVLM(ModelConfig(), load_weights=True)
        return 2

    answer = policies.get_policy(args.policy)(sample, __import__("random").Random(0))
    print(f"Q: {args.ask}")
    print(f"A: {answer}   (policy={args.policy})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
