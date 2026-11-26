"""`picovlm.cli`：一条命令问一张图。

    python -m picovlm.cli --image data/synth/images/ab12.png --ask "Which category is largest?"
    python -m picovlm.cli --image ... --ask "..." --model prior     # 盲基线，不加载模型
    python -m picovlm.cli --image ... --ask "..." --model sft       # 默认：载入 SFT 权重

设计取舍：CLI **不做** 训练/评测，只做"单张图的问答"，方便复试现场演示；
训练和评测各有自己的模块（S3/S4）。这样 CLI 的失败不会影响可复现的实验。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from picovlm.config import OUTPUTS, ROOT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="picovlm")
    ap.add_argument("--image", required=True, help="图片路径（绝对或相对项目根）")
    ap.add_argument("--ask", required=True, help="问题，例如 'Which bar is tallest?'")
    ap.add_argument("--model", default="sft", choices=["prior", "align", "sft"],
                    help="prior=不加载模型（盲基线）；align/sft=加载对应权重")
    ap.add_argument("--connector", default=None, help="连接器权重，默认用 model 默认值")
    ap.add_argument("--adapter", default=None, help="LoRA 目录，默认用 SFT pilot")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args(argv)

    img = Path(args.image)
    if not img.is_absolute():
        img = ROOT / img
    if not img.exists():
        print(f"找不到图片：{img}")
        return 2

    from PIL import Image

    from stage4_eval.predictor import PicoPredictor, RulePrior, load_pico
    from picovlm.schema import Sample

    sample = Sample(id="cli", source="cli", split="dev_test",
                    image_path=str(img), image_size=384, question=args.ask,
                    answer="(cli)", answer_kind="text")
    if args.model == "prior":
        from stage3_train.dataset import load_image
        pred = RulePrior()
        ans = pred.answer(sample, load_image(sample.image_path, 384))
        print(f"[prior] {ans}")
        print("（盲基线：没看这张图，只看问题的先验）")
        return 0

    ckpt = args.connector or f"outputs/ckpt/{'sft_pilot' if args.model == 'sft' else 'align'}/connector.pt"
    adapter = args.adapter if args.adapter is not None else (
        "outputs/ckpt/sft_pilot/lora" if args.model == "sft" else None)
    if not (ROOT / ckpt).exists():
        print(f"找不到连接器权重：{ckpt}\n先跑 stage3_train.align 或 .sft")
        return 2

    model, tok = load_pico(args.model, connector_ckpt=ckpt, adapter=adapter)
    pred = PicoPredictor(model, tok, model_id=f"pico:{args.model}",
                         max_new_tokens=args.max_new_tokens)

    from stage3_train.dataset import load_image
    img_arr = load_image(sample.image_path, 384)
    ans = pred.answer(sample, img_arr)
    print(f"[{args.model}] {ans!r}")
    print(f"（视觉 token {pred.visual_tokens}，图片 {Image.open(img).size}）")

    # 顺手给一张盲基线的对比，演示"看图 vs 不看图"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
