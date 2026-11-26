"""S5 · E2 反事实测量：模型对"输入变化"是否敏感。

两个互补的问题，用两种改动回答：

| 改动 | 金标 | 如果模型**真的**在读图 | 如果模型**在背题面/先验** |
|---|---|---|---|
| 只改问句措辞（图不变） | 不变 | 答案不变（不变性率 = 1） | 答案乱变（不变性率低） |
| 改表里的数值（图变、金标变） | 变 | 新答案 = 新金标（响应率 = 1） | 仍答旧答案（响应率 ≈ 0） |

注意：**不变性率高不代表看图**——一个只会拿先验的模型对措辞也可能稳定。
所以这两个指标必须一起读，单看任何一个都会得出相反结论。

    python -m stage5_intervene.invariance --limit 30 --model sft
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from picovlm.answers import normalize
from picovlm.config import OUTPUTS, REPORTS, ROOT
from picovlm.io_utils import ensure_dir, utc_stamp, write_json
from picovlm.schema import Prediction
from picovlm.stats import wilson_ci

from stage3_train.dataset import load_image, load_samples
from stage5_intervene.counterfactual import numeric_variant, wording_variant


def _build(model: str):
    from stage4_eval.predictor import PicoPredictor, load_pico
    kind = "sft" if model == "sft" else "align"
    ckpt = f"outputs/ckpt/{'sft_pilot' if kind == 'sft' else 'align'}/connector.pt"
    adapter = "outputs/ckpt/sft_pilot/lora" if kind == "sft" else None
    m, tok = load_pico(kind, connector_ckpt=ckpt, adapter=adapter)
    return PicoPredictor(m, tok, model_id=f"pico:{model}")


def measure(predictor, samples, limit: int) -> dict:
    rows: list[dict] = []
    for s in samples[:limit]:
        base_img = load_image(s.image_path, 384, ROOT)
        pred_base = predictor.answer(s, base_img)

        w = wording_variant(s)
        pred_w = None
        if w is not None:
            pred_w = predictor.answer(w, base_img)

        n = numeric_variant(s)
        pred_n = None
        if n is not None:
            pred_n = predictor.answer(n, load_image(n.image_path, 384, ROOT))

        rows.append({
            "id": s.id, "qtype": s.qtype, "acuity_bin": s.acuity_bin,
            "gold": s.answer,
            "base": pred_base,
            "wording": pred_w,
            "numeric": pred_n,
            "numeric_gold": n.answer if n else None,
            "unchanged_on_wording": (None if pred_w is None
                                     else normalize(pred_w) == normalize(pred_base)),
            "response_on_numeric": (None if pred_n is None
                                    else normalize(pred_n) == normalize(n.answer)),
            "stuck_on_old": (None if pred_n is None
                             else normalize(pred_n) == normalize(s.answer)),
        })
    return {"rows": rows}


def summarize(data: dict) -> dict:
    rows = data["rows"]
    w = [r["unchanged_on_wording"] for r in rows if r["unchanged_on_wording"] is not None]
    n = [r for r in rows if r["response_on_numeric"] is not None]
    k_w, k_n = sum(w), sum(r["response_on_numeric"] for r in n)
    k_old = sum(r["stuck_on_old"] for r in n)
    ci_w = wilson_ci(k_w, len(w)) if w else (None, None)
    ci_n = wilson_ci(k_n, len(n)) if n else (None, None)
    return {
        "n_wording": len(w), "unchanged_on_wording": k_w,
        "invariance_rate": (k_w / len(w)) if w else None,
        "invariance_ci": ci_w,
        "n_numeric": len(n), "response_on_numeric": k_n,
        "response_rate": (k_n / len(n)) if n else None,
        "response_ci": ci_n,
        "stuck_on_old": k_old,
        "stuck_rate": (k_old / len(n)) if n else None,
    }


def to_markdown(s: dict, model: str) -> str:
    def pct(x):
        return "—" if x is None else f"{x:.3f}"

    def ci(c):
        return "—" if not c or c[0] is None else f"[{c[0]:.3f}, {c[1]:.3f}]"

    return "\n".join([
        f"# E2 反事实测量（model={model}）",
        "",
        "| 指标 | n | 值 | 95% CI | 读法 |",
        "|---|---:|---:|---|---|",
        f"| 措辞不变性率 | {s['n_wording']} | {pct(s['invariance_rate'])} | "
        f"{ci(s['invariance_ci'])} | 高：对问法稳定（也可能只是懒得看题） |",
        f"| 数值响应率 | {s['n_numeric']} | {pct(s['response_rate'])} | "
        f"{ci(s['response_ci'])} | 高：真的重新读了图 |",
        f"| 仍答旧答案 | {s['n_numeric']} | {pct(s['stuck_rate'])} | — | "
        "高：把旧答案当先验沿用 |",
        "",
        "> 两个指标必须一起读：不变性率高 + 响应率低 = 背了先验；两者都高 = 真在读图。",
    ])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sft", choices=["align", "sft"])
    ap.add_argument("--split", default="dev_test")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args(argv)

    samples = load_samples(args.split)
    if not samples:
        print("没有样本，先跑 stage1_data.build")
        return 2
    predictor = _build(args.model)
    data = measure(predictor, samples, args.limit)
    s = summarize(data)
    md = to_markdown(s, args.model)
    print(md)

    stamp = utc_stamp()
    if args.save:
        ensure_dir(OUTPUTS / "eval")
        write_json(OUTPUTS / "eval" / f"invariance_{args.model}.json",
                   {"model": args.model, "split": args.split,
                    "stamp": stamp, **s, "rows": data["rows"]})
        ensure_dir(REPORTS)
        (REPORTS / f"invariance_{args.model}.md").write_text(md + "\n", encoding="utf-8")
        print(f"\n[saved] reports/invariance_{args.model}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
