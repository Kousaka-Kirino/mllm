"""S3 验收：数据管线 → 梯度通路 → 存盘。

    python -m stage3_train.verify_stage3            # CPU 检查（秒级）
    python -m stage3_train.verify_stage3 --load     # 额外检查真权重 LoRA
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from picovlm.config import OUTPUTS, ROOT
from picovlm.report import Skip, approx, require, run

from .dataset import (EOS_SUFFIX, PROMPT, ChartSFT, answer_with_eos, collate,
                      load_samples)


def py(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HUB_OFFLINE": "1"}
    return subprocess.run([sys.executable, *args], cwd=str(ROOT),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=env)


def check_dataset() -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    samples = load_samples("train", limit=4)
    ds = ChartSFT(samples, tok, image_size=384)
    item = ds[0]
    px = item["pixel_values"]
    require(px.shape == (3, 384, 384), f"像素形状 {px.shape}")
    require(-1.0 <= float(px.min()) and float(px.max()) <= 1.0,
            f"像素范围 [{px.min():.2f},{px.max():.2f}] 不在 [-1,1]")
    ids, lab = item["input_ids"], item["labels"]
    require(len(ids) == len(lab), "input_ids/labels 长度不一致")
    n_sup = sum(1 for x in lab if x != -100)
    want = len(tok(answer_with_eos(tok, samples[0].answer),
                   add_special_tokens=False)["input_ids"])
    require(n_sup == want, f"监督位数 {n_sup} 不等于答案 token 数 {want}")
    require(all(x == -100 for x in lab[:len(lab) - n_sup]), "prompt 段没有全部 -100")
    require(lab[-1] != -100, "答案末尾（EOS）没有被监督")
    return f"sample={samples[0].id} seq={len(ids)} supervised={n_sup}"


def check_collate() -> str:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tok.pad_token = tok.eos_token
    ds = ChartSFT(load_samples("train", limit=3), tok, image_size=384)
    b = collate([ds[0], ds[1]], tok.pad_token_id)
    ids, msk, lab = b["input_ids"], b["attention_mask"], b["labels"]
    require(ids.shape == msk.shape == lab.shape, "三个张量形状不一致")
    for row_ids, row_msk, row_lab in zip(ids, msk, lab):
        n_real = int(row_msk.sum())
        require(all(int(x) == tok.pad_token_id for x in row_ids[n_real:]),
                "padding 位置不是 pad_id")
        require(all(int(x) == -100 for x in row_lab[n_real:]),
                "padding 位置的 label 不是 -100")
        require(all(int(x) == 0 for x in row_msk[n_real:]), "mask 没有清零")
    return f"batch={ids.shape} padded_ok"


def check_prompt_contract() -> str:
    require(PROMPT == "Question: {q}\nAnswer:", f"prompt 模板被改动：{PROMPT!r}")
    require(EOS_SUFFIX == "\n", "EOS 后缀被改动")
    return "prompt/eos 模板稳定"


def check_freeze() -> str:
    from transformers import AutoTokenizer

    from picovlm.config import load_config, model_config_from
    from picovlm.model.vlm import PicoVLM

    cfg = load_config("align_pilot.yaml")
    mc = model_config_from(cfg)
    tok = AutoTokenizer.from_pretrained(mc.llm)
    model = PicoVLM(mc, load_weights=False)
    from .align import freeze_for_align

    params = freeze_for_align(model)
    n_train = sum(p.numel() for p in params)
    n_vision = sum(p.numel() for p in model.vision.parameters()
                   if p.requires_grad)
    require(n_vision == 0, "视觉塔没有被冻结")
    approx(n_train / 1e6, sum(p.numel() for p in model.connector.parameters()) / 1e6,
           1e-6, "可训参数应只有连接器")
    return f"trainable {n_train/1e6:.2f}M = connector only"


def _dry(module: str, tag: str) -> str:
    out = Path(tempfile.mkdtemp(prefix=f"s3_{tag}_")) / "ckpt"
    r = py("-m", module, "--dry-run", "--limit", "4", "--batch-size", "1",
           "--grad-accum", "2", "--max-steps", "1", "--epochs", "1",
           "--out", str(out))
    if r.returncode != 0:
        raise AssertionError(f"{module} dry-run 失败：\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    ck = out / "connector.pt"
    require(ck.exists(), f"{module} 没有存下 connector.pt")
    state = torch.load(ck, map_location="cpu")
    require(all(torch.isfinite(v).all() for v in state.values()
                if torch.is_floating_point(v)), "存的连接器权重里有 NaN/Inf")
    return f"{tag} dry-run ok  state={len(state)} tensors"


def check_align_dry() -> str:
    return _dry("stage3_train.align", "align")


def check_sft_dry() -> str:
    return _dry("stage3_train.sft", "sft")


def check_align_ckpt() -> str:
    p = OUTPUTS / "ckpt" / "align" / "connector.pt"
    if not p.exists():
        raise Skip("还没跑过真对齐训练（python -m stage3_train.align）")
    state = torch.load(p, map_location="cpu")
    bad = [k for k, v in state.items()
           if torch.is_floating_point(v) and not torch.isfinite(v).all()]
    require(not bad, f"连接器里有非有限值：{bad[:3]}")
    norm = float(torch.cat([v.flatten() for v in state.values()]).norm())
    require(norm > 0, "连接器权重全 0")
    return f"{p.name} ok  tensors={len(state)}  |w|={norm:.2f}"


def check_lora_attach() -> str:
    from transformers import AutoTokenizer

    from picovlm.config import load_config, model_config_from
    from picovlm.model.vlm import PicoVLM

    cfg = load_config("sft_pilot.yaml")
    mc = model_config_from(cfg)
    tok = AutoTokenizer.from_pretrained(mc.llm)
    model = PicoVLM(mc, load_weights=True)
    from .sft import attach_lora

    n = attach_lora(model, cfg.get("lora", {}))
    require(n > 0, "LoRA 没有挂上任何参数")
    require(1e5 < n < 5e7, f"LoRA 参数量异常：{n}")
    return f"lora params {n/1e6:.2f}M  (r={cfg['lora']['r']})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true", help="额外做真权重 LoRA 检查")
    args = ap.parse_args(argv)
    checks = [
        ("dataset-shapes", check_dataset),
        ("collate-padding", check_collate),
        ("prompt-contract", check_prompt_contract),
        ("freeze-policy", check_freeze),
        ("align-dry-run", check_align_dry),
        ("sft-dry-run", check_sft_dry),
        ("align-ckpt", check_align_ckpt),
    ]
    if args.load:
        checks.append(("lora-attach", check_lora_attach))
    return run(checks, "S3 验收（2.0）")


if __name__ == "__main__":
    raise SystemExit(main())
