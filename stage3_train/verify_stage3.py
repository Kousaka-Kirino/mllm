"""S3 验收：labels 掩码、dry-run 两个训练入口、评测管线端到端。

dry-run 用替身模型 + 字符级 tokenizer，不下载 1.5GB 权重——CPU 几十秒跑完。
"""

from __future__ import annotations

import io as _io
import subprocess
import sys

from picovlm.config import DATA
from picovlm.report import Skip, require, run
from picovlm.schema import Sample


class _CharTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 1000 + 10 for c in text]}


def _run(module: str, *args: str) -> str:
    proc = subprocess.run([sys.executable, "-m", module, *args],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=600)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise AssertionError(f"{module} 退出码 {proc.returncode}\n{out[-1500:]}")
    return out


def main() -> int:
    from stage3_train.data import build_example, collate

    path = DATA / "synth" / "train.jsonl"
    if not path.exists():
        raise Skip("数据未生成")

    samples = [Sample.from_dict(r) for r in
               [__import__("json").loads(l) for l in
                _io.open(path, encoding="utf-8")][:4]]

    def labels_only_answer() -> str:
        ex = build_example(samples[0], _CharTok(), image_size=64)
        n_prompt = int((ex["labels"] == -100).sum())
        n_answer = ex["labels"].numel() - n_prompt
        require(n_answer > 0, "答案 token 全被掩掉了")
        require(bool((ex["labels"][:n_prompt] == -100).all()), "prompt 段没掩干净")
        require(bool((ex["labels"][n_prompt:] != -100).all()), "答案段被误掩")
        return f"prompt {n_prompt} token 掩掉，答案 {n_answer} token 参与 loss"

    def collate_padding() -> str:
        batch = collate([build_example(s, _CharTok(), 64) for s in samples])
        L = batch["input_ids"].shape[1]
        require(batch["pixel_values"].shape == (len(samples), 3, 64, 64), "图像 batch 形状错")
        pad = batch["attention_mask"] == 0
        require(bool((batch["labels"][pad] == -100).all()), "padding 位置 label 必须 -100")
        require(int(batch["attention_mask"].sum()) > 0, "attention_mask 全 0")
        require(int((batch["labels"] != -100).sum()) > 0, "没有任何监督 token")
        return f"{len(samples)}×{L}，padding 已同步掩码"

    def align_dry_run() -> str:
        out = _run("stage3_train.train_align", "--dry-run")
        require("loss=" in out, f"没有 loss 输出：{out[-300:]}")
        return out.strip().splitlines()[-1][:80]

    def sft_dry_run() -> str:
        out = _run("stage3_train.train_sft", "--dry-run")
        require("loss=" in out, f"没有 loss 输出：{out[-300:]}")
        return out.strip().splitlines()[-1][:80]

    def eval_pipeline() -> str:
        _run("stage3_train.eval_suite", "--policy", "oracle", "--split", "test",
             "--run-id", "verify-oracle")
        import json
        from picovlm.config import OUTPUTS
        recs = json.loads((OUTPUTS / "eval" / "verify-oracle" / "records.json")
                          .read_text(encoding="utf-8"))
        require(recs and all(r["accuracy"] == 1.0 for r in recs),
                f"oracle 不是 100%：{[(r['suite'], r['accuracy']) for r in recs]}")
        return f"{len(recs)} 个 suite 全部 100%"

    return run([
        ("labels-only-answer", labels_only_answer),
        ("collate-padding", collate_padding),
        ("align-dry-run", align_dry_run),
        ("sft-dry-run", sft_dry_run),
        ("eval-pipeline", eval_pipeline),
    ], title="stage3_train")


if __name__ == "__main__":
    sys.exit(main())
