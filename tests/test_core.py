"""pytest：契约、判分、统计、渲染、连接器。

    pytest -q            # 全部
    pytest -q -k stats   # 只跑统计

这些测试与 `stage*/verify_*.py` 的分工：
- verify 是**阶段验收**（要读数据、要跑脚本，慢）；
- tests 是**单元测试**（纯函数、毫秒级），改代码时第一时间跑。
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from picovlm import answers, stats
from picovlm.schema import Prediction, Sample


class TestSchema:
    def test_sample_roundtrip(self) -> None:
        s = Sample(id="a", source="synth", split="train",
                   image_path="data/images/a.png", image_size=384,
                   question="q", answer="1")
        assert Sample.from_dict(s.to_dict()) == s

    def test_prior_requires_answer(self) -> None:
        with pytest.raises(ValueError):
            Sample(id="a", source="s", split="train", image_path="p",
                   image_size=8, question="q", answer="1", prior_type="marginal")

    def test_bad_split(self) -> None:
        with pytest.raises(ValueError):
            Sample(id="a", source="s", split="validation", image_path="p",
                   image_size=8, question="q", answer="1")

    def test_prediction_defaults(self) -> None:
        p = Prediction(sample_id="a", model_id="m", run_id="r", raw="1")
        assert p.correct_relaxed is None and p.budget == "fixed"


class TestAnswers:
    @pytest.mark.parametrize("pred,gold", [
        ("42", "42"), ("42.0", "42"), ("1,234", "1234"),
        ("12.5%", "12.5"), ("about 7", "7"),
    ])
    def test_numeric_relaxed(self, pred, gold) -> None:
        assert answers.is_correct(pred, gold, "number")

    def test_strict_rejects_format(self) -> None:
        assert not answers.is_correct("42.0", "42", "number", mode="strict")
        assert answers.is_correct("42", "42", "number", mode="strict")

    @pytest.mark.parametrize("pred,gold,ok", [
        ("Yes", "yes", True), ("是", "yes", True), ("no", "yes", False),
        ("true", "True", True),
    ])
    def test_bool(self, pred, gold, ok) -> None:
        assert answers.is_correct(pred, gold, "bool") is ok

    def test_long_answer_containing_gold(self) -> None:
        assert answers.is_correct("The answer is 42.", "42", "number")
        assert answers.is_correct("答案是类别 B", "B", "label")

    def test_empty_is_wrong(self) -> None:
        assert not answers.is_correct("", "42", "number")


class TestStats:
    def test_wilson_known_values(self) -> None:
        lo, hi = stats.wilson_ci(43, 50)
        assert lo == pytest.approx(0.738, abs=0.01)
        assert hi == pytest.approx(0.930, abs=0.01)

    def test_wilson_bounds(self) -> None:
        assert stats.wilson_ci(0, 10)[0] == 0.0
        assert stats.wilson_ci(10, 10)[1] == 1.0
        assert stats.wilson_ci(0, 0) == (0.0, 1.0)

    def test_mcnemar_symmetric(self) -> None:
        assert stats.mcnemar_exact(5, 5) == pytest.approx(1.0)
        assert stats.mcnemar_exact(10, 0) < 0.01

    def test_paired_compare(self) -> None:
        a = [True] * 8 + [False] * 2
        b = [True] * 2 + [False] * 8
        r = stats.paired_compare(a, b)
        assert r["wins"] == 6 and r["losses"] == 0 and r["ties"] == 4
        assert r["delta"] == pytest.approx(0.6)

    def test_bootstrap_deterministic(self) -> None:
        vals = [0.1, 0.5, 0.9, 0.3]
        assert stats.bootstrap_ci(vals, seed=1) == stats.bootstrap_ci(vals, seed=1)


class TestRender:
    def test_glyph_identity(self) -> None:
        from stage1_data.render import ChartSpec, glyph_px_modelled
        for fs in (0.55, 1.0, 1.45):
            spec = ChartSpec(seed=1, kind="bar", n_categories=6, n_series=1,
                             font_scale=fs, size=1152, legend_gap=0.02,
                             distractors=0, noise=0.0, annotate=False)
            assert glyph_px_modelled(spec) == pytest.approx(8.0 * fs, abs=1e-6)

    def test_render_is_deterministic(self) -> None:
        from stage1_data.render import ChartSpec, render
        spec = ChartSpec(seed=3, kind="bar", n_categories=5, n_series=1,
                         font_scale=1.0, size=600, legend_gap=0.02,
                         distractors=0, noise=0.0, annotate=False)
        a = np.asarray(render(spec).image)
        b = np.asarray(render(spec).image)
        assert np.array_equal(a, b)


class TestConnectors:
    def test_mlp_shape(self) -> None:
        from picovlm.model import build_connector
        conn = build_connector("mlp", 16, 24)
        assert conn(torch.randn(2, 49, 16)).shape == (2, 49, 24)

    def test_resampler_shape(self) -> None:
        from picovlm.model import build_connector
        conn = build_connector("resampler", 16, 24, n_queries=8)
        assert conn(torch.randn(3, 49, 16)).shape == (3, 8, 24)

    def test_pixelshuffle_shape(self) -> None:
        from picovlm.model import build_connector
        conn = build_connector("pixelshuffle", 16, 24, factor=2)
        assert conn(torch.randn(1, 36, 16)).shape == (1, 9, 24)

    def test_pixelshuffle_rejects_non_square(self) -> None:
        from picovlm.model import build_connector
        conn = build_connector("pixelshuffle", 8, 8, factor=2)
        with pytest.raises(ValueError):
            conn(torch.randn(1, 35, 8))


class TestAssemble:
    def test_labels_prefix(self) -> None:
        from picovlm.model import assemble_inputs
        vis = torch.zeros(1, 5, 4)
        txt = torch.zeros(1, 3, 4)
        mask = torch.ones(1, 3, dtype=torch.long)
        labels = torch.tensor([[-100, 7, 8]])
        _, am, lb = assemble_inputs(vis, txt, mask, labels)
        assert am.shape == (1, 8)
        assert (lb[0, :5] == -100).all() and lb[0, 6] == 7

    def test_padding_masked(self) -> None:
        from picovlm.model import assemble_inputs
        vis = torch.zeros(1, 2, 4)
        txt = torch.zeros(1, 3, 4)
        mask = torch.tensor([[1, 1, 0]])
        labels = torch.tensor([[7, 8, 9]])
        _, _, lb = assemble_inputs(vis, txt, mask, labels)
        assert lb[0, -1].item() == -100
