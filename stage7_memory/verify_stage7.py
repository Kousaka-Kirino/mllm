"""S7 验收：索引召回、引用判定、拒答边界。"""

from __future__ import annotations

import sys

import numpy as np

from picovlm.report import require, run

from .index import PageIndex, citation_ok, recall_at_k


def main() -> int:
    rng = np.random.default_rng(0)

    def index_recall() -> str:
        pages = [f"p{i}" for i in range(20)]
        base = rng.normal(size=(20, 32)).astype(np.float32)
        idx = PageIndex(dim=32)
        idx.extend(pages, base)
        require(len(idx) == 20, "索引长度错")
        queries = [(base[i] + rng.normal(scale=0.01, size=32), pages[i])
                   for i in range(20)]
        r1 = recall_at_k(idx, queries, 1)
        require(r1 == 1.0, f"Recall@1 应为 1.0，得到 {r1}")
        return f"Recall@1={r1:.0%}（20 页）"

    def topk_order() -> str:
        idx = PageIndex(dim=4)
        idx.add("a", np.array([1, 0, 0, 0]))
        idx.add("b", np.array([0.9, 0.1, 0, 0]))
        idx.add("c", np.array([0, 1, 0, 0]))
        hits = idx.search(np.array([1, 0, 0, 0]), k=2)
        require([h[0] for h in hits] == ["a", "b"], f"排序错：{hits}")
        require(hits[0][1] > hits[1][1], "分数未降序")
        return f"{hits[0][0]}({hits[0][1]:.2f}) > {hits[1][0]}({hits[1][1]:.2f})"

    def empty_index() -> str:
        idx = PageIndex(dim=8)
        require(idx.search(np.zeros(8), k=3) == [], "空索引应返回空列表")
        return "空索引返回 []"

    def citation_and_abstain() -> str:
        require(citation_ok("According to page 3, Q2 is 42.", ["p3", "p5"]),
                "正常引用应通过")
        require(not citation_ok("Q2 is 42.", ["p3"]), "无引用不应通过")
        return "引用缺失可检出"

    def dim_mismatch() -> str:
        idx = PageIndex(dim=8)
        idx.add("a", np.zeros(8))
        try:
            idx.add("b", np.zeros(4))
        except ValueError:
            return "维度不符时报错"
        raise AssertionError("维度不符应报错")

    return run([
        ("index-recall", index_recall),
        ("topk-order", topk_order),
        ("empty-index", empty_index),
        ("citation", citation_and_abstain),
        ("dim-mismatch", dim_mismatch),
    ], title="stage7_memory")


if __name__ == "__main__":
    sys.exit(main())
