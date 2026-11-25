"""S7：视觉检索记忆的索引与召回（numpy 版，零依赖、CPU 可跑）。

真实场景：用户上传 50 页 PDF/幻灯片，问"第 3 页那个图表里 Q2 是多少"。
模型要先**检索到对的那一页**，再回答，再给出引用。

这里把"检索"与"回答"解耦：
- 本文件只负责向量索引与 top-k（后端可以是 CLIP、ColPali、BGE）；
- `stage7_memory/answer.py` 负责把 top-k 交给 VLM，并要求引用页码。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


@dataclass
class PageIndex:
    """页面向量的余弦索引。数据量小（<10k 页）时暴力检索足够快。"""

    dim: int = 512
    ids: list[str] = field(default_factory=list)
    vecs: np.ndarray = field(default_factory=lambda: np.zeros((0, 512), dtype=np.float32))

    def add(self, page_id: str, vec: np.ndarray) -> None:
        v = l2norm(np.asarray(vec, dtype=np.float32).reshape(1, -1))
        if self.vecs.size == 0:
            self.vecs = v
        else:
            self.vecs = np.vstack([self.vecs, v])
        self.ids.append(page_id)

    def extend(self, page_ids: list[str], vecs: np.ndarray) -> None:
        if len(page_ids) != len(vecs):
            raise ValueError("page_ids 与 vecs 数量不一致")
        for pid, v in zip(page_ids, vecs):
            self.add(pid, v)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if not self.ids:
            return []
        q = l2norm(np.asarray(query, dtype=np.float32).reshape(1, -1))
        scores = (self.vecs @ q.T).ravel()
        order = np.argsort(-scores)[:k]
        return [(self.ids[i], float(scores[i])) for i in order]

    def __len__(self) -> int:
        return len(self.ids)


def recall_at_k(index: PageIndex, queries: list[tuple[np.ndarray, str]], k: int = 5) -> float:
    """queries: [(query_vec, gold_page_id)]。"""
    if not queries:
        return 0.0
    hit = 0
    for q, gold in queries:
        if any(pid == gold for pid, _ in index.search(q, k)):
            hit += 1
    return hit / len(queries)


def citation_ok(answer: str, pages: list[str]) -> bool:
    """答案里必须出现至少一个检索到的页码引用，否则算"无依据"。

    认这几种写法：`p3`、`page 3`、`p.3`、`#3`、`第3页`。
    """
    import re

    a = answer.lower()
    for p in pages:
        if p.lower() in a:
            return True
        num = re.sub(r"\D", "", p)
        if num and re.search(rf"(?:page|p\.?|#|第)\s*{num}(?!\d)", a):
            return True
    return False
