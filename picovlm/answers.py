"""判分：严格口径 + 宽松口径。

为什么必须两套：
- 严格口径（`strict`）只做空白/大小写归一化，用来发现"答案格式在漂"；
- 宽松口径（`relaxed`）允许数值容差 5%（ChartQA 的传统做法）与 yes/no 同义词，
  用来衡量"到底会不会"。

**任何报告都不能只给宽松口径。** 两者差值本身就是"格式 vs 能力"的证据。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

_TRUE = {"yes", "true", "是", "对", "正确", "y", "t", "1"}
_FALSE = {"no", "false", "否", "错", "不对", "错误", "n", "f", "0"}

_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_PUNCT = str.maketrans({"％": "%", "，": ",", "．": ".", "－": "-", "＋": "+",
                        "（": "(", "）": ")"})


def normalize(text: str) -> str:
    """去空白、统一全角、去句末标点。不做语义改写。"""
    if text is None:
        return ""
    s = unicodedata.normalize("NFKC", str(text)).translate(_PUNCT)
    s = s.strip().strip("。.!！?？,，;；")
    s = re.sub(r"\s+", " ", s)
    return s


def to_number(text: str) -> Optional[float]:
    """从答案里抠出第一个数；抠不到返回 None。

    支持 `1,234`、`12.5%`、`-3`、`about 7`。
    """
    s = normalize(text).replace(",", "")
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def canonical_bool(text: str) -> Optional[bool]:
    s = normalize(text).lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def is_correct(pred: str, gold: str, kind: str = "text",
               tol: float = 0.05, mode: str = "relaxed") -> bool:
    """判一道题。

    mode="strict"  : 归一化后逐字相等（布尔/数值也先规范成同一写法）
    mode="relaxed" : 数值按相对/绝对容差，布尔按同义词，其余按包含匹配
    """
    if mode not in ("strict", "relaxed"):
        raise ValueError("mode 必须是 strict / relaxed")

    p, g = normalize(pred), normalize(gold)
    if not p:
        return False

    # 严格口径 = 归一化后逐字相等（不做数值容差、不做同义词）
    if mode == "strict":
        return p.lower() == g.lower()

    if kind == "bool":
        pb, gb = canonical_bool(p), canonical_bool(g)
        if pb is not None and gb is not None:
            return pb == gb
        return p.lower() == g.lower()

    if kind == "number":
        pn, gn = to_number(p), to_number(g)
        if pn is None or gn is None:
            return p.lower() == g.lower()
        return abs(pn - gn) <= max(tol * max(abs(gn), 1e-6), 0.05)

    # text / label
    pl, gl = p.lower(), g.lower()
    if pl == gl:
        return True
    # 长句里出现金标、或金标作为独立 token 出现（如"答案是类别 B"里的 B）
    if len(gl) >= 2 and gl in pl:
        return True
    return gl in re.findall(r"[a-z0-9.]+", pl)


def score(pred: str, gold: str, kind: str = "text", tol: float = 0.05) -> dict:
    return {
        "correct_strict": is_correct(pred, gold, kind, tol, "strict"),
        "correct_relaxed": is_correct(pred, gold, kind, tol, "relaxed"),
    }
