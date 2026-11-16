"""Inter-annotator agreement and labeller accuracy for the Section V-F human-annotation stage."""
from __future__ import annotations

from collections import Counter
from typing import Sequence


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa for two annotators over the same items (any number of categories)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def precision_recall(auto: Sequence[int], human: Sequence[int], positive: int = 0) -> dict:
    """Precision/recall of the automatic labeller against human labels, with `positive` = hallucinated (y=0)."""
    tp = sum(1 for x, y in zip(auto, human) if x == positive and y == positive)
    fp = sum(1 for x, y in zip(auto, human) if x == positive and y != positive)
    fn = sum(1 for x, y in zip(auto, human) if x != positive and y == positive)
    return {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "tp": tp, "fp": fp, "fn": fn, "n": len(auto)}
