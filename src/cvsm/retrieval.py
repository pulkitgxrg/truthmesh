"""BM25 retrieval over a passage corpus (per-item corpus for HotpotQA distractor; a fixed JSONL
snapshot for datasets that need open retrieval). Same retriever and p for every system (Section V-B)."""
from __future__ import annotations

import re
from typing import Sequence

from .types import Passage

_TOKEN = re.compile(r"\w+")


def _tok(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(s)]


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75). Small and exact; used for per-item corpora."""

    def __init__(self, passages: Sequence[Passage], k1: float = 1.5, b: float = 0.75):
        import math
        self.passages = list(passages)
        self.docs = [_tok(p.as_text()) for p in self.passages]
        self.k1, self.b = k1, b
        n = len(self.docs)
        self.avgdl = sum(map(len, self.docs)) / n if n else 0.0
        df: dict[str, int] = {}
        for d in self.docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [{t: d.count(t) for t in set(d)} for d in self.docs]

    def search(self, query: str, p: int) -> list[Passage]:
        q = _tok(query)
        scored = []
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            s = 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            scored.append((s, i))
        scored.sort(key=lambda x: (-x[0], x[1]))      # deterministic tie-break by corpus order
        return [self.passages[i] for s, i in scored[:p] if s > 0]


class Retriever:
    def __init__(self, passages: Sequence[Passage]):
        self.index = BM25(passages) if passages else None

    def __call__(self, query: str, p: int) -> list[Passage]:
        return self.index.search(query, p) if self.index else []
