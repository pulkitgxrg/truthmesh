"""Deterministic stand-ins for the NLI model, encoder and LLM used in unit tests."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from cvsm.nli import NLIScore


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


class FakeNLI:
    """Entailment if the hypothesis (normalised) is a substring of the premise; contradiction for listed pairs
    of key phrases; otherwise neutral. `table` overrides exact (premise, hypothesis) pairs."""

    def __init__(self, contradictions=(), table=None):
        self.contradictions = [(_norm(a), _norm(b)) for a, b in contradictions]
        self.table = table or {}
        self.computed = 0

    def scores(self, pairs):
        out = []
        for p, h in pairs:
            self.computed += 1
            if (p, h) in self.table:
                e, n, c = self.table[(p, h)]
                out.append(NLIScore(e, n, c)); continue
            pn, hn = _norm(p), _norm(h)
            if hn and hn in pn:
                out.append(NLIScore(0.95, 0.04, 0.01))
            elif any((a in pn and b in hn) or (b in pn and a in hn) for a, b in self.contradictions):
                out.append(NLIScore(0.01, 0.04, 0.95))
            else:
                out.append(NLIScore(0.02, 0.96, 0.02))
        return out


class FakeEncoder:
    """Hashed bag-of-words embeddings (normalised), so overlapping sentences have high cosine."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts):
        vecs = []
        for t in texts:
            v = np.zeros(self.dim)
            for w in _norm(t).split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
            n = np.linalg.norm(v)
            vecs.append(v / n if n else v)
        return np.stack(vecs) if vecs else np.zeros((0, self.dim))


def extraction_response(user: str) -> str:
    """Fake claim extractor: each sentence of each [Rn] response's EXPLANATION (or body) is one claim."""
    out = {}
    for m in re.finditer(r"\[(R\d+)\] (.*?)(?=\n\n\[R\d+\] |\n\nFor each response|$)", user, re.S):
        lab, body = m.group(1), m.group(2)
        e = re.search(r"EXPLANATION:\s*(.*)", body, re.S)
        body = e.group(1) if e else body
        sents = [s.strip() for s in re.split(r"(?<=\.)\s+", body.strip()) if s.strip()]
        prem = re.findall(r"\[(E\d+)\]", user.split("Responses:")[0])
        out[lab] = [{"claim": s if s.endswith(".") else s + ".", "qualifiers": [], "premises": prem} for s in sents]
    return json.dumps(out)
