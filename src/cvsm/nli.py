"""NLI model (DeBERTa-v3, three-way) and sentence encoder, with call counting and memoisation."""
from __future__ import annotations

import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Sequence

import numpy as np


# One lock for all GPU work in the process: concurrent MPS use from several threads (NLI and encoder,
# or two NLI models) crashes Metal ("A command encoder is already encoding to this command buffer").
DEVICE_LOCK = threading.RLock()


@dataclass(frozen=True)
class NLIScore:
    ent: float
    neu: float
    con: float


class NLIModel:
    def __init__(self, checkpoint: str, device: str | None = None, batch_size: int = 16, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.checkpoint = checkpoint
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.idx = (labels["entailment"], labels["neutral"], labels["contradiction"])
        self.batch_size, self.max_length = batch_size, max_length
        self.revision = _snapshot_revision(checkpoint)
        self._memo: dict[tuple[str, str], NLIScore] = {}
        self._lock = DEVICE_LOCK
        self.computed = 0

    def scores(self, pairs: Sequence[tuple[str, str]]) -> list[NLIScore]:
        """pairs: (premise, hypothesis). Returns P(ent), P(neu), P(con) for premise => hypothesis."""
        import torch
        with self._lock:
            todo = sorted({p for p in pairs if p not in self._memo})
            for i in range(0, len(todo), self.batch_size):
                chunk = todo[i:i + self.batch_size]
                enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], truncation="only_first",
                               max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().numpy()
                for pair, pr in zip(chunk, probs):
                    self._memo[pair] = NLIScore(float(pr[self.idx[0]]), float(pr[self.idx[1]]), float(pr[self.idx[2]]))
                self.computed += len(chunk)
            return [self._memo[p] for p in pairs]


def _snapshot_revision(name: str) -> str | None:
    """Commit hash of the locally cached Hugging Face snapshot (recorded in environment.json)."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        ref = Path(HF_HUB_CACHE) / f"models--{name.replace('/', '--')}" / "refs" / "main"
        return ref.read_text().strip() if ref.exists() else None
    except Exception:
        return None


class Encoder:
    def __init__(self, name: str, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer
        self.name = name
        self.model = SentenceTransformer(name, device=device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.revision = _snapshot_revision(name)
        self._memo: dict[str, np.ndarray] = {}
        self._lock = DEVICE_LOCK

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            todo = [t for t in dict.fromkeys(texts) if t not in self._memo]
            if todo:
                vecs = self.model.encode(todo, normalize_embeddings=True, convert_to_numpy=True)
                for t, v in zip(todo, vecs):
                    self._memo[t] = v
            if not texts:
                return np.zeros((0, 1))
            return np.stack([self._memo[t] for t in texts])


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    return a @ b.T


class CountingNLI:
    """Wraps an NLI model and charges every requested pair to a ledger (cost metric 'NLI calls')."""

    def __init__(self, nli, ledger):
        self.nli, self.ledger = nli, ledger

    def scores(self, pairs):
        if self.ledger is not None:
            self.ledger.add_nli(len(pairs))
        return self.nli.scores(pairs)
