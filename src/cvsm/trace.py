"""Execution-trace logging: every step, claim, exposure, decision and memory write (needed for PD,
MCR and CLR, Sections III-C, III-E, III-F)."""
from __future__ import annotations

import itertools
import threading
from dataclasses import asdict

from .types import Claim, Step


class Trace:
    def __init__(self, qid: str):
        self.qid = qid
        self.steps: list[Step] = []
        self.claims: dict[str, Claim] = {}
        self.events: list[dict] = []
        self._seq = itertools.count()
        self._lock = threading.Lock()

    def new_id(self, prefix: str) -> str:
        with self._lock:
            return f"{prefix}{next(self._seq)}"

    def add_step(self, step: Step) -> Step:
        with self._lock:
            self.steps.append(step)
        self.event("step", step_id=step.step_id, phase=step.phase, subtask=step.subtask,
                   agent=step.agent_id, context_refs=list(step.context_refs))
        return step

    def add_claim(self, claim: Claim) -> Claim:
        with self._lock:
            self.claims[claim.cid] = claim
        return claim

    def event(self, kind: str, **data) -> None:
        with self._lock:
            self.events.append({"seq": len(self.events), "kind": kind, **data})

    def step(self, step_id: str) -> Step:
        return next(s for s in self.steps if s.step_id == step_id)

    def to_dict(self) -> dict:
        return {"qid": self.qid, "steps": [asdict(s) for s in self.steps],
                "claims": {k: asdict(v) for k, v in self.claims.items()}, "events": self.events}
