"""Shared Memory Layer (Section IV-H, Table II): append-only, versioned, session-scoped store."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field, asdict
from typing import Callable

ACTIVE, SUPERSEDED, RETRACTED = "active", "superseded", "retracted"


@dataclass
class Entry:
    # Table II fields
    id: str
    text: str
    embedding: list[float]
    conf: float                      # Conf*(c) at write time
    label: str                       # Corroborated | Singleton-grounded | Resolved (| Unverified under A2/B6)
    support: list[dict]              # independent supporters: agent, role, base model
    evidence: list[dict]             # passage ids with NLI scores used for G(c)
    premises: list[str]              # ids of entries this claim depends on
    qualifiers: list[str]
    subtask: str
    version: int = 1
    status: str = ACTIVE
    created_at: str = ""
    # bookkeeping used by re-verification and trace metrics (not part of Table II)
    A: float = 0.0
    M: float = 0.0
    G: float = 0.0
    source_claims: list[str] = field(default_factory=list)
    source_steps: list[str] = field(default_factory=list)
    note: str = ""


class SharedMemory:
    """Entries are never deleted; a status change appends a new version.

    `log` is the full version history; the write index t counts every appended version (Eq. 7)."""

    def __init__(self, trace=None):
        self.log: list[Entry] = []
        self._current: dict[str, Entry] = {}
        self.trace = trace
        self._n = 0

    @property
    def t(self) -> int:
        return len(self.log)

    def _append(self, e: Entry, op: str) -> Entry:
        e.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.log.append(e)
        self._current[e.id] = e
        if self.trace is not None:
            self.trace.event("memory_write", op=op, t=self.t, entry_id=e.id, version=e.version,
                             status=e.status, text=e.text, conf=e.conf, label=e.label,
                             premises=list(e.premises), subtask=e.subtask,
                             source_claims=list(e.source_claims), source_steps=list(e.source_steps),
                             active_ids=[x.id for x in self.active()])
        return e

    def new_id(self) -> str:
        self._n += 1
        return f"m{self._n}"

    def append(self, e: Entry) -> Entry:
        if e.id in self._current:
            raise ValueError("append-only: use set_status to change an existing entry")
        e.version, e.status = 1, ACTIVE
        return self._append(e, "append")

    def set_status(self, entry_id: str, status: str, note: str = "") -> Entry:
        new = copy.deepcopy(self._current[entry_id])
        new.version += 1
        new.status, new.note = status, note
        return self._append(new, f"status:{status}")

    def get(self, entry_id: str) -> Entry:
        return self._current[entry_id]

    def active(self, subtasks: set[str] | None = None) -> list[Entry]:
        return [e for e in self._current.values()
                if e.status == ACTIVE and (subtasks is None or e.subtask in subtasks)]

    def dependents(self, entry_id: str) -> list[Entry]:
        return [e for e in self.active() if entry_id in e.premises]

    def retract_transitively(self, entry_id: str, reverify: Callable[[Entry], bool]) -> list[str]:
        """After `entry_id` is superseded or retracted, flag every active descendant for re-verification.

        `reverify(e)` returns True if e still satisfies the write rule; otherwise e is retracted and its
        own descendants are flagged in turn. Returns the ids retracted."""
        retracted: list[str] = []
        queue = [entry_id]
        while queue:
            parent = queue.pop(0)
            for child in self.dependents(parent):
                if self.trace is not None:
                    self.trace.event("flag_reverify", entry_id=child.id, because=parent)
                if not reverify(child):
                    self.set_status(child.id, RETRACTED, note=f"premise {parent} no longer active")
                    retracted.append(child.id)
                    queue.append(child.id)
        return retracted

    def snapshot(self) -> list[dict]:
        return [asdict(e) | {"embedding": None} for e in self.log]
