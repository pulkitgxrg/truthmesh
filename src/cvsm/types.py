"""Core data records shared across the pipeline.

Names follow the notation of research.md Part B: a query q is decomposed into sub-tasks t_j,
agents a_i produce responses r_ij, D(r_ij) gives atomic claims, claims are aligned into clusters.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


GATEWAYS = {"aicredits"}


@dataclass
class ModelSpec:
    provider: str          # "anthropic" | "openai" | "google" | "ollama" | "fake"
    model: str             # exact model identifier sent to the provider
    supports_temperature: bool = True

    def key(self) -> str:
        return f"{self.provider}:{self.model}"

    def family(self) -> str:
        """Model family (vendor). Gateways serve several vendors under one provider, so the vendor is
        the prefix of the model id there (e.g. aicredits 'openai/gpt-4.1-nano' -> 'openai')."""
        if self.provider in GATEWAYS and "/" in self.model:
            return self.model.split("/", 1)[0]
        return {"ollama": "local"}.get(self.provider, self.provider)


@dataclass
class AgentSpec:
    agent_id: str
    role: str              # role name, e.g. "retriever", "reasoner", "specialist", "critic"
    model: ModelSpec


@dataclass
class Passage:
    pid: str
    title: str
    text: str

    def as_text(self) -> str:
        return f"{self.title}: {self.text}" if self.title else self.text


@dataclass
class Example:
    qid: str
    dataset: str
    question: str
    answer: str | None                      # gold short answer (HotpotQA, HaluEval) or label (FEVER)
    gold_evidence: list[str]                # E*(q): reference evidence sentences for truth labels
    passages: list[Passage] = field(default_factory=list)  # retrieval corpus for this item (if per-item)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubTask:
    sid: str
    description: str
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Claim:
    cid: str
    text: str
    agent_id: str
    subtask: str
    step_id: str
    phase: str                               # draft | revision | debate_r{n} | synthesis | final
    qualifiers: list[str] = field(default_factory=list)
    premises: list[str] = field(default_factory=list)   # memory entry ids this claim relies on


@dataclass
class Step:
    """One agent's output for one sub-task in one phase (Section III-E)."""
    step_id: str
    phase: str
    subtask: str
    agent_id: str
    role: str
    model: str
    context_refs: list[str]                  # "step:<id>" or "entry:<id>" items present in the context
    output_text: str = ""
    claim_ids: list[str] = field(default_factory=list)
    llm_call_keys: list[str] = field(default_factory=list)


def to_dict(obj) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [to_dict(o) for o in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj
