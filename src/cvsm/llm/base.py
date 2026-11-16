"""Single provider-agnostic LLM interface with a content-addressed disk cache.

Every call is keyed by sha256 over (provider, model, system, messages, temperature, max_tokens, tag).
`tag` distinguishes stochastic samples (run seed, agent, sample index) so that repeated runs draw
distinct samples while any single run can be recomputed exactly from the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Protocol

from ..types import ModelSpec
from .pricing import cost_usd


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float            # wall-clock latency of the original (uncached) API call
    cached: bool
    key: str
    model_key: str
    model_returned: str = ""
    stop_reason: str = ""
    created_at: str = ""


@dataclass
class Ledger:
    """Per-query accounting of LLM calls (system cost, Section V-E)."""
    calls: list[dict] = field(default_factory=list)
    nli_pairs: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, component: str, res: LLMResult) -> None:
        with self._lock:
            self.calls.append({
                "component": component, "key": res.key, "model": res.model_key,
                "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
                "latency_s": res.latency_s, "cached": res.cached,
            })

    def add_nli(self, n: int) -> None:
        with self._lock:
            self.nli_pairs += n

    def summary(self) -> dict:
        inp = sum(c["input_tokens"] for c in self.calls)
        out = sum(c["output_tokens"] for c in self.calls)
        costs = [cost_usd(c["model"], c["input_tokens"], c["output_tokens"]) for c in self.calls]
        priced = all(c is not None for c in costs)
        return {
            "llm_calls": len(self.calls),
            "input_tokens": inp,
            "output_tokens": out,
            "nli_calls": self.nli_pairs,
            "cost_usd": sum(c for c in costs if c is not None) if priced else None,
            "unpriced_models": sorted({c["model"] for c, k in zip(self.calls, costs) if k is None}),
            "cached_calls": sum(1 for c in self.calls if c["cached"]),
            "by_component": _by_component(self.calls),
        }


def _by_component(calls: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for c in calls:
        d = out.setdefault(c["component"], {"calls": 0, "input_tokens": 0, "output_tokens": 0})
        d["calls"] += 1
        d["input_tokens"] += c["input_tokens"]
        d["output_tokens"] += c["output_tokens"]
    return out


class Provider(Protocol):
    def generate(self, model: ModelSpec, system: str, messages: list[dict], temperature: float | None,
                 max_tokens: int) -> dict: ...


def cache_key(model: ModelSpec, system: str, messages: list[dict], temperature: float | None,
              max_tokens: int, tag: str) -> str:
    payload = {
        "provider": model.provider, "model": model.model, "system": system, "messages": messages,
        "temperature": temperature if model.supports_temperature else None,
        "max_tokens": max_tokens, "tag": tag,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class LLMClient:
    def __init__(self, cache_dir: str | Path, providers: dict[str, Provider] | None = None,
                 offline: bool = False):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.providers: dict[str, Provider] = providers or {}
        self.offline = offline      # if True, a cache miss raises instead of calling an API
        self._locks: dict[str, threading.Lock] = {}
        self._glock = threading.Lock()
        # cap on simultaneous in-flight API requests per provider (gateways enforce concurrency limits)
        self.max_inflight = int(os.environ.get("CVSM_MAX_INFLIGHT", "4"))
        self._sems: dict[str, threading.Semaphore] = {}

    def _provider(self, name: str) -> Provider:
        if name not in self.providers:
            from .providers import make_provider
            self.providers[name] = make_provider(name)
        return self.providers[name]

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def complete(self, model: ModelSpec, system: str, user: str, *, temperature: float | None = 0.0,
                 max_tokens: int = 1024, tag: str = "", ledger: Ledger | None = None,
                 component: str = "") -> LLMResult:
        messages = [{"role": "user", "content": user}]
        key = cache_key(model, system, messages, temperature, max_tokens, tag)
        with self._glock:
            lock = self._locks.setdefault(key, threading.Lock())
        with lock:
            path = self._path(key)
            if path.exists():
                rec = json.loads(path.read_text())
                res = LLMResult(cached=True, key=key, model_key=model.key(), **rec["response"])
            else:
                if self.offline:
                    raise RuntimeError(f"cache miss in offline mode: {key}")
                with self._glock:
                    sem = self._sems.setdefault(model.provider, threading.Semaphore(self.max_inflight))
                with sem:
                    t0 = time.perf_counter()
                    out = self._provider(model.provider).generate(
                        model, system, messages, temperature if model.supports_temperature else None, max_tokens)
                    lat = time.perf_counter() - t0
                resp = {
                    "text": out["text"], "input_tokens": int(out["input_tokens"]),
                    "output_tokens": int(out["output_tokens"]), "latency_s": lat,
                    "model_returned": out.get("model_returned", ""), "stop_reason": out.get("stop_reason", ""),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                }
                rec = {"request": {"provider": model.provider, "model": model.model, "system": system,
                                   "messages": messages, "temperature": temperature,
                                   "max_tokens": max_tokens, "tag": tag},
                       "response": resp}
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
                os.replace(tmp, path)
                res = LLMResult(cached=False, key=key, model_key=model.key(), **resp)
        if ledger is not None:
            ledger.add(component, res)
        return res


class FakeProvider:
    """Deterministic provider for tests: `fn(system, user)` returns the response text."""

    def __init__(self, fn: Callable[[str, str], str]):
        self.fn = fn
        self.n_calls = 0

    def generate(self, model, system, messages, temperature, max_tokens):
        self.n_calls += 1
        user = messages[-1]["content"]
        text = self.fn(system, user)
        return {"text": text, "input_tokens": len((system + user).split()),
                "output_tokens": len(text.split()), "model_returned": model.model, "stop_reason": "end"}
