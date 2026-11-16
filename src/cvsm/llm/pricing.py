"""Per-token prices (USD per million tokens), copied from the providers' official pricing pages.

Every entry records its source URL and access date. Do not edit a price without re-checking the
source; cost reports cite these values verbatim.
"""
from __future__ import annotations

import json
from pathlib import Path

ACCESSED = "2026-10-04"

# provider:model -> {"input", "output", "batch_input", "batch_output", "source"}
PRICES: dict[str, dict] = {
    "anthropic:claude-haiku-4-5": {
        "input": 1.00, "output": 5.00, "batch_input": 0.50, "batch_output": 2.50,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing", "accessed": ACCESSED,
    },
    "anthropic:claude-sonnet-5-5": {
        "input": 2.00, "output": 10.00, "batch_input": 1.00, "batch_output": 5.00,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing", "accessed": ACCESSED,
    },
    "anthropic:claude-opus-5-5": {
        "input": 4.00, "output": 20.00, "batch_input": 2.00, "batch_output": 10.00,
        "source": "https://platform.claude.com/docs/en/about-claude/pricing", "accessed": ACCESSED,
    },
    "openai:gpt-5-nano": {
        "input": 0.05, "output": 0.40, "batch_input": 0.025, "batch_output": 0.20,
        "source": "https://developers.openai.com/api/docs/pricing", "accessed": ACCESSED,
    },
    "openai:gpt-4.1-nano": {
        "input": 0.10, "output": 0.40, "batch_input": 0.05, "batch_output": 0.20,
        "source": "https://developers.openai.com/api/docs/pricing", "accessed": ACCESSED,
    },
    "openai:gpt-4o-mini": {
        "input": 0.15, "output": 0.60, "batch_input": 0.075, "batch_output": 0.30,
        "source": "https://developers.openai.com/api/docs/pricing", "accessed": ACCESSED,
    },
    "google:gemini-3.1-flash-lite": {
        "input": 0.25, "output": 1.50, "batch_input": 0.125, "batch_output": 0.75,
        "source": "https://ai.google.dev/gemini-api/docs/pricing", "accessed": ACCESSED,
    },
    "google:gemini-3.5-flash-lite": {
        "input": 0.30, "output": 2.50, "batch_input": 0.15, "batch_output": 1.25,
        "source": "https://ai.google.dev/gemini-api/docs/pricing", "accessed": ACCESSED,
    },
    # Local open-weight models served by Ollama have no per-token price.
    "ollama:gemma3:4b": {"input": 0.0, "output": 0.0, "batch_input": 0.0, "batch_output": 0.0,
                         "source": "local (Ollama)", "accessed": ACCESSED},
}


def cost_usd(model_key: str, input_tokens: int, output_tokens: int, batch: bool = False) -> float | None:
    """Return the USD cost, or None when the model has no recorded price (never guessed)."""
    p = PRICES.get(model_key)
    if p is None:
        return None
    i, o = (p["batch_input"], p["batch_output"]) if batch else (p["input"], p["output"])
    return (input_tokens * i + output_tokens * o) / 1_000_000


def load_aicredits_prices(path: Path | None = None) -> str | None:
    """Register prices for `aicredits:<model id>` from a dated snapshot of the gateway's /v1/models listing
    (per-token USD prices as published by the gateway). Returns the snapshot file used."""
    d = Path(__file__).resolve().parents[3] / "data" / "pricing"
    files = sorted(d.glob("aicredits_models_*.json")) if path is None else [path]
    if not files:
        return None
    f = files[-1]
    accessed = f.stem.rsplit("_", 1)[-1]
    for m in json.loads(f.read_text())["data"]:
        if m.get("input_cost_per_token") is None:
            continue
        PRICES[f"aicredits:{m['id']}"] = {
            "input": m["input_cost_per_token"] * 1e6, "output": m["output_cost_per_token"] * 1e6,
            "batch_input": None, "batch_output": None,
            "source": f"https://api.aicredits.in/v1/models (snapshot {f.name})", "accessed": accessed}
    return str(f)


AICREDITS_PRICE_SNAPSHOT = load_aicredits_prices()
