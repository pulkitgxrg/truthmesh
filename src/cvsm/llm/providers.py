"""Concrete provider adapters. Each returns {"text", "input_tokens", "output_tokens", ...}.

Keys are read from the environment (or the git-ignored .env at the project root): ANTHROPIC_API_KEY,
OPENAI_API_KEY, GEMINI_API_KEY, AICREDITS_API_KEY. Ollama and the AICredits gateway are reached through their
OpenAI-compatible endpoints.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..types import ModelSpec


class AnthropicProvider:
    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=6)

    def generate(self, model: ModelSpec, system, messages, temperature, max_tokens):
        # anthropic SDK 1.x removed `temperature` from the signature; models such as
        # claude-haiku-4-5 still accept it, so it is forwarded through extra_body.
        extra = {"temperature": temperature} if temperature is not None else None
        resp = self.client.messages.create(
            model=model.model, max_tokens=max_tokens, system=system, messages=messages,
            extra_body=extra,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return {"text": text, "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens, "model_returned": resp.model,
                "stop_reason": resp.stop_reason or ""}


class OpenAICompatProvider:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, token_param: str = "max_completion_tokens"):
        import openai
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=6, timeout=120.0)
        self.token_param = token_param

    def generate(self, model: ModelSpec, system, messages, temperature, max_tokens):
        kwargs = {self.token_param: max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = _retry_transient(lambda: self.client.chat.completions.create(
            model=model.model, messages=[{"role": "system", "content": system}, *messages], **kwargs))
        u = resp.usage
        return {"text": resp.choices[0].message.content or "", "input_tokens": u.prompt_tokens,
                "output_tokens": u.completion_tokens, "model_returned": resp.model,
                "stop_reason": resp.choices[0].finish_reason or ""}


def _retry_transient(fn, attempts: int = 5, base_delay: float = 2.0):
    """Gateways can return a bare-router '404 page not found' (or 502) while restarting; the SDK only
    retries 408/409/429/5xx, so router 404s are retried here with exponential backoff."""
    import time
    import openai
    for i in range(attempts):
        try:
            return fn()
        except openai.NotFoundError as e:
            if "page not found" not in str(e) or i == attempts - 1:
                raise
        except (openai.InternalServerError, openai.APIConnectionError):
            if i == attempts - 1:
                raise
        time.sleep(base_delay * 2 ** i)


class GoogleProvider:
    def __init__(self):
        from google import genai
        self.client = genai.Client()

    def generate(self, model: ModelSpec, system, messages, temperature, max_tokens):
        from google.genai import types
        cfg = types.GenerateContentConfig(system_instruction=system, max_output_tokens=max_tokens,
                                          temperature=temperature)
        resp = self.client.models.generate_content(model=model.model, contents=messages[-1]["content"],
                                                   config=cfg)
        um = resp.usage_metadata
        out_tok = (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
        return {"text": resp.text or "", "input_tokens": um.prompt_token_count or 0,
                "output_tokens": out_tok, "model_returned": getattr(resp, "model_version", "") or "",
                "stop_reason": str(resp.candidates[0].finish_reason) if resp.candidates else ""}


def load_dotenv(path: Path = Path(__file__).resolve().parents[3] / ".env") -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


AICREDITS_BASE_URL = "https://api.aicredits.in/v1"


def make_provider(name: str):
    load_dotenv()
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAICompatProvider()
    if name == "google":
        return GoogleProvider()
    if name == "aicredits":       # third-party OpenAI-compatible gateway (https://aicredits.in)
        return OpenAICompatProvider(base_url=AICREDITS_BASE_URL, api_key=os.environ["AICREDITS_API_KEY"],
                                    token_param="max_tokens")       # documented gateway parameter
    if name == "ollama":
        return OpenAICompatProvider(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                                    api_key="ollama")
    raise ValueError(f"unknown provider {name!r}")
