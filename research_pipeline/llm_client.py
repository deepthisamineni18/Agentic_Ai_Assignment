"""
Shared real-LLM client used by the pipelines that are spec'd to call an
actual language model (e.g. the RAG Conversational Agent's "generates a
grounded response" step).

Every pipeline in this repo works with ZERO LLM calls out of the box —
everything falls back to template/keyword-based logic so `make run` etc.
work with no API key and no internet dependency beyond what's already
required. This client is an opt-in upgrade: set either ANTHROPIC_API_KEY or
GROQ_API_KEY and the RAG chat (and anything else wired to use it) will use
real model calls for generation instead of the template fallback. Unset it,
or leave it unset, and behavior is unchanged from before.

This is intentionally a thin wrapper, not a framework: one `generate()`
call, clear availability check, and every call site is expected to catch
exceptions from it and fall back to the template path rather than let a
network hiccup break the pipeline.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger("LLMClient")

try:
    import anthropic
    _ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    _ANTHROPIC_SDK_AVAILABLE = False

try:
    from openai import OpenAI
    _OPENAI_SDK_AVAILABLE = True
except ImportError:
    OpenAI = None
    _OPENAI_SDK_AVAILABLE = False

DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
DEFAULT_GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def _build_anthropic_client(api_key: str):
    if not _ANTHROPIC_SDK_AVAILABLE:
        return None
    try:
        return anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.warning("Failed to initialize Anthropic client: %s", e)
        return None


def _build_groq_client(api_key: str):
    if not _OPENAI_SDK_AVAILABLE:
        return None
    try:
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    except Exception as e:
        logger.warning("Failed to initialize Groq client: %s", e)
        return None


class LLMClient:
    """Thin wrapper around Anthropic or Groq-compatible chat APIs.

    Usage:
        llm = LLMClient()
        if llm.is_available():
            text = llm.generate(system="...", user="...")
        else:
            text = template_fallback(...)
    """

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = model or self._default_model_for_provider()
        self._provider = self._resolve_provider(api_key)
        self._api_key = api_key or self._api_key_for_provider(self._provider)
        self._client = None
        if self._provider == "groq" and self._api_key:
            self._client = _build_groq_client(self._api_key)
        elif self._provider == "anthropic" and self._api_key:
            self._client = _build_anthropic_client(self._api_key)

    def _resolve_provider(self, api_key: str | None) -> str | None:
        if os.environ.get("GROQ_API_KEY"):
            return "groq"
        if os.environ.get("ANTHROPIC_API_KEY") or api_key:
            return "anthropic"
        return None

    def _default_model_for_provider(self) -> str:
        if os.environ.get("GROQ_API_KEY"):
            return DEFAULT_GROQ_MODEL
        return DEFAULT_ANTHROPIC_MODEL

    def _api_key_for_provider(self, provider: str | None) -> str | None:
        if provider == "groq":
            return os.environ.get("GROQ_API_KEY")
        if provider == "anthropic":
            return os.environ.get("ANTHROPIC_API_KEY")
        return None

    def is_available(self) -> bool:
        return self._client is not None

    def generate(self, system: str, user: str, max_tokens: int = 600, temperature: float = 0.2) -> str:
        """Raises on any failure (missing client, API error, timeout) — callers
        are expected to catch and fall back to their template path. This
        class never silently returns a fake/placeholder answer."""
        text, _in_tok, _out_tok = self.generate_with_usage(
            system=system, user=user, max_tokens=max_tokens, temperature=temperature)
        return text

    def generate_with_usage(
        self, system: str, user: str, max_tokens: int = 600, temperature: float = 0.2,
    ) -> tuple[str, int, int]:
        """Same as generate(), but also returns (input_tokens, output_tokens) from
        the provider's reported usage, so callers that need real token accounting
        (e.g. a token-budgeted agent) don't have to estimate it themselves.
        Raises on any failure — callers should catch and fall back."""
        if not self.is_available():
            raise RuntimeError(
                "LLMClient is not available: set ANTHROPIC_API_KEY or GROQ_API_KEY (and install the relevant SDK) to enable it."
            )
        if self._provider == "groq":
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            message = response.choices[0].message
            content = getattr(message, "content", "")
            if isinstance(content, list):
                text = "".join(part.text if hasattr(part, "text") else str(part) for part in content)
            else:
                text = content or ""
            usage = getattr(response, "usage", None)
            in_tok = getattr(usage, "prompt_tokens", 0) or 0
            out_tok = getattr(usage, "completion_tokens", 0) or 0
            return text, in_tok, out_tok

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        return text, in_tok, out_tok
