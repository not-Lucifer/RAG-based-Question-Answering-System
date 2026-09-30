"""Chat model factory: OpenAI (cloud) or Ollama (local LLaMA)."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from app.core.config import get_settings
from app.core.exceptions import LLMUnavailableError
from app.core.logging import get_logger

log = get_logger(__name__)
_override: BaseChatModel | None = None


def set_llm_override(llm: BaseChatModel | None) -> None:
    """Inject a chat model (tests / evaluation). ``None`` restores normal behaviour."""
    global _override
    _override = llm


def get_llm(streaming: bool = False) -> BaseChatModel:
    """Build the chat model selected by ``LLM_PROVIDER``.

    Raises:
        LLMUnavailableError: provider misconfigured (e.g. missing API key) or client init failed.
    """
    if _override is not None:
        return _override
    s = get_settings()
    try:
        if s.llm_provider == "openai":
            if s.openai_api_key is None:
                raise LLMUnavailableError(
                    "OPENAI_API_KEY is not set. Add it to .env or switch LLM_PROVIDER=ollama."
                )
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=s.openai_model,
                api_key=s.openai_api_key,
                temperature=s.llm_temperature,
                max_tokens=s.llm_max_tokens,
                timeout=s.llm_timeout_s,
                max_retries=2,
                streaming=streaming,
            )
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=s.ollama_model,
            base_url=s.ollama_base_url,
            temperature=s.llm_temperature,
            num_predict=s.llm_max_tokens,
        )
    except LLMUnavailableError:
        raise
    except Exception as exc:
        log.error(
            "llm init failed", extra={"fields": {"provider": s.llm_provider, "error": type(exc).__name__}}
        )
        raise LLMUnavailableError(f"Could not initialise the '{s.llm_provider}' chat model.") from exc
