"""Provider selection. The only place that knows which implementations exist."""

from __future__ import annotations

from ledger.config import Settings, get_settings
from ledger.providers.base import Completion, ModelProvider, ProviderError
from ledger.providers.fake import FakeProvider
from ledger.providers.gemini import GeminiProvider

__all__ = [
    "Completion",
    "FakeProvider",
    "GeminiProvider",
    "ModelProvider",
    "ProviderError",
    "build_provider",
]


def build_provider(settings: Settings | None = None) -> ModelProvider:
    """Construct the configured provider.

    Never performs network I/O — a bad key must not stop the app from importing or
    the /health endpoint from reporting *why* it is unhealthy. Validation lives in
    healthcheck().
    """
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()

    if provider == "fake":
        return FakeProvider()
    if provider == "gemini":
        return GeminiProvider(
            api_key=settings.gemini_api_key or "",
            model=settings.gemini_model,
            model_cheap=settings.gemini_model_cheap,
        )

    raise ProviderError(
        f"Unknown LLM_PROVIDER={settings.llm_provider!r}. Supported: gemini, fake."
    )
