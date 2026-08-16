"""The model provider interface.

Every model call in the system goes through this, for three reasons that all pay off
later: swapping providers stays a config change, the test suite can substitute a
deterministic stub and run with no key, and there is exactly one place to instrument
for cost and latency (behavior 10).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class ProviderError(RuntimeError):
    """Raised when a provider cannot serve a request. Carries enough context that the
    operator knows what to do next, rather than a bare status code."""


@dataclass(frozen=True)
class Completion:
    """One model response, plus everything the metering layer needs."""

    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class ModelProvider(ABC):
    """Implementations must be safe to construct without network access — all
    validation belongs in `healthcheck()`, so that importing the app never blocks."""

    name: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        cheap: bool = False,
        temperature: float = 0.0,
    ) -> Completion:
        """Return a completion. When `schema` is supplied the provider must return
        JSON conforming to it — extraction never parses prose.

        `cheap=True` selects the low-cost model for high-volume passes.
        """

    @abstractmethod
    def healthcheck(self) -> dict[str, Any]:
        """Verify this provider can actually serve requests *now*.

        Raises ProviderError with actionable detail on failure. Called at startup and
        by GET /health, so a broken key surfaces immediately rather than eight stages
        into a run.
        """
