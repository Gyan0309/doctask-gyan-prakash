"""The metered model client — one funnel for every model call in the system.

It does three jobs that would otherwise be scattered and inconsistent:

  1. Content-addressed caching, which makes resumption after a kill nearly free
  2. Per-stage cost and latency accounting (behavior 10)
  3. The evidence for "an update costs like an update" — cache hits are counted, so
     the claim is measured rather than asserted

Point 3 is the one that matters most. A resume that correctly skipped completed work
and a resume that silently redid everything both finish successfully and produce the
same output. Only the hit count distinguishes them, which means without this the
central claim of movement 3 is untestable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from models import ModelCacheEntry, StageMetric
from providers.base import Completion, ModelProvider
from utils.hashing import cache_key
from utils.logging_config import get_logger, log

logger = get_logger(__name__)

# Per million tokens. Approximate and clearly labelled — reporting a made-up precise
# figure would be worse than reporting an honest estimate.
PRICE_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.10, 0.40),
}


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICE_PER_MTOK.get(model, (0.0, 0.0))
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


@dataclass
class StageTally:
    stage: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    started: float = field(default_factory=time.monotonic)


class MeteredClient:
    """Wraps a provider with caching and accounting.

    Scoped to one run. Stage is passed per call rather than held as mutable state,
    because a graph node that forgets to reset it would silently bill its work to the
    previous stage — and the resulting cost report would be wrong in a way nobody
    would notice.
    """

    def __init__(
        self,
        provider: ModelProvider,
        session: Session,
        run_id: UUID,
        *,
        prompt_version: str = "v1",
        use_cache: bool = True,
    ) -> None:
        self._provider = provider
        self._session = session
        self._run_id = run_id
        self._prompt_version = prompt_version
        self._use_cache = use_cache
        self.tallies: dict[str, StageTally] = {}

    def _tally(self, stage: str) -> StageTally:
        return self.tallies.setdefault(stage, StageTally(stage=stage))

    def generate(
        self,
        prompt: str,
        *,
        stage: str,
        schema: dict[str, Any] | None = None,
        cheap: bool = False,
    ) -> Completion:
        tally = self._tally(stage)
        model = getattr(self._provider, "model_cheap" if cheap else "model", self._provider.name)

        key = cache_key(
            stage=stage, input_text=prompt, model=str(model), prompt_version=self._prompt_version
        )

        if self._use_cache:
            hit = self._session.execute(
                select(ModelCacheEntry).where(ModelCacheEntry.cache_key == key)
            ).scalar_one_or_none()
            if hit is not None:
                tally.cache_hits += 1
                # DEBUG, not INFO: on a resumed run this fires for every cached call
                # and would drown the lines that matter. The aggregate that proves the
                # incremental claim is logged once per stage in record_stage().
                log(logger, logging.DEBUG, "model cache hit", stage=stage, model=hit.model)
                return Completion(
                    text=hit.response_text,
                    model=hit.model,
                    tokens_in=hit.tokens_in,
                    tokens_out=hit.tokens_out,
                    cached=True,
                )

        tally.cache_misses += 1
        completion = self._provider.generate(prompt, schema=schema, cheap=cheap)

        tally.tokens_in += completion.tokens_in
        tally.tokens_out += completion.tokens_out
        tally.cost_usd += estimate_cost(
            completion.model, completion.tokens_in, completion.tokens_out
        )

        if self._use_cache:
            # ON CONFLICT DO NOTHING, not a pre-check: two concurrent runs on the same
            # corpus legitimately race for the same cache key, and a naive
            # check-then-insert turns that ordinary race into a crash.
            self._session.execute(
                pg_insert(ModelCacheEntry)
                .values(
                    cache_key=key,
                    stage=stage,
                    model=completion.model,
                    prompt_version=self._prompt_version,
                    response_text=completion.text,
                    tokens_in=completion.tokens_in,
                    tokens_out=completion.tokens_out,
                )
                .on_conflict_do_nothing(index_elements=["cache_key"])
            )
            self._session.flush()

        return completion

    def record_stage(self, stage: str, *, skipped: bool = False) -> None:
        """Persist a stage's tally.

        Skipped stages are recorded too, with zero cost. A stage that legitimately did
        not run is evidence the graph took a different path — omitting it would make
        the skip indistinguishable from a stage that was never wired up.
        """
        tally = self.tallies.get(stage) or StageTally(stage=stage)

        # The line that carries behavior 10. Everything needed to answer "what did
        # this stage cost, and did resumption actually save anything" is here, so the
        # claim can be checked from a log tail without opening the database.
        log(
            logger,
            logging.INFO,
            "stage skipped" if skipped else "stage metered",
            stage=stage,
            elapsed_ms=round((time.monotonic() - tally.started) * 1000),
            tokens_in=tally.tokens_in,
            tokens_out=tally.tokens_out,
            cost_usd=round(tally.cost_usd, 6),
            cache_hits=tally.cache_hits,
            cache_misses=tally.cache_misses,
        )

        self._session.add(
            StageMetric(
                run_id=self._run_id,
                stage=stage,
                ended=datetime.now(UTC),
                tokens_in=tally.tokens_in,
                tokens_out=tally.tokens_out,
                cost_usd=tally.cost_usd,
                cache_hits=tally.cache_hits,
                cache_misses=tally.cache_misses,
                skipped=skipped,
            )
        )
        self._session.flush()

    @property
    def total_cache_hits(self) -> int:
        return sum(t.cache_hits for t in self.tallies.values())

    @property
    def total_cache_misses(self) -> int:
        return sum(t.cache_misses for t in self.tallies.values())
