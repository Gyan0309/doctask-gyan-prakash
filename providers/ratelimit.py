"""A token-bucket rate limiter for outbound model calls.

Added after a 7-document run died on `429 ... limit: 20` — the Gemini free tier allows
20 requests per minute, and a corpus of any size blows through that in seconds when
requests are issued as fast as they are produced.

Retrying a 429 is the wrong primary fix. Backoff reacts *after* the quota is spent, so
every run still slams into the wall and then waits; and because the quota is per
minute, a backoff long enough to clear it looks indistinguishable from a hang. Pacing
prevents the failure instead of recovering from it, which is the difference between a
system that survives its constraints and one that apologises for them.

Process-wide and thread-safe, because the quota is per *key*, not per run. Two
concurrent runs share one budget, and a limiter scoped per run would let them
cheerfully exceed it together — which is precisely the case that matters, since
concurrent runs are a requirement here.
"""

from __future__ import annotations

import logging
import threading
import time

from utils.logging_config import get_logger, log

logger = get_logger(__name__)


class RateLimiter:
    """Allows `rate_per_minute` acquisitions per rolling minute.

    Implemented as a token bucket rather than a fixed window: a fixed window permits a
    double-rate burst across a boundary (all of minute one's budget at :59, all of
    minute two's at :01), which is exactly the pattern that trips a provider's own
    limiter even though the average looks compliant.
    """

    def __init__(self, rate_per_minute: int) -> None:
        self._capacity = max(1, rate_per_minute)
        self._tokens = float(self._capacity)
        self._refill_per_second = self._capacity / 60.0
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Block until a token is available. Returns seconds spent waiting."""
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity, self._tokens + (now - self._last) * self._refill_per_second
                )
                self._last = now

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return waited

                needed = (1.0 - self._tokens) / self._refill_per_second

            # Sleep outside the lock so other threads can refill and proceed.
            time.sleep(min(needed, 5.0))
            waited += min(needed, 5.0)

            if waited > 0 and waited % 15 < 5:
                # Long waits are legitimate here, but a silent multi-minute pause looks
                # exactly like a hang. Say what is happening.
                log(
                    logger,
                    logging.INFO,
                    "waiting on local rate limit",
                    waited_s=round(waited, 1),
                    limit_per_min=self._capacity,
                )


_limiters: dict[str, RateLimiter] = {}
_registry_lock = threading.Lock()


def get_limiter(name: str, rate_per_minute: int) -> RateLimiter:
    """Fetch or create the shared limiter for a provider key."""
    with _registry_lock:
        limiter = _limiters.get(name)
        if limiter is None or limiter._capacity != max(1, rate_per_minute):
            limiter = RateLimiter(rate_per_minute)
            _limiters[name] = limiter
        return limiter


def reset_limiters() -> None:
    """Tests must not inherit a drained bucket from an earlier test."""
    with _registry_lock:
        _limiters.clear()
