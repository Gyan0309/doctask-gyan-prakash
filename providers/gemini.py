"""Google AI Studio (Gemini) provider.

Two things here are the product of hitting the wall rather than reading the docs, and
both are commented where they bite.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import httpx

from providers.base import Completion, ModelProvider, ProviderError
from providers.ratelimit import get_limiter
from utils.logging_config import get_logger, log

logger = get_logger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# Transient by nature: the same request will likely succeed shortly. Everything else
# — 400, 401, 403, 404 — is a fact about the request and retrying it only wastes time
# and quota while making the real error take longer to surface.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

MAX_ATTEMPTS = 4
BASE_BACKOFF_SECONDS = 1.5

# A 429 is not like the others. The free tier's quota is per *minute*, so a backoff
# measured in seconds is guaranteed to fail again — the window simply has not moved.
# These get their own, longer schedule and a bigger attempt budget.
QUOTA_BACKOFF_SECONDS = 20.0
QUOTA_MAX_ATTEMPTS = 5


def _is_daily_quota(resp: httpx.Response) -> bool:
    """Is this 429 a per-day cap rather than a per-minute burst?

    Google distinguishes them only in the structured `details`, via a quotaId such as
    `GenerateRequestsPerDayPerProjectPerModel-FreeTier`. The human-readable message is
    identical for both, so parsing prose here would be guesswork.
    """
    try:
        for detail in resp.json().get("error", {}).get("details", []):
            if "QuotaFailure" not in detail.get("@type", ""):
                continue
            for violation in detail.get("violations", []):
                if "PerDay" in (violation.get("quotaId") or ""):
                    return True
    except Exception:
        return False
    return False


class GeminiProvider(ModelProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        model_cheap: str,
        *,
        timeout: float = 120.0,
        requests_per_minute: int = 15,
        fallbacks: list[str] | None = None,
    ) -> None:
        if not api_key:
            raise ProviderError(
                "GEMINI_API_KEY is empty. Set it in .env, or set LLM_PROVIDER=fake to "
                "run without a key."
            )
        self._key = api_key
        self._model = model
        self._model_cheap = model_cheap
        self._timeout = timeout
        self._fallbacks = fallbacks or []
        # Models known to be out of daily quota. Held for the life of the process so
        # a run does not re-discover the same exhausted model on every single call —
        # each rediscovery costs a full round trip and a backoff.
        self._exhausted: set[str] = set()
        self._limiter = get_limiter("gemini", requests_per_minute)

    # -- internals -----------------------------------------------------------

    def _url(self, model: str, verb: str = "generateContent") -> str:
        # The AI Studio key is a *query parameter*, not a Bearer token. Sending it as
        # `Authorization: Bearer <key>` returns 401 UNAUTHENTICATED with no hint that
        # the auth *scheme* is the problem — easy to misread as a bad key.
        return f"{API_ROOT}/models/{model}:{verb}?key={self._key}"

    @staticmethod
    def _redact(text: str, key: str) -> str:
        """Google echoes the request URL — key included — into some error bodies. This
        runs on every error path so a key can never reach a log or a traceback."""
        return text.replace(key, "<redacted>") if key else text

    def _post(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._post_with_retries(model, payload)

        if resp.status_code == 404:
            # The trap that cost us an hour: a model can be listed by GET /models and
            # still be uncallable by this key ("no longer available to new users").
            # A bare 404 sends you checking your URL, so we answer the question the
            # operator is about to ask — what should I put in GEMINI_MODEL instead?
            #
            # The suggestion list is filtered to drop the model that just failed.
            # Without that it reappears in its own error message, because the listing
            # endpoint is exactly the source that overstates availability — which is
            # the whole point of this message.
            candidates = [m for m in self.list_models() if m != model]
            suggestion = ", ".join(candidates[:12]) if candidates else "(none returned)"
            raise ProviderError(
                f"Model {model!r} returned 404 — not callable with this key, even "
                f"though GET /v1beta/models may still list it.\n"
                f"Other models this key lists (listing is not proof of access — verify "
                f"before relying on one): {suggestion}\n"
                f"Set GEMINI_MODEL in .env to a model you have confirmed responds."
            )

        if resp.status_code == 429 and _is_daily_quota(resp):
            # Marked so the fallback layer can recognise it. A per-day exhaustion and
            # a per-minute burst both arrive as 429, but they need opposite responses:
            # wait a moment, versus give up on this model entirely. Treating them
            # alike means either abandoning a run that would have recovered, or
            # sleeping through a quota that will not return until tomorrow.
            raise ProviderError(
                f"__QUOTA__ {model} exhausted its free-tier daily request quota: "
                f"{self._redact(resp.text[:250], self._key)}"
            )

        if resp.status_code != 200:
            raise ProviderError(
                f"Gemini HTTP {resp.status_code}: "
                f"{self._redact(resp.text[:400], self._key)}"
            )

        return resp.json()

    def _post_with_fallback(self, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Try the requested model, then each fallback, on daily-quota exhaustion.

        The Gemini free tier limits requests **per day, per model** — 20/day for
        gemini-3.6-flash, measured 2026-08-15. A single run over a seven-document
        corpus exhausts that, and no retry policy can conjure more: the window is
        tomorrow.

        But because the cap is per *model*, a different model is a different budget.
        Falling back is therefore real capacity, not a trick — and the alternative is
        abandoning a partially completed run over a limit that a sibling model would
        have absorbed. The substitution is logged at WARNING, because quietly answering
        with a different model than the operator configured would be its own kind of
        dishonesty.
        """
        chain = [model] + [m for m in self._fallbacks if m != model]
        available = [m for m in chain if m not in self._exhausted] or chain[-1:]

        last_quota_error: ProviderError | None = None

        for index, candidate in enumerate(available):
            try:
                return self._post(candidate, payload)
            except ProviderError as exc:
                if "__QUOTA__" not in str(exc):
                    raise

                self._exhausted.add(candidate)
                last_quota_error = exc
                remaining = available[index + 1 :]

                log(
                    logger,
                    logging.WARNING,
                    "model out of daily quota, falling back"
                    if remaining
                    else "model out of daily quota and no fallback remains",
                    exhausted=candidate,
                    next_model=remaining[0] if remaining else None,
                )

        raise ProviderError(
            f"All configured Gemini models are out of free-tier daily quota "
            f"({', '.join(available)}). The free tier caps requests per day per "
            f"model, so this resets tomorrow rather than in minutes. Options: add "
            f"another model to GEMINI_MODEL_FALLBACKS, enable billing, or run with "
            f"LLM_PROVIDER=fake.\nUnderlying: {last_quota_error}"
        )

    def _post_with_retries(self, model: str, payload: dict[str, Any]) -> httpx.Response:
        """POST with backoff on transient failures.

        Added after a live run died on a 503 "this model is currently experiencing
        high demand" — an upstream condition that resolves on its own in seconds. A
        pipeline that abandons an eight-stage run because a shared model was briefly
        busy is not resilient, it is lucky.

        Backoff is exponential with jitter. The jitter matters more than it looks:
        concurrent runs that all failed on the same upstream blip would otherwise
        retry in lockstep and recreate the spike they are backing off from.

        Only RETRYABLE_STATUS and transport errors retry. A 400 or a 403 is a fact
        about the request, and retrying it burns quota to arrive at the same answer
        more slowly.
        """
        last_error: Exception | None = None
        attempt = 0
        max_attempts = MAX_ATTEMPTS

        while attempt < max_attempts:
            attempt += 1

            # Pace before sending, not after failing. This is what keeps the 429 from
            # happening in the first place.
            waited = self._limiter.acquire()

            started = time.monotonic()
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.post(self._url(model), json=payload)
            except httpx.RequestError as exc:
                last_error = exc
                log(
                    logger,
                    logging.WARNING,
                    "gemini transport error",
                    model=model,
                    attempt=attempt,
                    error=type(exc).__name__,
                )
                if attempt == MAX_ATTEMPTS:
                    raise ProviderError(
                        f"Gemini unreachable after {MAX_ATTEMPTS} attempts: "
                        f"{type(exc).__name__}"
                    ) from exc
                self._sleep_before_retry(attempt)
                continue

            elapsed_ms = round((time.monotonic() - started) * 1000)

            if resp.status_code == 429:
                if _is_daily_quota(resp):
                    # Waiting is futile — the window is tomorrow. Return immediately
                    # so the fallback layer can switch models, which is the only
                    # thing that actually helps. Retrying here would spend two
                    # minutes proving what the quotaId already said.
                    log(
                        logger,
                        logging.WARNING,
                        "daily quota exhausted; not retrying, deferring to fallback",
                        model=model,
                    )
                    return resp

                # A per-minute burst, which *does* clear. Extend the attempt budget so
                # the window can actually elapse — giving up after ten seconds on a
                # limit that resets in sixty throws away work that would have landed.
                max_attempts = QUOTA_MAX_ATTEMPTS

            if resp.status_code in RETRYABLE_STATUS and attempt < max_attempts:
                delay = self._retry_delay(resp, attempt)
                log(
                    logger,
                    logging.WARNING,
                    "gemini quota exceeded, backing off"
                    if resp.status_code == 429
                    else "gemini transient failure, retrying",
                    model=model,
                    status=resp.status_code,
                    attempt=attempt,
                    of=max_attempts,
                    retry_in_s=round(delay, 1),
                    paced_wait_s=round(waited, 1),
                    elapsed_ms=elapsed_ms,
                )
                time.sleep(delay)
                continue

            log(
                logger,
                logging.INFO if resp.status_code == 200 else logging.WARNING,
                "gemini call complete",
                model=model,
                status=resp.status_code,
                attempt=attempt,
                paced_wait_s=round(waited, 1),
                elapsed_ms=elapsed_ms,
            )
            return resp

        raise ProviderError(  # pragma: no cover - loop always returns or raises above
            f"Gemini failed after {max_attempts} attempts: {last_error}"
        )

    @staticmethod
    def _retry_delay(resp: httpx.Response, attempt: int) -> float:
        """Honour Retry-After when the server sends it; it knows better than we do."""
        header = resp.headers.get("Retry-After")
        if header:
            try:
                return min(float(header), 65.0)
            except ValueError:
                pass

        if resp.status_code == 429:
            # Linear, not exponential, and starting high: the quota window is a fixed
            # 60 seconds, so the useful question is "has the minute rolled over yet",
            # which doubling answers far too slowly.
            return min(QUOTA_BACKOFF_SECONDS * attempt, 65.0) * (0.8 + 0.4 * random.random())

        return BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) * (0.5 + random.random())

    @staticmethod
    def _sleep_before_retry(attempt: int) -> None:
        time.sleep(BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)) * (0.5 + random.random()))

    # -- interface -----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        cheap: bool = False,
        temperature: float = 0.0,
    ) -> Completion:
        model = self._model_cheap if cheap else self._model

        generation_config: dict[str, Any] = {"temperature": temperature}
        if schema is not None:
            # Structured output. Without responseMimeType the model wraps JSON in
            # ```json fences and the parse fails intermittently — intermittently
            # being the worst kind, because it passes review and fails in the demo.
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = schema

        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }

        data = self._post_with_fallback(model, body)

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            # Safety blocks and recitation stops both produce a 200 with no parts.
            # Treating that as an empty string would silently poison extraction.
            finish = (data.get("candidates") or [{}])[0].get("finishReason", "unknown")
            raise ProviderError(
                f"Gemini returned 200 with no usable content (finishReason={finish})."
            ) from exc

        usage = data.get("usageMetadata", {})
        return Completion(
            text=text,
            model=model,
            tokens_in=usage.get("promptTokenCount", 0),
            tokens_out=usage.get("candidatesTokenCount", 0),
            raw=data,
        )

    def list_models(self) -> list[str]:
        """Models supporting generateContent. Best-effort: used to build a helpful
        error message, so it must never raise from inside an error path."""
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(f"{API_ROOT}/models?key={self._key}&pageSize=200")
            resp.raise_for_status()
            return [
                m["name"].removeprefix("models/")
                for m in resp.json().get("models", [])
                if "generateContent" in m.get("supportedGenerationMethods", [])
            ]
        except Exception:
            return []

    def healthcheck(self) -> dict[str, Any]:
        """Prove both configured models are callable, with a real structured-output
        request. Listing them is not proof; calling them is."""
        probe_schema = {
            "type": "OBJECT",
            "properties": {"ok": {"type": "BOOLEAN"}},
            "required": ["ok"],
        }
        checked = {}
        for label, model in (("primary", self._model), ("cheap", self._model_cheap)):
            completion = self.generate(
                'Reply with {"ok": true} and nothing else.',
                schema=probe_schema,
                cheap=(label == "cheap"),
            )
            checked[label] = {"model": model, "responded": bool(completion.text)}
        return {"provider": self.name, "models": checked}
