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

from ledger.logging_config import get_logger, log
from ledger.providers.base import Completion, ModelProvider, ProviderError

logger = get_logger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

# Transient by nature: the same request will likely succeed shortly. Everything else
# — 400, 401, 403, 404 — is a fact about the request and retrying it only wastes time
# and quota while making the real error take longer to surface.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

MAX_ATTEMPTS = 4
BASE_BACKOFF_SECONDS = 1.5


class GeminiProvider(ModelProvider):
    name = "gemini"

    def __init__(
        self,
        api_key: str,
        model: str,
        model_cheap: str,
        *,
        timeout: float = 120.0,
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

        if resp.status_code != 200:
            raise ProviderError(
                f"Gemini HTTP {resp.status_code}: "
                f"{self._redact(resp.text[:400], self._key)}"
            )

        return resp.json()

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

        for attempt in range(1, MAX_ATTEMPTS + 1):
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

            if resp.status_code in RETRYABLE_STATUS and attempt < MAX_ATTEMPTS:
                delay = self._retry_delay(resp, attempt)
                log(
                    logger,
                    logging.WARNING,
                    "gemini transient failure, retrying",
                    model=model,
                    status=resp.status_code,
                    attempt=attempt,
                    of=MAX_ATTEMPTS,
                    retry_in_s=round(delay, 2),
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
                elapsed_ms=elapsed_ms,
            )
            return resp

        raise ProviderError(  # pragma: no cover - loop always returns or raises above
            f"Gemini failed after {MAX_ATTEMPTS} attempts: {last_error}"
        )

    @staticmethod
    def _retry_delay(resp: httpx.Response, attempt: int) -> float:
        """Honour Retry-After when the server sends it; it knows better than we do."""
        header = resp.headers.get("Retry-After")
        if header:
            try:
                return min(float(header), 30.0)
            except ValueError:
                pass
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

        data = self._post(model, body)

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
