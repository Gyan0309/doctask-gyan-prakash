"""FastAPI application.

Phase 0 exposes only what proves the system is alive and correctly wired. The run,
decision and deliverable endpoints from DESIGN.md §8 arrive with the stages that back
them — an endpoint that returns a plausible shape backed by nothing is the "present
and broken" failure the brief calls out, and is worse than an honestly absent route.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ledger import db
from ledger.api.routes import router
from ledger.config import get_settings
from ledger.logging_config import configure_logging, get_logger, log, run_context
from ledger.providers import build_provider
from ledger.providers.base import ProviderError

_settings = get_settings()
configure_logging(_settings.log_level, _settings.log_format)
logger = get_logger(__name__)

app = FastAPI(
    title="Ledger",
    description=(
        "Agentic multi-document system over a vendor contract corpus: extracts cited "
        "facts, detects cross-document contradictions, and maintains a living "
        "obligation register under human review."
    ),
    version="0.1.0",
)

app.include_router(router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One line per request, with a correlation id.

    The id is echoed back as `X-Request-ID` so a user reporting "my run failed" can
    hand over a token that finds every line the request produced — including lines
    emitted deep inside a graph node.
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    started = time.monotonic()

    with run_context(request_id):
        try:
            response = await call_next(request)
        except Exception as exc:
            log(
                logger,
                logging.ERROR,
                "request failed",
                method=request.method,
                path=request.url.path,
                error=type(exc).__name__,
                elapsed_ms=round((time.monotonic() - started) * 1000),
            )
            raise

        log(
            logger,
            # 5xx is our fault and belongs at ERROR; 4xx is the caller's and does not.
            logging.ERROR if response.status_code >= 500 else logging.INFO,
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=round((time.monotonic() - started) * 1000),
        )
        response.headers["X-Request-ID"] = request_id
        return response


@app.on_event("startup")
async def announce_startup() -> None:
    settings = get_settings()
    log(
        logger,
        logging.INFO,
        "ledger api ready",
        version=app.version,
        provider=settings.llm_provider,
        model=settings.gemini_model,
        log_format=settings.log_format,
    )


@app.get("/health")
def health() -> JSONResponse:
    """Liveness plus dependency status.

    Deliberately returns 200 with per-dependency detail rather than failing outright,
    so `docker compose up` on a machine with no API key still comes up and tells you
    precisely what is missing. Degrade with an explanation, never die silently.
    """
    settings = get_settings()
    report: dict[str, Any] = {"status": "ok", "checks": {}}

    try:
        report["checks"]["database"] = {"ok": db.ping()}
    except Exception as exc:
        report["status"] = "degraded"
        report["checks"]["database"] = {"ok": False, "error": type(exc).__name__}

    try:
        provider = build_provider(settings)
        report["checks"]["provider"] = {
            "ok": True,
            "name": provider.name,
            "configured_model": settings.gemini_model,
            # Not called here: /health must stay fast and free. Use /health/provider
            # for the round trip that actually proves the key works.
            "verified": False,
        }
    except ProviderError as exc:
        report["status"] = "degraded"
        report["checks"]["provider"] = {"ok": False, "error": str(exc)}

    return JSONResponse(report, status_code=200)


@app.get("/health/provider")
def health_provider() -> JSONResponse:
    """Round-trip the model provider for real.

    Separate from /health because this one costs a call and takes seconds. Listing a
    model does not prove your key can call it — this does.
    """
    try:
        provider = build_provider()
        return JSONResponse({"ok": True, **provider.healthcheck()})
    except ProviderError as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


@app.get("/version")
def version() -> dict[str, Any]:
    settings = get_settings()
    return {
        "name": "ledger",
        "version": app.version,
        "provider": settings.llm_provider,
        "model": settings.gemini_model,
        "model_cheap": settings.gemini_model_cheap,
    }
