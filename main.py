"""Application entry point.

`uvicorn main:app` — the FastAPI instance, its middleware, the health and watch
endpoints, and the startup wiring. Everything with real logic lives behind it:
`api/` holds the routers, `services/` the orchestration, `domain/` the rules.

An endpoint that returns a plausible shape backed by nothing is the "present and
broken" failure the brief calls out, and is worse than an honestly absent route — so
routes appear here only once the stages behind them exist.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.routes import router
from database import db
from database.config import get_settings
from providers import build_provider
from providers.base import ProviderError
from services import service
from services.watcher import Watcher
from utils.logging_config import configure_logging, get_logger, log, run_context

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


_watcher: Watcher | None = None


@app.on_event("startup")
async def announce_startup() -> None:
    global _watcher
    settings = get_settings()

    log(
        logger,
        logging.INFO,
        "ledger api ready",
        version=app.version,
        provider=settings.llm_provider,
        model=settings.gemini_model,
        log_format=settings.log_format,
        watching=settings.watch_enabled,
    )

    # The watcher instance always exists; `watch_enabled` controls only whether it
    # polls on a timer. The two were conflated at first, so /watch/poll fell back to
    # constructing a throwaway Watcher per request — with empty state, which made every
    # poll report every file as newly added and start a redundant run. The endpoint
    # looked like it worked, because a run did happen each time.
    _watcher = Watcher(
        settings.watch_dir,
        corpus_name=settings.watch_corpus_name,
        interval_seconds=settings.watch_interval_seconds,
        start_run=service.start_run,
    )

    if settings.watch_enabled:
        # `start()` primes first, so a restart does not re-process the whole directory
        # as though it had just arrived.
        _watcher.start()


@app.on_event("shutdown")
async def stop_watcher() -> None:
    if _watcher is not None:
        _watcher.stop()


@app.get("/watch")
def watch_status() -> dict[str, Any]:
    """Whether the watcher is running, and what it is watching."""
    settings = get_settings()
    return {
        # Whether the timer loop is on. The watcher itself always exists, so
        # /watch/poll works either way.
        "enabled": settings.watch_enabled,
        "polling_on_a_timer": settings.watch_enabled and _watcher is not None,
        "directory": str(settings.watch_dir),
        "interval_seconds": settings.watch_interval_seconds,
        "corpus": settings.watch_corpus_name,
    }


@app.post("/watch/poll")
def watch_poll() -> dict[str, Any]:
    """Poll the watched directory once, now.

    Exists so the folder-drop behaviour can be demonstrated and driven by a machine
    without waiting on a timer — and so an operator can ask "did you see my file?"
    and get an answer rather than a shrug.
    """
    if _watcher is None:  # pragma: no cover - startup always creates it
        raise HTTPException(status_code=503, detail="watcher not initialised")

    result = _watcher.poll()
    return {
        "added": result.added,
        "modified": result.modified,
        "removed": result.removed,
        "triggered": result.triggered,
        "run_id": result.run_id,
    }


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
