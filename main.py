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
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from database import db
from database.config import get_settings
from domain.ingest import SUPPORTED_SUFFIXES
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

    # Nothing can be mid-run while this process is still booting, so any row saying so
    # is a run whose process died. Marked before the API serves its first request —
    # otherwise the first caller reads a status that was never true.
    service.mark_interrupted_runs()

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
        known=service.known_document_hashes,
    )

    # Primed unconditionally, not only when the timer is enabled.
    #
    # Priming used to happen inside `start()`, which only runs with `watch_enabled=true`
    # — off by default. So on the shipped configuration `/watch/poll` came up with empty
    # state after every restart and reported the entire inbox as newly added, starting a
    # run for it. Cheap (facts are reused, no model calls) and false, which is the worse
    # half.
    _watcher.prime()

    if settings.watch_enabled:
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


# Module-level so the call is not evaluated in a default argument on every request.
_UPLOADED_FILES = File(...)


@app.post("/documents/upload")
async def upload_documents(files: list[UploadFile] = _UPLOADED_FILES) -> dict[str, Any]:
    """Accept documents through the browser and hand them to the watcher.

    Deliberately *not* a second ingestion path: the file lands in the watched
    directory and the existing poll picks it up, so an upload and a file dropped into
    the folder converge on the same code within one line of each other. A parallel
    "upload run" would be a second path to keep correct, and the one used less often
    is the one that rots.
    """
    if _watcher is None:  # pragma: no cover - startup always creates it
        raise HTTPException(status_code=503, detail="watcher not initialised")

    directory = get_settings().watch_dir
    directory.mkdir(parents=True, exist_ok=True)

    saved: list[str] = []
    for upload in files:
        # basename() only — an uploaded name is caller-controlled, and "../../etc/x"
        # is a path traversal write, not a document.
        name = Path(upload.filename or "").name
        if not name:
            raise HTTPException(status_code=400, detail="a file arrived with no name")

        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{name}: unsupported format '{suffix or 'none'}'. "
                    f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}."
                ),
            )

        (directory / name).write_bytes(await upload.read())
        saved.append(name)

    log(logger, logging.INFO, "documents uploaded", files=", ".join(saved))

    # Return as soon as the run has an id, not when the run finishes.
    #
    # Holding the request open for the whole run was honest but unusable: a cold corpus
    # is a minute of a dropzone saying "Running…", and the browser cannot show the
    # stages it is waiting on because it is blocked on the same request. The run id is
    # available the moment the row is inserted, and the page already polls a running
    # run — so handing back the id turns a blocking wait into a live one.
    started: dict[str, str] = {}
    has_id = threading.Event()

    def _announce(run_id: str) -> None:
        started["run_id"] = run_id
        has_id.set()

    outcome: dict[str, Any] = {}

    def _drive() -> None:
        try:
            outcome["result"] = _watcher.poll(on_started=_announce)
        except Exception as exc:
            log(
                logger,
                logging.ERROR,
                "upload-triggered run failed",
                error=type(exc).__name__,
                detail=str(exc)[:300],
            )
        finally:
            # Unblocks the request even when the poll found nothing to do or died
            # before a run existed; otherwise the caller waits out the timeout for an
            # id that is never coming.
            has_id.set()

    threading.Thread(target=_drive, daemon=True, name="ledger-upload").start()

    # Bounded: ingestion is filesystem work and reaches the insert quickly. If it has
    # not by now, answer anyway rather than reintroducing the blocking wait.
    has_id.wait(timeout=15)

    # How many documents the run actually covers, which is not how many were just
    # uploaded.
    #
    # A drop of three files starts a run over the whole watched folder — correct, because
    # the register is over a corpus rather than over an upload, and cheap, because the
    # documents already ingested are reused without a model call. But "3 added — run
    # started" reads as though three documents are being processed, and a reviewer who
    # then watches fourteen classifications go by has been misled by the interface rather
    # than by the system.
    in_corpus = len(
        [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES]
    )

    return {
        "saved": saved,
        "corpus_documents": in_corpus,
        "run_id": started.get("run_id"),
        "status": "running" if started.get("run_id") else "no_run_started",
        "poll": f"/runs/{started['run_id']}" if started.get("run_id") else None,
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


# ---------------------------------------------------------------------------
# The review page, served by the API itself.
#
# Mounted at the very bottom of this file, and that placement is load-bearing:
# StaticFiles with html=True answers *every* unmatched path, so mounting it before the
# routes above are registered would swallow /runs, /health and the rest. Routes are
# matched in registration order.
#
# Serving the built bundle from this process keeps `docker compose up` the one
# documented command. A second container to serve half a dozen static files would buy
# nothing and cost the promise that a fresh clone works in one step.
#
# If the bundle is absent — someone running uvicorn without building the frontend —
# the API simply has no UI. That is an honest absence rather than a broken route.
# ---------------------------------------------------------------------------

_WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"

if _WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
    log(logger, logging.INFO, "review UI mounted at /", bundle=str(_WEB_DIST))
else:
    log(logger, logging.INFO, "no built review UI; API only", looked_in=str(_WEB_DIST))
