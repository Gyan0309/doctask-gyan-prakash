"""Logging.

Two decisions shape this module, and both come from what these logs are actually for.

**Every line carries its run id.** Behavior 9 requires concurrent runs to stay
isolated, and the first thing anyone does when investigating a concurrency problem is
read the logs. Interleaved lines from two runs with no way to tell them apart make
that investigation impossible — so the run id rides in a contextvar and is stamped
onto every record automatically, rather than depending on each call site to remember
to pass it.

**Human format by default, JSON on request.** A developer reading `docker compose up`
output wants aligned columns. A log aggregator wants JSON. `LOG_FORMAT=json` switches;
neither audience is served by a compromise between the two.

Logs never contain secrets. API keys are redacted at the provider boundary before any
error string is constructed, so a key cannot reach a handler here in the first place.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
_stage: ContextVar[str | None] = ContextVar("stage", default=None)

_CONFIGURED = False

# Third-party loggers that are useful at WARNING and deafening at INFO.
NOISY = {
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "urllib3": logging.WARNING,
    "sqlalchemy.engine": logging.WARNING,
    "alembic.runtime.migration": logging.INFO,
    "psycopg.pool": logging.WARNING,
}


class ContextFilter(logging.Filter):
    """Stamp run/stage context onto every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id.get() or "-"
        record.stage = _stage.get() or "-"
        return True


class HumanFormatter(logging.Formatter):
    """Aligned, scannable output.

    The run id is truncated to eight characters: enough to distinguish concurrent runs
    at a glance, short enough that the message still starts in a predictable column.
    Full ids are always available in the JSON format and in the database.
    """

    LEVEL_WIDTH = 7

    def format(self, record: logging.LogRecord) -> str:
        run = (getattr(record, "run_id", "-") or "-")[:8]
        stage = getattr(record, "stage", "-") or "-"
        origin = record.name.removeprefix("ledger.")
        base = (
            f"{self.formatTime(record, '%H:%M:%S')} "
            f"{record.levelname:<{self.LEVEL_WIDTH}} "
            f"[{run:>8}/{stage:<9}] "
            f"{origin:<22} {record.getMessage()}"
        )
        extras = getattr(record, "extra_fields", None)
        if extras:
            base += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "run_id": getattr(record, "run_id", None),
            "stage": getattr(record, "stage", None),
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}) or {})
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Install handlers. Idempotent — safe to call from the API, the CLI, and tests.

    Without the idempotence guard, uvicorn's reloader and pytest's per-module imports
    both end up adding a second handler, and every line prints twice. That is a
    remarkably annoying bug to chase precisely because it looks like the code ran twice.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    fmt = (fmt or os.environ.get("LOG_FORMAT") or "human").lower()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else HumanFormatter())
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    for name, lvl in NOISY.items():
        logging.getLogger(name).setLevel(lvl)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def log(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    """Log with structured fields that survive into JSON output.

    Using this instead of f-strings everywhere means a value like `facts=12` is a real
    field an aggregator can filter on, not a substring someone has to regex out.
    """
    logger.log(level, message, extra={"extra_fields": fields})


@contextmanager
def run_context(run_id: str | None, stage: str | None = None):
    """Bind run id (and optionally stage) for everything logged inside the block."""
    run_token = _run_id.set(run_id)
    stage_token = _stage.set(stage) if stage is not None else None
    try:
        yield
    finally:
        _run_id.reset(run_token)
        if stage_token is not None:
            _stage.reset(stage_token)


@contextmanager
def stage_context(stage: str):
    token = _stage.set(stage)
    try:
        yield
    finally:
        _stage.reset(token)


@contextmanager
def timed(logger: logging.Logger, message: str, **fields: Any):
    """Log the start and end of an operation, with elapsed milliseconds.

    Latency is recorded on the *failure* path too. A stage that is slow only when it
    errors is a real pattern, and logging duration only on success hides it.
    """
    start = time.monotonic()
    log(logger, logging.DEBUG, f"{message} started", **fields)
    try:
        yield
    except Exception as exc:
        log(
            logger,
            logging.ERROR,
            f"{message} failed",
            elapsed_ms=round((time.monotonic() - start) * 1000),
            error=type(exc).__name__,
            **fields,
        )
        raise
    else:
        log(
            logger,
            logging.INFO,
            f"{message} finished",
            elapsed_ms=round((time.monotonic() - start) * 1000),
            **fields,
        )
