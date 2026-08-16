"""Database engine and session management."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from database.config import get_settings
from utils.logging_config import get_logger, log

logger = get_logger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def _safe_dsn(url: str) -> str:
    """Host and database only. A DSN carries a password, and a connection string in a
    log line is the same disclosure as a key in a log line."""
    tail = url.rsplit("@", 1)[-1]
    return tail or "<unparseable>"


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        settings = get_settings()
        log(
            logger,
            logging.INFO,
            "opening database engine",
            target=_safe_dsn(settings.database_url),
        )
        _engine = create_engine(
            settings.database_url,
            # pool_pre_ping matters more than usual here: a run can sit parked at the
            # human gate for hours, and Postgres will have dropped the connection by
            # the time someone clicks approve. Without this, resuming a gate that
            # worked perfectly yesterday fails on a stale socket.
            pool_pre_ping=True,
            future=True,
        )
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionFactory


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on any exception.

    Never swallows the exception — a failed write must propagate, because a run that
    reports success while its transaction rolled back is exactly the "success message
    that does not mean what it says" failure this system is judged on (I5).
    """
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def ping() -> bool:
    """Cheap liveness probe for /health."""
    with get_engine().connect() as conn:
        return conn.execute(text("SELECT 1")).scalar_one() == 1


def reset_engine() -> None:
    """Drop cached engine/session factory. Tests point at a different database than
    the process started with, and a cached engine would silently ignore that."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
