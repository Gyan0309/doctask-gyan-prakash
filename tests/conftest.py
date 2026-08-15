"""Shared fixtures.

DATABASE_URL is read from the environment so the same suite runs against the compose
database locally and the services database in CI, with no branching.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Port 55432 matches docker-compose.yml. Not 5432 — see the comment there; a local
# Postgres on the default port will happily accept the connection and then fail auth,
# which reads as "my credentials are wrong" rather than "I reached the wrong server".
DEFAULT_TEST_DB = "postgresql+psycopg://ledger:ledger@localhost:55432/ledger"


@pytest.fixture(scope="session", autouse=True)
def _offline_by_default() -> None:
    """Force the offline provider for the whole suite.

    Autouse and session-scoped so no test can accidentally reach a real API — if a
    developer has a key in their environment, the suite must still be the suite, not
    a slower suite that quietly spends money.
    """
    os.environ["LLM_PROVIDER"] = "fake"
    os.environ.setdefault("DATABASE_URL", DEFAULT_TEST_DB)

    from ledger.config import get_settings

    get_settings.cache_clear()


@pytest.fixture(scope="session")
def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_TEST_DB)


@pytest.fixture(scope="session", autouse=True)
def _clear_model_cache(request) -> None:
    """Start each test session with an empty model cache.

    The cache is keyed on (stage, prompt, model, prompt_version) — deliberately, so
    that in production an identical prompt to an identical model reuses its answer.
    But the offline provider's output is determined by *code*, and no part of that key
    changes when the code does. Editing the stub therefore leaves the old responses
    being served, and tests assert against behavior that no longer exists.

    Learned the hard way: a fix to the injection stub appeared to do nothing for
    several runs because every call was a cache hit from before the change.
    """
    try:
        eng = request.getfixturevalue("engine")
    except Exception:
        return
    with eng.begin() as conn:
        conn.execute(text("TRUNCATE TABLE model_cache"))


@pytest.fixture(scope="session")
def engine(database_url: str) -> Engine:
    """A live engine, or a skip with a usable instruction.

    Skipping is deliberate: `pytest` on a laptop with nothing running should report
    "these need a database, start compose" rather than a wall of connection errors
    that buries the unit tests that did run.
    """
    eng = create_engine(database_url, pool_pre_ping=True)
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(
            f"No database at {database_url.rsplit('@', 1)[-1]} ({type(exc).__name__}). "
            "Start one with `docker compose up -d db`."
        )
    return eng
