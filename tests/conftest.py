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

# Bound the wait when nothing is listening. Without this the probe can hang instead of
# failing, and a suite that hangs is worse than one that fails — the developer has no
# idea whether it is working or stuck, and CI eventually kills it with no useful output.
CONNECT_ARGS = {"connect_timeout": 3}


def _test_database_url(configured: str) -> str:
    """The suite's own database, beside whatever the developer is using.

    Tests used to run against the same database the app does, so a `pytest` run buried
    the run list under hundreds of `floor2-…` and `changes-…` entries — and that list
    is the first thing anyone opening the UI sees. Worse, the suite creates runs that
    are *deliberately* broken (killed mid-flight, blocked at verification), so the
    development database ended up full of states no real corpus would produce.

    Deriving the name rather than hardcoding it keeps one knob: point DATABASE_URL at
    any Postgres and its `_test` sibling is what the suite uses.
    """
    if configured.endswith("_test"):
        return configured
    base, _, name = configured.rpartition("/")
    name, sep, query = name.partition("?")
    return f"{base}/{name}_test{sep}{query}"


def _ensure_database(url: str) -> None:
    """Create the test database and bring it to head, if it is not there already.

    `CREATE DATABASE` cannot run inside a transaction, hence the autocommit isolation
    level, and it is issued against the `postgres` maintenance database because you
    cannot create a database from inside itself.
    """
    import subprocess
    import sys

    base, _, name = url.rpartition("/")
    name = name.partition("?")[0]

    admin = create_engine(f"{base}/postgres", connect_args=CONNECT_ARGS)
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        exists = conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": name}
        ).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()

    # Migrations rather than `create_all`: the suite must exercise the same schema path
    # a deployment takes, or a broken migration passes every test and fails on deploy.
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        check=True,
    )


@pytest.fixture(scope="session", autouse=True)
def _offline_by_default() -> None:
    """Force the offline provider for the whole suite.

    Autouse and session-scoped so no test can accidentally reach a real API — if a
    developer has a key in their environment, the suite must still be the suite, not
    a slower suite that quietly spends money.
    """
    os.environ["LLM_PROVIDER"] = "fake"

    # Redirect the whole session onto the `_test` sibling before anything reads config,
    # so no test can reach the development database even by importing the app directly.
    configured = os.environ.get("DATABASE_URL", DEFAULT_TEST_DB)
    os.environ["DATABASE_URL"] = _test_database_url(configured)

    # Pinned to empty, because a developer's `.env` must not decide test outcomes.
    #
    # Setting ORGANISATION_NAME for a live run turned twenty-six tests red at once: the
    # synthetic fixtures name no organisation, so every one of them escalated for not
    # being "ours", runs stopped before composing, and half the suite failed on
    # preconditions about sections that were never built. The failures were real — that is
    # exactly what the check does — but they were about the developer's environment, not
    # about the code under test. A test that changes answer when `.env` changes is not
    # testing what it says it is.
    #
    # The tests that care about this check set it themselves.
    os.environ["ORGANISATION_NAME"] = ""

    from database.config import get_settings

    get_settings.cache_clear()


@pytest.fixture(scope="session")
def database_url() -> str:
    url = _test_database_url(os.environ.get("DATABASE_URL", DEFAULT_TEST_DB))
    try:
        _ensure_database(url)
    except Exception:
        # No Postgres, or no permission to create. Integration tests skip individually
        # with a usable message; the unit tests neither need nor care.
        pass
    return url


@pytest.fixture(scope="session", autouse=True)
def _clear_model_cache(database_url: str) -> None:
    """Start each test session with an empty model cache.

    The cache is keyed on (stage, prompt, model, prompt_version) — deliberately, so
    that in production an identical prompt to an identical model reuses its answer.
    But the offline provider's output is determined by *code*, and no part of that key
    changes when the code does. Editing the stub therefore leaves the old responses
    being served, and tests assert against behavior that no longer exists.

    Learned the hard way: a fix to the injection stub appeared to do nothing for
    several runs because every call was a cache hit from before the change.

    This connects directly rather than requesting the `engine` fixture. `engine` calls
    pytest.skip() when no database is present, and Skipped derives from BaseException —
    so an autouse session fixture that triggers it skips the **entire suite**, unit
    tests included. That silently turned `pytest -m "not integration"` into a no-op
    that reported success.
    """
    try:
        eng = create_engine(database_url, connect_args=CONNECT_ARGS)
        with eng.begin() as conn:
            conn.execute(text("TRUNCATE TABLE model_cache"))
        eng.dispose()
    except Exception:
        # No database, or no schema yet. Integration tests will skip individually with
        # a usable message; the unit tests neither need nor care.
        return


@pytest.fixture(scope="session")
def engine(database_url: str) -> Engine:
    """A live engine, or a skip with a usable instruction.

    Skipping is deliberate: `pytest` on a laptop with nothing running should report
    "these need a database, start compose" rather than a wall of connection errors
    that buries the unit tests that did run.
    """
    eng = create_engine(database_url, pool_pre_ping=True, connect_args=CONNECT_ARGS)
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(
            f"No database at {database_url.rsplit('@', 1)[-1]} ({type(exc).__name__}). "
            "Start one with `docker compose up -d db`."
        )
    return eng
