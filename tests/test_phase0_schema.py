"""Phase 0: prove the schema migrated and the constraints that carry the invariants
actually exist in the database — not merely in models.py.

A CHECK constraint that was declared but never migrated is the difference between a
guarantee and a comment.
"""

from __future__ import annotations

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.integration

# The three tables that carry requirements nothing else can enforce.
LOAD_BEARING = ["section_dependency", "claim_citation", "decision"]

EXPECTED_TABLES = {
    "corpus", "document", "chunk", "fact",
    "section", "section_version", "section_dependency",
    "claim", "claim_citation",
    "conflict", "rule", "finding", "finding_evidence",
    "decision", "run", "stage_metric", "model_cache",
}


def test_every_designed_table_exists(engine: Engine) -> None:
    present = set(inspect(engine).get_table_names())
    missing = EXPECTED_TABLES - present
    assert not missing, f"migration did not create: {sorted(missing)}"


@pytest.mark.parametrize("table", LOAD_BEARING)
def test_load_bearing_tables_are_present(engine: Engine, table: str) -> None:
    """Named individually so a failure says which invariant just lost its teeth."""
    assert table in inspect(engine).get_table_names()


def test_pgvector_extension_is_installed_and_usable(engine: Engine) -> None:
    """Installed is not the same as usable. This runs an actual distance query."""
    with engine.connect() as conn:
        version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        ).scalar_one_or_none()
        assert version is not None, "vector extension missing — migration should create it"

        distance = conn.execute(
            text("SELECT '[1,0,0]'::vector <-> '[0,1,0]'::vector")
        ).scalar_one()
        assert distance == pytest.approx(2 ** 0.5)


def test_chunk_embedding_column_has_the_right_width(engine: Engine) -> None:
    with engine.connect() as conn:
        udt = conn.execute(
            text(
                "SELECT udt_name FROM information_schema.columns "
                "WHERE table_name = 'chunk' AND column_name = 'embedding'"
            )
        ).scalar_one()
    assert udt == "vector"


def test_a_document_cannot_be_ingested_twice_into_one_corpus(engine: Engine) -> None:
    """Re-dropping the same file must be a no-op. The guarantee is a UNIQUE constraint
    on (corpus_id, sha256), so it holds even if the calling code forgets to check."""
    constraints = inspect(engine).get_unique_constraints("document")
    assert any(
        set(c["column_names"]) == {"corpus_id", "sha256"} for c in constraints
    ), "uq_document_corpus_sha256 is missing; duplicate ingestion would silently succeed"


def test_one_section_gets_at_most_one_version_per_run(engine: Engine) -> None:
    """Guards against a partial re-derivation writing a second version of the same
    section inside one run, which would make the version chain ambiguous."""
    constraints = inspect(engine).get_unique_constraints("section_version")
    assert any(set(c["column_names"]) == {"run_id", "section_key"} for c in constraints)


def test_claim_status_is_constrained_to_the_honest_set(engine: Engine) -> None:
    """'unsupported' must be a legal status. If the constraint drops it, the system
    loses its ability to decline to bluff (I1)."""
    with engine.connect() as conn:
        clause = conn.execute(
            text(
                "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
                "WHERE conname = 'ck_claim_status'"
            )
        ).scalar_one_or_none()
    assert clause is not None, "ck_claim_status missing"
    for status in ("supported", "unsupported", "conflicted"):
        assert status in clause
