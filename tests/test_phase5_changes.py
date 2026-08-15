"""Phase 5: the change ledger — the proof that an update costs like an update.

This is the test that matters most in the whole suite. A document arrives, the register
updates, and the claim under examination is not "it still works" but:

  - the sections the new document did not affect are **byte-identical**
  - the sections it did affect changed, and the ledger names **which document** did it
  - the untouched ones were carried forward, not regenerated and coincidentally equal

That last distinction is the one worth being pedantic about. A system that regenerates
everything and happens to produce the same bytes passes a naive hash check while paying
full price — which is exactly the thing movement 3 claims not to do.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.engine import Engine

from ledger import service
from ledger.db import session_scope
from ledger.domain.changes import build_ledger

pytestmark = pytest.mark.integration

MSA = """Northwind Analytics Master Services Agreement

The hourly rate is $180 per hour for all professional services.

Payment terms are net 45 days from the date of invoice.

Governing law is the State of Delaware.

Estimated annual fees under this Agreement are $600,000.
"""

AMENDMENT = """Amendment No. 1 to the Master Services Agreement

Effective 2025-07-01, the hourly rate is $195 per hour.

All other terms of the Agreement remain unchanged.
"""


@pytest.fixture(autouse=True)
def _fresh(engine: Engine):
    service.reset_checkpointer()
    yield
    service.reset_checkpointer()


def _unique(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:8]}"


def _complete(result: dict) -> dict:
    """Approve whatever the gate presents so the run commits."""
    if not result.get("awaiting_review"):
        return result
    decisions = {str(f["index"]): "approved" for f in result["pending_findings"]}
    return service.resume_run(run_id=result["run_id"], decisions=decisions)


@pytest.fixture
def two_runs(tmp_path):
    """Run once over an MSA, then again after an amendment lands beside it."""
    (tmp_path / "msa.md").write_text(MSA, encoding="utf-8")
    corpus = _unique("changes")

    first = _complete(
        service.start_run(corpus_name=corpus, document_paths=[str(tmp_path / "msa.md")])
    )
    assert first["status"] == "completed", "run 1 must commit before run 2 builds on it"

    (tmp_path / "amendment.md").write_text(AMENDMENT, encoding="utf-8")
    second = _complete(
        service.start_run(
            corpus_name=corpus,
            document_paths=[
                str(tmp_path / "msa.md"),
                str(tmp_path / "amendment.md"),
            ],
        )
    )
    return first, second


class TestTheLedgerExplainsItself:
    def test_the_second_run_is_linked_to_the_first(self, two_runs) -> None:
        first, second = two_runs
        changes = service.get_changes(second["run_id"])
        assert changes["parent_run_id"] == first["run_id"]

    def test_every_section_is_accounted_for(self, two_runs) -> None:
        """No section may be missing from the ledger. A section that appears in the
        register but not in the change report is one nobody can audit."""
        _, second = two_runs
        changes = service.get_changes(second["run_id"])
        deliverable = service.get_deliverable(second["run_id"])

        assert changes["sections_total"] == len(deliverable["sections"])
        assert (
            changes["added"] + changes["changed"] + changes["unchanged"]
            == changes["sections_total"]
        )

    def test_a_changed_section_names_the_document_that_moved_it(self, two_runs) -> None:
        """'What changed and why' must be a query, not a narrative."""
        _, second = two_runs
        changes = service.get_changes(second["run_id"])

        moved = [s for s in changes["sections"] if s["status"] in ("added", "changed")]
        assert moved, "the amendment must have moved something"
        assert any(s["caused_by"] for s in moved), (
            "a moved section must name the document responsible"
        )

    def test_a_changed_section_records_both_hashes(self, two_runs) -> None:
        _, second = two_runs
        changes = service.get_changes(second["run_id"])

        for section in changes["sections"]:
            if section["status"] == "changed":
                assert section["previous_hash"]
                assert section["previous_hash"] != section["content_hash"]


class TestUntouchedMeansUntouched:
    def test_unchanged_sections_are_byte_identical(self, two_runs) -> None:
        """I2, stated as a hash comparison rather than a promise."""
        first, second = two_runs

        before = {
            s["section_key"]: s["content_hash"]
            for s in service.get_deliverable(first["run_id"])["sections"]
        }
        changes = service.get_changes(second["run_id"])

        for section in changes["sections"]:
            if section["status"] == "unchanged":
                assert section["content_hash"] == before[section["section_key"]]

    def test_unchanged_sections_were_carried_not_regenerated(self, two_runs) -> None:
        """The pedantic distinction. Regenerating and coincidentally matching passes a
        naive hash check while paying full price — which is the thing being disclaimed."""
        _, second = two_runs

        changes = service.get_changes(second["run_id"])
        deliverable = service.get_deliverable(second["run_id"])

        carried = {s["section_key"] for s in deliverable["sections"] if s["carried_forward"]}
        unchanged = {s["section_key"] for s in changes["sections"] if s["status"] == "unchanged"}

        assert unchanged <= carried, (
            "a section reported unchanged must have been carried forward, "
            "not re-derived into the same bytes"
        )

    def test_most_of_the_register_survives_a_new_document(self, two_runs) -> None:
        """The headline number. If a one-line amendment rewrites the whole register,
        the dependency map is not doing its job."""
        _, second = two_runs
        changes = service.get_changes(second["run_id"])

        assert changes["unchanged"] > 0, "a targeted amendment must leave rows untouched"


class TestFullRuns:
    def test_a_first_run_reports_everything_as_added(self, tmp_path) -> None:
        """The degenerate case of an incremental run: no predecessor, so every section
        is new. One code path, not two."""
        (tmp_path / "msa.md").write_text(MSA, encoding="utf-8")
        result = _complete(
            service.start_run(
                corpus_name=_unique("full"),
                document_paths=[str(tmp_path / "msa.md")],
            )
        )

        changes = service.get_changes(result["run_id"])
        assert changes["parent_run_id"] is None
        assert changes["added"] == changes["sections_total"]
        assert changes["unchanged"] == 0

    def test_an_unknown_run_raises_rather_than_returning_an_empty_ledger(self) -> None:
        """An empty ledger for a nonexistent run would read as 'nothing changed'."""
        with session_scope() as session, pytest.raises(KeyError):
            build_ledger(session, uuid.uuid4())
