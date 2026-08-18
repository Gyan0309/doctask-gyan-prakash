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

import services.service as service
from database.db import session_scope
from domain.changes import build_ledger

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


class TestReuseIsInvalidatedByLogicToo:
    """The counterweight to incrementality, and it was missing.

    Invalidation keyed on fact identity alone: same facts, reuse the bytes. But a section's
    content is *derived* from its facts, so changing the derivation changes the answer while
    the fact set stays identical.

    Found live rather than in review. Teaching the reconciler that a renewal notice restates
    terms rather than governing them was correct, tested and deployed — and the register went
    on reporting `net forty-five (45) days` sourced from the renewal notice, because every
    section it applied to carried forward untouched. Structurally the same trap as a prompt
    fix that looks inert because facts are reused, one layer up.
    """

    def test_a_derivation_change_re_derives_everything(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "msa.md").write_text(MSA, encoding="utf-8")
        corpus = _unique("composer")
        paths = [str(tmp_path / "msa.md")]

        first = _complete(service.start_run(corpus_name=corpus, document_paths=paths))
        assert first["status"] == "completed"

        # A second run over an unchanged corpus: everything carries forward.
        unchanged = _complete(service.start_run(corpus_name=corpus, document_paths=paths))
        assert unchanged["plan"]["sections_carried_forward"] > 0
        assert unchanged["plan"]["sections_rederived"] == 0

        # Now the derivation logic changes. Nothing about the documents or the facts moves.
        monkeypatch.setattr("domain.compose.COMPOSER_VERSION", "compose-test-next")

        after = _complete(service.start_run(corpus_name=corpus, document_paths=paths))

        assert after["plan"]["sections_carried_forward"] == 0, (
            "a derivation change must invalidate reuse — otherwise a fix to how the "
            "register is derived silently changes nothing"
        )
        assert after["plan"]["sections_rederived"] == after["plan"]["sections_total"]

    def test_carried_forward_content_keeps_the_version_that_made_it(
        self, tmp_path
    ) -> None:
        """Stamping the current version onto copied bytes would claim the current logic
        produced them, and the next derivation change would find nothing to invalidate."""
        from models import SectionVersion

        (tmp_path / "msa.md").write_text(MSA, encoding="utf-8")
        corpus = _unique("composer")
        paths = [str(tmp_path / "msa.md")]

        _complete(service.start_run(corpus_name=corpus, document_paths=paths))
        second = _complete(service.start_run(corpus_name=corpus, document_paths=paths))

        with session_scope() as session:
            carried = (
                session.query(SectionVersion)
                .filter(
                    SectionVersion.run_id == uuid.UUID(second["run_id"]),
                    SectionVersion.carried_forward.is_(True),
                )
                .all()
            )
            assert carried, "precondition: something was carried forward"
            for version in carried:
                assert version.composer_version is not None

    def test_a_section_written_before_this_existed_is_re_derived(self, tmp_path) -> None:
        """Rows predating the column hold NULL, which compares unequal to any version, so
        they are re-derived exactly once. Backfilling a value would have asserted that old
        content came from current logic — the thing this column exists to deny.

        Simulated by clearing the column, which is precisely the state the migration
        leaves an existing deployment in.
        """
        from sqlalchemy import update

        from models import SectionVersion

        (tmp_path / "msa.md").write_text(MSA, encoding="utf-8")
        corpus = _unique("composer")
        paths = [str(tmp_path / "msa.md")]

        first = _complete(service.start_run(corpus_name=corpus, document_paths=paths))

        with session_scope() as session:
            session.execute(
                update(SectionVersion)
                .where(SectionVersion.run_id == uuid.UUID(first["run_id"]))
                .values(composer_version=None)
            )

        after = _complete(service.start_run(corpus_name=corpus, document_paths=paths))

        assert after["plan"]["sections_carried_forward"] == 0
        assert after["plan"]["sections_rederived"] == after["plan"]["sections_total"]


class TestAnUnrelatedDocumentTouchesNothing:
    """A new vendor's agreement must not re-derive another vendor's rows.

    Measured on the live corpus and it did: adding one agreement for a brand-new vendor
    re-derived 69 sections, of which **62 produced byte-identical content**. Model cost was
    zero — the facts were reused — so the claim "an update costs like an update" survives on
    calls. But 62 sections of pointless work is 62 chances for the carry-forward guarantee
    to be quietly wrong, and "it re-derived and happened to match" is exactly the
    regenerate-then-compare behaviour this module exists to avoid.
    """

    OTHER_VENDOR = """Halstead Weighing Systems Services Agreement

Engineer attendance is charged at ninety-five dollars ($95) per hour.

Payment terms are net thirty (30) days from the date of invoice.

Governing law is the State of Illinois.
"""

    def test_a_new_vendor_does_not_disturb_an_existing_one(self, tmp_path) -> None:
        (tmp_path / "msa.md").write_text(MSA, encoding="utf-8")
        corpus = _unique("unrelated")

        first = _complete(
            service.start_run(corpus_name=corpus, document_paths=[str(tmp_path / "msa.md")])
        )
        assert first["status"] == "completed"
        before = len(service.get_deliverable(first["run_id"])["sections"])

        (tmp_path / "other.md").write_text(self.OTHER_VENDOR, encoding="utf-8")
        second = _complete(
            service.start_run(
                corpus_name=corpus,
                document_paths=[str(tmp_path / "msa.md"), str(tmp_path / "other.md")],
            )
        )

        plan = second["plan"]
        added = plan["sections_total"] - before

        assert plan["sections_rederived"] == added, (
            f"only the {added} new sections should be re-derived, "
            f"but {plan['sections_rederived']} were"
        )
        assert plan["sections_carried_forward"] == before

    def test_nothing_re_derived_reproduces_bytes_it_already_had(self, tmp_path) -> None:
        """The stronger form. A section that re-derives to the same hash was re-derived
        for no reason, and that is the signal worth catching — it is invisible in the
        output and only shows up in the plan."""
        from models import SectionVersion

        (tmp_path / "msa.md").write_text(MSA, encoding="utf-8")
        corpus = _unique("unrelated")

        first = _complete(
            service.start_run(corpus_name=corpus, document_paths=[str(tmp_path / "msa.md")])
        )
        (tmp_path / "other.md").write_text(self.OTHER_VENDOR, encoding="utf-8")
        second = _complete(
            service.start_run(
                corpus_name=corpus,
                document_paths=[str(tmp_path / "msa.md"), str(tmp_path / "other.md")],
            )
        )

        with session_scope() as session:
            before = {
                sv.section_key: sv.content_hash
                for sv in session.query(SectionVersion)
                .filter(SectionVersion.run_id == uuid.UUID(first["run_id"]))
                .all()
            }
            pointless = [
                sv.section_key
                for sv in session.query(SectionVersion)
                .filter(
                    SectionVersion.run_id == uuid.UUID(second["run_id"]),
                    SectionVersion.carried_forward.is_(False),
                )
                .all()
                if before.get(sv.section_key) == sv.content_hash
            ]

        assert pointless == [], (
            f"{len(pointless)} sections were re-derived and produced the bytes they "
            f"already had: {pointless[:5]}"
        )
