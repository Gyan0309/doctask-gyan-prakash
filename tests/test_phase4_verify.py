"""Phase 4: Stage C verification, and the commit block it triggers.

The valuable tests here are the ones that make verification *fail*. A gate that has
never been observed to close is not known to be a gate — and this one is the mechanism
behind I5, the promise that a success message means what it says.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.engine import Engine

import services.service as service
from database.db import session_scope
from domain.verify import verify_run
from models import Chunk, Claim, ClaimCitation, Fact, SectionVersion
from services.graph import route_after_verify

pytestmark = pytest.mark.integration

CONTRACT = """Northwind Analytics Master Services Agreement

The hourly rate is $195 per hour for all professional services.

Payment terms are net 30 days from the date of invoice.

Governing law is the State of Delaware.
"""


@pytest.fixture
def corpus_dir(tmp_path):
    (tmp_path / "msa.md").write_text(CONTRACT, encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _fresh(engine: Engine):
    service.reset_checkpointer()
    yield
    service.reset_checkpointer()


def _unique(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:8]}"


def _completed_run(corpus_dir):
    result = service.start_run(
        corpus_name=_unique("verify"),
        document_paths=[str(corpus_dir / "msa.md")],
    )
    if result["awaiting_review"]:
        decisions = {str(f["index"]): "approved" for f in result["pending_findings"]}
        result = service.resume_run(run_id=result["run_id"], decisions=decisions)
    return result


class TestRouting:
    def test_a_passing_verification_reaches_the_human_gate(self) -> None:
        assert route_after_verify({"verification_passed": True}) == "gate"

    def test_a_failing_verification_never_reaches_the_gate(self) -> None:
        """Presenting a reviewer with findings drawn from a register we know is
        unsound invites an approval that is worse than no approval at all."""
        assert route_after_verify({"verification_passed": False}) == "blocked"

    def test_an_absent_verdict_is_treated_as_passing_only_where_unset(self) -> None:
        assert route_after_verify({}) == "gate"


class TestClaimsAndCitations:
    def test_a_run_produces_claims_with_citations(self, corpus_dir) -> None:
        """I1's enforcement table must actually be populated. It was empty for three
        phases while everything above it looked correct."""
        result = _completed_run(corpus_dir)

        with session_scope() as session:
            versions = (
                session.query(SectionVersion)
                .filter(SectionVersion.run_id == uuid.UUID(result["run_id"]))
                .all()
            )
            assert versions

            claims = (
                session.query(Claim)
                .filter(Claim.section_version_id.in_([v.id for v in versions]))
                .all()
            )
            assert claims, "every section must assert something"

            citations = (
                session.query(ClaimCitation)
                .filter(ClaimCitation.claim_id.in_([c.id for c in claims]))
                .count()
            )
            assert citations >= len(claims), "every claim needs at least one citation"

    def test_verification_passes_on_an_untampered_run(self, corpus_dir) -> None:
        result = _completed_run(corpus_dir)

        with session_scope() as session:
            report = verify_run(session, uuid.UUID(result["run_id"]))

        assert report.passed, [f.detail for f in report.failures]
        assert report.checked > 0, "a verification that checked nothing proves nothing"
        assert report.citations_checked > 0


class TestVerificationCatchesTampering:
    """Each test breaks the register in a different way and asserts it is caught."""

    def test_a_deleted_supporting_fact_is_caught(self, corpus_dir) -> None:
        """The headline case: evidence removed after the claim was made.

        Note what actually happens. `claim_citation.fact_id` is ON DELETE CASCADE, so
        deleting the fact takes the citation with it — the database will not hold a
        citation to nothing. The claim is therefore left *uncited* rather than
        dangling, and that is what verification must catch. (The dangling branch stays
        as defence in depth for a database whose constraints were altered.)

        Either way the outcome is the one that matters: the register keeps asserting a
        value, and verification refuses to let it be committed.
        """
        result = _completed_run(corpus_dir)
        run_id = uuid.UUID(result["run_id"])

        with session_scope() as session:
            claim = (
                session.query(Claim)
                .join(SectionVersion, SectionVersion.id == Claim.section_version_id)
                .filter(SectionVersion.run_id == run_id, Claim.status == "supported")
                .first()
            )
            assert claim is not None, "precondition: the run made a supported claim"

            fact_ids = [
                c.fact_id
                for c in session.query(ClaimCitation)
                .filter(ClaimCitation.claim_id == claim.id)
                .all()
            ]
            assert fact_ids, "precondition: that claim cited something"

            session.query(Fact).filter(Fact.id.in_(fact_ids)).delete(
                synchronize_session=False
            )

        with session_scope() as session:
            report = verify_run(session, run_id)

        assert not report.passed, "a claim whose evidence was deleted must not verify"
        assert any(
            f.reason in ("uncited_claim", "dangling_citation") for f in report.failures
        )

    def test_a_source_edited_after_the_claim_is_caught(self, corpus_dir) -> None:
        """A citation that still *looks* fine but points at text that is no longer
        there. This is the failure a document-name-only citation cannot detect."""
        result = _completed_run(corpus_dir)
        run_id = uuid.UUID(result["run_id"])

        with session_scope() as session:
            # Scoped to *this* run's citations. Querying facts globally mutated a
            # chunk belonging to an unrelated run, so the run under test verified
            # cleanly and the test passed for the wrong reason.
            fact = (
                session.query(Fact)
                .join(ClaimCitation, ClaimCitation.fact_id == Fact.id)
                .join(Claim, Claim.id == ClaimCitation.claim_id)
                .join(SectionVersion, SectionVersion.id == Claim.section_version_id)
                .filter(SectionVersion.run_id == run_id, Fact.chunk_id.isnot(None))
                .first()
            )
            assert fact is not None, "precondition: this run cited a fact with a chunk"

            chunk = session.get(Chunk, fact.chunk_id)
            # Overwrite the passage entirely, so whatever the fact's value was, it can
            # no longer be found where the claim says it came from.
            chunk.text = "This passage was replaced after the claim was made."

        with session_scope() as session:
            report = verify_run(session, run_id)

        assert not report.passed
        assert any(f.reason == "citation_does_not_resolve" for f in report.failures)

    def test_a_supported_claim_stripped_of_citations_is_caught(self, corpus_dir) -> None:
        result = _completed_run(corpus_dir)
        run_id = uuid.UUID(result["run_id"])

        with session_scope() as session:
            claim = (
                session.query(Claim)
                .join(SectionVersion, SectionVersion.id == Claim.section_version_id)
                .filter(SectionVersion.run_id == run_id, Claim.status == "supported")
                .first()
            )
            if claim is None:
                pytest.skip("no supported claim in this run")
            session.query(ClaimCitation).filter(
                ClaimCitation.claim_id == claim.id
            ).delete(synchronize_session=False)

        with session_scope() as session:
            report = verify_run(session, run_id)

        assert not report.passed
        assert any(f.reason == "uncited_claim" for f in report.failures)


class TestAFailedVerificationBlocksTheRun:
    """The end-to-end proof of I5: a run that cannot verify itself does not succeed.

    Unit-testing `verify_run` shows the check can detect tampering. This shows the
    detection actually stops the run — a distinction that matters, because a verifier
    whose verdict is ignored downstream is decoration.
    """

    @pytest.fixture
    def failing_verification(self, monkeypatch):
        from domain.verify import VerificationFailure, VerificationReport

        def _always_fails(session, run_id):
            report = VerificationReport(checked=3, citations_checked=5)
            report.failures.append(
                VerificationFailure(
                    claim_id=uuid.uuid4(),
                    section_key="Northwind::hourly_rate",
                    reason="citation_does_not_resolve",
                    detail="forced failure for test",
                )
            )
            return report

        monkeypatch.setattr("services.graph.verify_run", _always_fails)

    def test_the_run_reports_blocked_never_completed(
        self, corpus_dir, failing_verification
    ) -> None:
        result = service.start_run(
            corpus_name=_unique("blocked"),
            document_paths=[str(corpus_dir / "msa.md")],
        )

        assert result["status"] == "blocked_by_verification"
        assert result["status"] != "completed"

    def test_the_human_gate_is_never_reached(self, corpus_dir, failing_verification) -> None:
        """Showing a reviewer findings from a register known to be unsound invites an
        approval that is worse than no approval."""
        result = service.start_run(
            corpus_name=_unique("blocked"),
            document_paths=[str(corpus_dir / "msa.md")],
        )
        assert result["awaiting_review"] is False

    def test_the_run_is_recorded_as_failed(self, corpus_dir, failing_verification) -> None:
        result = service.start_run(
            corpus_name=_unique("blocked"),
            document_paths=[str(corpus_dir / "msa.md")],
        )
        assert service.get_run(result["run_id"])["status"] == "failed"

    def test_a_blocked_run_cannot_become_the_basis_of_the_next_one(
        self, corpus_dir, monkeypatch
    ) -> None:
        """A blocked run commits nothing, so a later run must rebuild rather than
        carry forward from an unverified predecessor."""
        from domain.verify import VerificationFailure, VerificationReport

        corpus = _unique("blocked")
        path = str(corpus_dir / "msa.md")

        def _always_fails(session, run_id):
            report = VerificationReport(checked=1, citations_checked=1)
            report.failures.append(
                VerificationFailure(
                    claim_id=uuid.uuid4(),
                    section_key="x",
                    reason="citation_does_not_resolve",
                    detail="forced",
                )
            )
            return report

        monkeypatch.setattr("services.graph.verify_run", _always_fails)
        service.start_run(corpus_name=corpus, document_paths=[path])

        monkeypatch.undo()
        service.reset_checkpointer()
        second = service.start_run(corpus_name=corpus, document_paths=[path])
        deliverable = service.get_deliverable(second["run_id"])

        assert deliverable["carried_forward"] == 0, (
            "nothing may be carried forward from a run that failed verification"
        )


class TestFormattingDoesNotBreakVerification:
    def test_presentation_differences_are_tolerated(self) -> None:
        """The register stores "$195 per hour" where the chunk says "$195.00". Failing
        on that would make the check fire constantly and get switched off, which is
        worse than not having it."""
        from domain.verify import _value_present

        assert _value_present("$195", "The rate is $195.00 per hour.")
        assert _value_present("$195 per hour", "rate: $195.00/hr")
        assert _value_present("$2,500,000", "cap of $2500000 in aggregate")
        assert _value_present("State of Delaware", "governed by the state of delaware")

    def test_a_genuinely_different_value_is_still_caught(self) -> None:
        """Tolerance must not become blindness."""
        from domain.verify import _value_present

        assert not _value_present("$195", "The rate is $240 per hour.")
        assert not _value_present("State of Delaware", "governed by the laws of Texas")
