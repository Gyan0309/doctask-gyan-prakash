"""Phase 1: the walking skeleton, end to end.

One document → cited facts → composed sections → human gate → commit, plus the first
real proof of I2: a second run that changes nothing must leave every section
byte-identical.

Runs with no API key. The offline provider supplies deterministic extraction; every
assertion below is about what the *system* did with it.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.engine import Engine

from ledger import service

pytestmark = pytest.mark.integration


CONTRACT = """ACME Consulting Master Services Agreement

The hourly rate is $195 per hour for all professional services.

Payment terms are net 30 days from the date of invoice.

The liability cap is $2,000,000 in aggregate for all claims.

Governing law is the State of Delaware.
"""


@pytest.fixture
def corpus_dir(tmp_path):
    (tmp_path / "acme-msa.md").write_text(CONTRACT, encoding="utf-8")
    return tmp_path


@pytest.fixture(autouse=True)
def _fresh_checkpointer(engine: Engine):
    """`engine` is requested so these tests skip cleanly with no database."""
    service.reset_checkpointer()
    yield
    service.reset_checkpointer()


def _unique(name: str) -> str:
    return f"{name}-{uuid.uuid4().hex[:8]}"


class TestWalkingSkeleton:
    def test_a_run_produces_cited_sections_and_reaches_a_terminal_state(
        self, corpus_dir
    ) -> None:
        result = service.start_run(
            corpus_name=_unique("acme"),
            document_paths=[str(corpus_dir / "acme-msa.md")],
        )

        assert result["counts"]["documents"] == 1
        assert result["counts"]["facts"] > 0, "no facts survived citation resolution"

        # A clean corpus has nothing for a human to decide, so the gate is skipped
        # rather than shown empty. Stopping a reviewer to approve nothing teaches them
        # to click through without reading, which destroys the gate's value.
        assert result["status"] == "completed"
        assert result["awaiting_review"] is False

        deliverable = service.get_deliverable(result["run_id"])
        assert deliverable["sections"], "a run with facts must produce sections"

        # Every section carries a hash. Without it, nothing downstream can prove
        # anything about what did or did not change.
        for section in deliverable["sections"]:
            assert len(section["content_hash"]) == 64

    def test_the_first_run_derives_everything_because_there_is_no_previous_run(
        self, corpus_dir
    ) -> None:
        """The degenerate case of the incremental path — not a separate mode."""
        result = service.start_run(
            corpus_name=_unique("acme"),
            document_paths=[str(corpus_dir / "acme-msa.md")],
        )
        deliverable = service.get_deliverable(result["run_id"])

        assert deliverable["carried_forward"] == 0
        assert deliverable["rederived"] == len(deliverable["sections"])

    def test_stage_metrics_are_recorded_for_the_run(self, corpus_dir) -> None:
        """Behavior 10 — cost and time are measured per stage, not estimated later."""
        result = service.start_run(
            corpus_name=_unique("acme"),
            document_paths=[str(corpus_dir / "acme-msa.md")],
        )
        run = service.get_run(result["run_id"])

        assert any(s["stage"] == "extract" for s in run["stages"])
        extract_stage = next(s for s in run["stages"] if s["stage"] == "extract")
        # Hits or misses, but the stage must account for its calls. Asserting misses
        # specifically would be wrong: the model cache is content-addressed and
        # persists across runs, so an identical prompt from an earlier test legitimately
        # arrives as a hit. That is the cache working, not the metering failing.
        assert extract_stage["cache_hits"] + extract_stage["cache_misses"] >= 1


class TestIncrementalityProof:
    """I2: re-running with no new source must not touch a single section."""

    def test_a_rerun_with_identical_input_carries_every_section_forward(
        self, corpus_dir
    ) -> None:
        corpus = _unique("acme")
        path = str(corpus_dir / "acme-msa.md")

        first = service.start_run(corpus_name=corpus, document_paths=[path])
        first_sections = service.get_deliverable(first["run_id"])["sections"]
        assert first_sections, "precondition: the first run produced sections"

        second = service.start_run(corpus_name=corpus, document_paths=[path])
        second_deliverable = service.get_deliverable(second["run_id"])

        # Not "similar" — identical, by hash, section for section.
        before = {s["section_key"]: s["content_hash"] for s in first_sections}
        after = {
            s["section_key"]: s["content_hash"] for s in second_deliverable["sections"]
        }
        assert after == before, "unchanged input must produce byte-identical sections"

        # And they must have been *carried forward*, not regenerated and coincidentally
        # matching. Regeneration that happens to agree still paid the cost, which is
        # precisely the thing movement 3 claims not to do.
        assert second_deliverable["rederived"] == 0
        assert second_deliverable["carried_forward"] == len(second_deliverable["sections"])

    def test_reingesting_the_same_file_does_not_duplicate_the_document(
        self, corpus_dir
    ) -> None:
        """Identity is content, not filename — re-dropping a file is a no-op."""
        corpus = _unique("acme")
        path = str(corpus_dir / "acme-msa.md")

        service.start_run(corpus_name=corpus, document_paths=[path])
        second = service.start_run(corpus_name=corpus, document_paths=[path])

        assert second["counts"]["documents"] == 1


class TestUnreadableDocumentsDegradeGracefully:
    def test_an_unsupported_file_is_reported_and_the_run_continues(
        self, corpus_dir, tmp_path
    ) -> None:
        """One bad document must not take down the corpus — and must not vanish
        silently either."""
        bad = tmp_path / "scan.tiff"
        bad.write_bytes(b"not a document")

        result = service.start_run(
            corpus_name=_unique("acme"),
            document_paths=[str(corpus_dir / "acme-msa.md"), str(bad)],
        )

        assert result["counts"]["documents"] == 1, "the good document still ingested"
        assert result["counts"]["findings"] >= 1, "the skipped file must be reported"

        # And because there is now something to decide, the gate engages.
        assert result["awaiting_review"] is True
        assert result["pending_findings"], "the gate must present what it wants decided"


class TestHumanGate:
    """Floor 3: a real gate, driven entirely through the service layer."""

    @pytest.fixture
    def parked_run(self, corpus_dir, tmp_path):
        (tmp_path / "broken.tiff").write_bytes(b"not a document")
        result = service.start_run(
            corpus_name=_unique("acme"),
            document_paths=[
                str(corpus_dir / "acme-msa.md"),
                str(tmp_path / "broken.tiff"),
            ],
        )
        assert result["awaiting_review"] is True
        return result

    def test_a_run_parks_at_the_gate_and_waits(self, parked_run) -> None:
        """The run must genuinely stop — not finish and offer a review afterwards."""
        assert parked_run["status"] == "awaiting_review"

        stored = service.get_run(parked_run["run_id"])
        assert stored["status"] != "completed"

    def test_approving_resumes_the_run_to_completion(self, parked_run) -> None:
        decisions = {str(f["index"]): "approved" for f in parked_run["pending_findings"]}
        resumed = service.resume_run(run_id=parked_run["run_id"], decisions=decisions)

        assert resumed["status"] == "completed"
        assert service.get_run(parked_run["run_id"])["status"] == "completed"

    def test_rejected_findings_are_recorded_as_rejected_never_deleted(
        self, parked_run
    ) -> None:
        """A gate whose rejections vanish cannot answer 'why is this not in the
        register?' six weeks later. Rejection is a recorded verdict, not a delete."""
        decisions = {str(f["index"]): "rejected" for f in parked_run["pending_findings"]}
        service.resume_run(run_id=parked_run["run_id"], decisions=decisions)

        recorded = service.get_decisions(parked_run["run_id"])
        assert recorded, "the decision must be in the audit trail"
        assert all(d["verdict"] == "rejected" for d in recorded)
        # The finding itself still exists, carrying the verdict.
        assert all(d["explanation"] for d in recorded)
