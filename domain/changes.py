"""The change ledger: what changed, when, and because of which source.

Movement 3 asks the system to prove that an update cost like an update. The proof has
two halves, and this module is the second:

  *Nothing else changed* — a hash comparison, already guaranteed by construction in
  `compose.py`, where a carried-forward section copies its predecessor's content and
  hash rather than regenerating them.

  *This changed, and here is why* — which is a query, not a narrative. That is what
  this module answers.

The "why" is recoverable because each section records the facts it derives from, and
each fact records the document it came from. So the causal chain from "a new PDF
landed" to "this row changed" is data the system already holds, not a story assembled
afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select

from models import (
    Claim,
    ClaimCitation,
    Document,
    Fact,
    Run,
    SectionVersion,
)
from utils.paths import basename


@dataclass
class SectionChange:
    section_key: str
    status: str  # added | changed | unchanged
    content_hash: str
    previous_hash: str | None
    caused_by: list[str] = field(default_factory=list)  # document filenames


@dataclass
class ChangeLedger:
    run_id: str
    parent_run_id: str | None
    changes: list[SectionChange] = field(default_factory=list)

    @property
    def added(self) -> list[SectionChange]:
        return [c for c in self.changes if c.status == "added"]

    @property
    def changed(self) -> list[SectionChange]:
        return [c for c in self.changes if c.status == "changed"]

    @property
    def unchanged(self) -> list[SectionChange]:
        return [c for c in self.changes if c.status == "unchanged"]

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "sections_total": len(self.changes),
            "added": len(self.added),
            "changed": len(self.changed),
            "unchanged": len(self.unchanged),
            # The headline number. On a well-behaved incremental run this is most of
            # the register, and it is the figure the incrementality claim rests on.
            "untouched_fraction": (
                round(len(self.unchanged) / len(self.changes), 3) if self.changes else 0.0
            ),
        }


def _citing_documents(session, section_version_id: UUID) -> set[str]:
    """Filenames of the documents behind a section version's claims."""
    rows = session.execute(
        select(Document.uri)
        .join(Fact, Fact.document_id == Document.id)
        .join(ClaimCitation, ClaimCitation.fact_id == Fact.id)
        .join(Claim, Claim.id == ClaimCitation.claim_id)
        .where(Claim.section_version_id == section_version_id)
    ).scalars().all()
    return {basename(uri) for uri in rows}


def build_ledger(session, run_id: UUID) -> ChangeLedger:
    """Diff a run's register against its predecessor.

    Compares by content hash rather than by the `carried_forward` flag. The flag says
    what the *planner* intended; the hash says what actually landed. They should always
    agree, and comparing the thing that matters means a bug in the planner shows up
    here instead of being papered over by its own bookkeeping.
    """
    run = session.get(Run, run_id)
    if run is None:
        raise KeyError(run_id)

    ledger = ChangeLedger(
        run_id=str(run_id),
        parent_run_id=str(run.parent_run_id) if run.parent_run_id else None,
    )

    current = (
        session.execute(
            select(SectionVersion)
            .where(SectionVersion.run_id == run_id)
            .order_by(SectionVersion.section_key)
        )
        .scalars()
        .all()
    )

    previous: dict[str, SectionVersion] = {}
    if run.parent_run_id:
        previous = {
            sv.section_key: sv
            for sv in session.execute(
                select(SectionVersion).where(SectionVersion.run_id == run.parent_run_id)
            ).scalars()
        }

    for version in current:
        prior = previous.get(version.section_key)

        if prior is None:
            status = "added"
        elif prior.content_hash != version.content_hash:
            status = "changed"
        else:
            status = "unchanged"

        caused_by: list[str] = []
        if status in ("added", "changed"):
            # Only the documents that are new to this section. Listing every source
            # would name the MSA on every row of the register and say nothing — the
            # useful answer is which arriving document moved this particular value.
            now = _citing_documents(session, version.id)
            before = _citing_documents(session, prior.id) if prior else set()
            caused_by = sorted(now - before) or sorted(now)

        ledger.changes.append(
            SectionChange(
                section_key=version.section_key,
                status=status,
                content_hash=version.content_hash,
                previous_hash=prior.content_hash if prior else None,
                caused_by=caused_by,
            )
        )

    return ledger
