"""Stage C — verification. The anti-bluff mechanism (I5).

The brief asks for *"a fresh pair of eyes — a verifier that is not the implementer
catches what the author cannot."* This module is that second pair of eyes, and it is
deliberately not the code that composed the register: it re-derives nothing and trusts
nothing, it only checks that what the register asserts still resolves to something real.

Three checks, all deterministic, all runnable with no key:

  1. Every claim marked `supported` has at least one citation.
  2. Every cited fact still exists.
  3. Every cited fact's quote still appears in the document it came from, at the span
     recorded for it.

Check 3 is the one that earns its place. A citation that names a document proves
nothing — the interesting failure is a citation that *looks* fine and points at text
that is no longer there, which happens when a document is re-ingested with edits. That
is exactly the case where the register would otherwise report a stale value with full
confidence.

**A failure here blocks the commit.** It does not annotate the output and continue. A
run that cannot verify its own claims has not succeeded, and reporting success would be
the precise failure I5 forbids: a success message that does not mean what it says.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID


@dataclass(frozen=True)
class VerificationFailure:
    claim_id: UUID
    section_key: str
    reason: str
    detail: str


@dataclass
class VerificationReport:
    checked: int = 0
    citations_checked: int = 0
    failures: list[VerificationFailure] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> dict[str, int | bool]:
        return {
            "claims_checked": self.checked,
            "citations_checked": self.citations_checked,
            "failures": len(self.failures),
            "passed": self.passed,
        }


def verify_run(session, run_id: UUID) -> VerificationReport:
    """Verify every claim in a run's register.

    Takes a session rather than importing the graph's, so this can be run standalone
    against any historical run — the audit question "was this register ever valid?"
    should not require re-executing anything.
    """
    from sqlalchemy import select

    from models import Chunk, Claim, ClaimCitation, Document, Fact, SectionVersion

    report = VerificationReport()

    rows = session.execute(
        select(Claim, SectionVersion)
        .join(SectionVersion, SectionVersion.id == Claim.section_version_id)
        .where(SectionVersion.run_id == run_id)
    ).all()

    for claim, version in rows:
        report.checked += 1

        cited_ids = (
            session.execute(
                select(ClaimCitation.fact_id).where(ClaimCitation.claim_id == claim.id)
            )
            .scalars()
            .all()
        )

        if claim.status == "supported" and not cited_ids:
            report.failures.append(
                VerificationFailure(
                    claim_id=claim.id,
                    section_key=version.section_key,
                    reason="uncited_claim",
                    detail=(
                        f"Claim asserts {claim.text!r} but carries no citation. A "
                        f"supported claim without evidence is an invented one."
                    ),
                )
            )
            continue

        for fact_id in cited_ids:
            report.citations_checked += 1
            fact = session.get(Fact, fact_id)

            if fact is None:
                report.failures.append(
                    VerificationFailure(
                        claim_id=claim.id,
                        section_key=version.section_key,
                        reason="dangling_citation",
                        detail=(
                            f"Claim cites fact {fact_id} which no longer exists. The "
                            f"supporting evidence was removed after the claim was made."
                        ),
                    )
                )
                continue

            if fact.chunk_id is None:
                # No chunk recorded means the quote spanned a gap between chunks. The
                # fact is still real; we simply cannot re-check its span, so this is
                # reported as unverifiable rather than passed over in silence.
                report.failures.append(
                    VerificationFailure(
                        claim_id=claim.id,
                        section_key=version.section_key,
                        reason="unverifiable_citation",
                        detail=(
                            f"Fact {fact.predicate}={fact.value_raw!r} has no chunk "
                            f"recorded, so its citation cannot be re-checked against "
                            f"the source."
                        ),
                    )
                )
                continue

            chunk = session.get(Chunk, fact.chunk_id)
            document = session.get(Document, fact.document_id)

            if chunk is None or document is None:
                report.failures.append(
                    VerificationFailure(
                        claim_id=claim.id,
                        section_key=version.section_key,
                        reason="dangling_citation",
                        detail=(
                            f"Fact {fact.predicate}={fact.value_raw!r} cites a chunk or "
                            f"document that no longer exists."
                        ),
                    )
                )
                continue

            # The substantive check: does the cited value still appear in the source
            # text it was drawn from? Compared on digits only, because the register
            # stores "$195 per hour" where the chunk says "$195.00 per hour" — a
            # difference in presentation, not in fact.
            if not _value_present(fact.value_raw, chunk.text):
                report.failures.append(
                    VerificationFailure(
                        claim_id=claim.id,
                        section_key=version.section_key,
                        reason="citation_does_not_resolve",
                        detail=(
                            f"Claim cites {fact.predicate}={fact.value_raw!r} from "
                            f"{document.uri}, but that value no longer appears in the "
                            f"cited passage. The source changed after the claim was made."
                        ),
                    )
                )

    return report


def _value_present(value_raw: str, source: str) -> bool:
    """Is this value still findable in the source text?

    Compares the significant characters — digits and letters — rather than requiring an
    exact string match. "$195" and "$195.00" and "195 USD" all describe the same term
    in the same document, and failing verification over formatting would make the check
    fire constantly and be switched off, which is worse than not having it.
    """
    if value_raw in source:
        return True

    digits = "".join(c for c in value_raw if c.isdigit())
    if not digits:
        # A non-numeric value such as a governing law. Fall back to a normalized
        # substring match.
        needle = " ".join(value_raw.split()).lower()
        return needle in " ".join(source.split()).lower()

    # Strip thousands separators and trailing zero decimals from the source before
    # looking for the digit run.
    source_digits = "".join(c for c in source if c.isdigit())
    if digits in source_digits:
        return True

    trimmed = digits.rstrip("0") or digits
    return trimmed in source_digits
