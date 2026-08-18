"""Reconciliation: deciding which value currently governs.

Given several documents stating the same term with different values, exactly one of
them is in force. Working that out is the difference between a register that reports
"the rate is $180, or maybe $195, or $210" and one that reports the rate, names the
instrument that set it, and can show its work.

Three rules, in order:

  1. **Scope.** A term set by a Statement of Work governs that SOW, not the agreement.
     A specialist rate of $210 for one project does not supersede the standard rate,
     and treating it as a competing value would manufacture a contradiction that does
     not exist.
  2. **Effective date.** Later beats earlier. This is what makes an amendment chain
     resolvable at all.
  3. **Precedence.** When dates tie, document kind decides — an amendment beats the
     MSA it amends.

What this deliberately does *not* do is silently discard the losers. A superseded fact
is linked, not deleted, because the superseded value is exactly what a late invoice
will contradict. Deleting it would destroy the evidence for the most valuable finding
the system produces (I4: never silently resolve a contradiction).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from uuid import UUID

from domain.classify import KIND_PRECEDENCE

# Kinds whose terms bind only their own engagement, not the agreement as a whole.
# Retained for callers that still ask; grouping no longer consults it, because scope is
# decided by the producer of the fact rather than by the kind of document it came from.
SCOPED_KINDS = {"sow"}


def canonical_subject(subject: str) -> str:
    """Fold a vendor name to one identity.

    Subjects were grouped by exact string, so `ARDENT FACILITIES MANAGEMENT LLC` on a
    rate-adjustment notice and `Ardent Facilities Management LLC` on the agreement it
    amends became two separate vendors living in parallel. The register then asserted
    two different current rates for one company, and the notice could never supersede
    the agreement it existed to change — supersession only compares within a subject.

    Case and internal whitespace only. Deliberately not fuzzy matching: merging
    `Northwind Analytics LLC` with `Northwind Analytics Ltd` would be a guess about
    corporate identity, and a wrong merge silently combines two companies' obligations
    — worse than the split it was fixing.
    """
    return " ".join(subject.split()).casefold()


def canonical_scope(scope: str | None) -> str | None:
    """Fold a scope label so one rate-card row is one scope.

    Case-folding alone was not enough. Documents label the same column differently —
    `out-of-hours` in the agreement, `Out of Hours` on the rate notice, `All trades —
    standard` with an em dash — and each spelling became its own scope, so the same
    column across two documents could never supersede itself and instead sat there as
    two live values.

    Punctuation is flattened to spaces rather than deleted, so `out-of-hours` and
    `out of hours` meet while `nonstandard` and `non standard` stay apart. Digits are
    kept: `Grade 3` and `Grade 4` are different rows and must not merge.
    """
    if not scope:
        return None
    folded = re.sub(r"[^0-9a-z]+", " ", scope.casefold()).strip()
    return " ".join(folded.split()) or None

# Kinds that record what happened rather than what was agreed. They never govern, and
# that is what makes them able to contradict.
#
# `unknown` and `renewal_notice` are here for a different reason than `invoice`, and both
# were live faults. A document nobody could type still governed, and precedence is only a
# tie-break — `_sort_key` compares the effective date *first* — so a 2026 data-protection
# addendum outranked the 2023 MSA it sat beside on date alone, whatever its rank. Ranking
# them low looked like a safeguard and was not one.
#
# The renewal notice is the sharper case, because it was already understood and the
# defence still did not work. `KIND_PRECEDENCE` puts it *below* the MSA with a comment
# saying a restatement must not become the governing source for the agreement's own terms
# — and it did anyway. Measured: Talus's payment terms resolved to `net forty-five (45)
# days` **from the renewal notice**, dated after Amendment No. 1, with the amendment's
# `net thirty (30) days` filed as superseded. The system read the amendment correctly,
# then let a courtesy letter restating the old figure overrule it, because the letter was
# newer.
#
# A document that restates or records must not be able to define an obligation. It can
# still be evidence that something disagrees, which is what being observational means:
# the notice's stale 24-month renewal term now contradicts the agreement's 12 and is
# reported as a finding, rather than quietly becoming the answer. If nothing else supplies
# the term the resolution reports `unsupported` — the corpus records a value with no
# agreement behind it, which is the honest reading.
OBSERVATIONAL_KINDS = {"invoice", "unknown", "renewal_notice"}


@dataclass(frozen=True)
class FactView:
    """A fact plus the context reconciliation needs. Decoupled from the ORM so this
    logic is testable with plain objects and no database."""

    fact_id: UUID
    predicate: str
    subject: str
    value_raw: str
    value_norm: Decimal | None
    unit: str | None
    effective_date: date | None
    document_id: UUID
    document_kind: str
    scope: str | None = None  # e.g. a SOW reference; None means agreement-wide

    @property
    def precedence(self) -> int:
        return KIND_PRECEDENCE.get(self.document_kind, -1)

    @property
    def governs(self) -> bool:
        """Whether this fact is eligible to set the current value of a term."""
        return self.document_kind not in OBSERVATIONAL_KINDS


@dataclass
class Resolution:
    """The outcome for one (subject, predicate) — or one scope within it.

    `subject` and `subject_key` are deliberately two fields. The first is what a person
    reads; the second is what the system means. Collapsing them was a real bug twice
    over — see `key()`.
    """

    subject: str
    predicate: str
    scope: str | None
    governing: FactView | None
    superseded: list[FactView] = field(default_factory=list)
    observations: list[FactView] = field(default_factory=list)
    # The folded vendor name. Defaulted from `subject` so callers constructing a
    # Resolution by hand still get a coherent identity rather than an empty one.
    subject_key: str | None = None

    def __post_init__(self) -> None:
        if self.subject_key is None:
            self.subject_key = canonical_subject(self.subject)

    @property
    def status(self) -> str:
        if self.governing is None:
            # Observations with nothing governing them: the corpus records a billed
            # rate but contains no agreement setting one. Reported as unsupported
            # rather than promoting the invoice, which would let what was billed
            # define what was agreed.
            return "unsupported"
        return "agreed"

    def key(self) -> str:
        """The section's identity, built from folded parts only.

        This used to be built from `subject` — the governing document's spelling — and
        that caused two separate faults:

          1. **One company appeared as two vendors.** Grouping was already folded, so
             supersession worked, but `Ardent Facilities Management LLC::auto_renew_months`
             and `ARDENT FACILITIES MANAGEMENT LLC::hourly_rate::apprentice` were
             different section keys, sorted apart, and read as two companies in the
             register. Fixing the grouping without fixing the key fixed the arithmetic
             and left the presentation wrong — which is the half a reviewer sees.
          2. **The identity was unstable.** A new document with a different spelling
             becoming governing *renamed* every section it touched, so an update
             registered as a removal plus an insertion rather than an edit — churn in
             the incrementality figures and, on publish, a section deleted and rewritten
             instead of amended.

        Identity must not depend on presentation. It is the same rule that already
        governs headings: anchors derive from the key, so rewording a label cannot make
        a section look edited. The displayed spelling still comes from the governing
        document, in `subject`.
        """
        base = f"{self.subject_key}::{self.predicate}"
        return f"{base}::{self.scope}" if self.scope else base


def _sort_key(fact: FactView) -> tuple:
    """Later date wins; on a tie, higher precedence wins.

    Facts with no effective date sort earliest. That is the conservative choice: an
    undated fact cannot displace a dated one, so a missing date degrades to "does not
    govern" rather than to "governs everything".
    """
    return (
        fact.effective_date or date.min,
        fact.precedence,
    )


def display_names(facts: list[FactView]) -> dict[str, str]:
    """One spelling per vendor, chosen once for the whole register.

    Folding identity was necessary and not sufficient. Display was still decided per
    resolution — "the governing document's spelling wins" — which has no answer when a
    resolution has nothing governing it, and falls back to whichever fact sorted first.
    Measured on a live register: 60 rows read `Ardent Facilities Management LLC` and one
    read `ARDENT FACILITIES MANAGEMENT LLC`, because that one row was `unsupported` and
    inherited the shouted variant from a rate notice.

    The keys were unified, so nothing downstream was wrong. A reviewer scanning the
    register still saw the company twice, which is the thing the fix was for.

    Weighted by how often each spelling appears, so one shouted notice cannot outvote an
    agreement; governing facts count double, because those are the instruments rather than
    the correspondence. Ties prefer a spelling that is not all upper case — a company's
    name in a contract is not shouted, and the SHOUTED variant is reliably the outlier
    that a letterhead or a header row introduced. The final tie-break is alphabetical,
    only so the answer is stable: a display name that varied between runs would make every
    affected section look edited.
    """
    from collections import Counter

    tally: dict[str, Counter] = {}
    for fact in facts:
        key = canonical_subject(fact.subject)
        counts = tally.setdefault(key, Counter())
        counts[fact.subject] += 2 if fact.governs else 1

    def rank(item: tuple[str, int]) -> tuple:
        spelling, weight = item
        shouted = spelling.isupper()
        return (weight, not shouted, spelling)

    return {
        key: max(counts.items(), key=rank)[0] for key, counts in tally.items()
    }


def reconcile(facts: list[FactView]) -> list[Resolution]:
    """Group facts and resolve each group to a single governing value.

    Grouping includes scope, so a SOW rate and an agreement rate are resolved
    separately rather than competing. Both appear in the register, distinguishable.
    """
    names = display_names(facts)

    grouped: dict[tuple[str, str, str | None], list[FactView]] = {}
    for fact in facts:
        # Scope is honoured for every kind, not only SOWs.
        #
        # It used to be dropped unless the document was a SOW, which meant a rate card
        # in an MSA or an engagement letter lost its row labels and collapsed into one
        # contested predicate — five timekeepers competing to be "the" hourly rate. The
        # producer already decides what limits a fact's reach (`_scope_for`); discarding
        # that here on the basis of document kind second-guessed it, and got it wrong
        # for every rate card outside a SOW.
        grouped.setdefault(
            (
                canonical_subject(fact.subject),
                fact.predicate,
                canonical_scope(fact.scope),
            ),
            [],
        ).append(fact)

    resolutions: list[Resolution] = []
    for (folded_subject, predicate, scope), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "")
    ):
        governing_candidates = [f for f in group if f.governs]
        observations = [f for f in group if not f.governs]

        ordered = sorted(governing_candidates, key=_sort_key)
        governing = ordered[-1] if ordered else None
        superseded = ordered[:-1] if len(ordered) > 1 else []

        # Group on the folded name, display the written one — and use the same written one
        # for every row of this vendor, not whichever fact this particular row happened to
        # resolve to. Deciding it per row left a company reading as two on any row with
        # nothing governing it.
        display = names.get(folded_subject) or group[0].subject

        resolutions.append(
            Resolution(
                subject=display,
                subject_key=folded_subject,
                predicate=predicate,
                scope=scope,
                governing=governing,
                superseded=superseded,
                observations=observations,
            )
        )

    return resolutions


def governing_value_at(facts: list[FactView], when: date) -> FactView | None:
    """Which value governed on a given date.

    Needed by conflict detection: to judge whether a September invoice billed the
    right rate, you must know what the rate was *in September*, not what it is now.
    Comparing against the current value would flag every historical invoice the moment
    a new amendment lands, which is a false-positive machine.
    """
    eligible = [
        f
        for f in facts
        if f.governs and f.effective_date is not None and f.effective_date <= when
    ]
    if not eligible:
        return None
    return sorted(eligible, key=_sort_key)[-1]
