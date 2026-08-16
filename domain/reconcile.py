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

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from uuid import UUID

from domain.classify import KIND_PRECEDENCE

# Kinds whose terms bind only their own engagement, not the agreement as a whole.
SCOPED_KINDS = {"sow"}

# Kinds that record what happened rather than what was agreed. They never govern, and
# that is what makes them able to contradict.
OBSERVATIONAL_KINDS = {"invoice"}


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
    """The outcome for one (subject, predicate) — or one scope within it."""

    subject: str
    predicate: str
    scope: str | None
    governing: FactView | None
    superseded: list[FactView] = field(default_factory=list)
    observations: list[FactView] = field(default_factory=list)

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
        base = f"{self.subject}::{self.predicate}"
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


def reconcile(facts: list[FactView]) -> list[Resolution]:
    """Group facts and resolve each group to a single governing value.

    Grouping includes scope, so a SOW rate and an agreement rate are resolved
    separately rather than competing. Both appear in the register, distinguishable.
    """
    grouped: dict[tuple[str, str, str | None], list[FactView]] = {}
    for fact in facts:
        scope = fact.scope if fact.document_kind in SCOPED_KINDS else None
        grouped.setdefault((fact.subject, fact.predicate, scope), []).append(fact)

    resolutions: list[Resolution] = []
    for (subject, predicate, scope), group in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "")
    ):
        governing_candidates = [f for f in group if f.governs]
        observations = [f for f in group if not f.governs]

        ordered = sorted(governing_candidates, key=_sort_key)
        governing = ordered[-1] if ordered else None
        superseded = ordered[:-1] if len(ordered) > 1 else []

        resolutions.append(
            Resolution(
                subject=subject,
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
