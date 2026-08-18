"""Conflict detection: finding the places where the documents disagree.

The work is split deliberately, and the split is the design:

  **Candidate generation is deterministic.** Three general comparators over the
  normalized fact graph. No model. Exhaustive, cheap, and testable with no key.

  **Adjudication is the model's job.** Is this candidate a real contradiction, or two
  different things wearing similar names? How severe? What should a human read?

The model only ever sees *candidates* — never the cross product of every fact against
every other. That is what makes it affordable, and it is what keeps this from being a
pile of hardcoded special cases wearing intelligence as a costume: the comparators are
three general mechanisms over normalized values, and the genuinely ambiguous judgement
is the only part that goes to a model.

The comparators are general in a specific sense worth stating: none of them knows what
an hourly rate or a liability cap *is*. They operate on predicates, normalized
magnitudes, effective dates and document precedence. Adding a new term to the register
does not require touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from domain.direction import Comparison, compare
from domain.normalize import normalize
from domain.reconcile import (
    FactView,
    canonical_scope,
    canonical_subject,
    governing_value_at,
)

# What each comparator is for, in one line, because these strings surface in the UI.
# Which governing term an observational predicate is evidence *about*.
#
# Without this, the central conflict in the domain is structurally undetectable: an
# invoice's rate extracts as `invoice_rate` and an agreement's as `hourly_rate`, so
# grouping by predicate alone means the two never meet and the comparator is silent no
# matter how badly the invoice is wrong. Found by running the full corpus and noticing
# the one conflict it was built around was missing.
#
# Data, not code — relating a new observational term to a governing one is an entry
# here, not a new comparator.
OBSERVES: dict[str, str] = {
    "invoice_rate": "hourly_rate",
}


def governed_predicate(predicate: str) -> str:
    """The term a predicate speaks to, which is itself for governing predicates."""
    return OBSERVES.get(predicate, predicate)


CONFLICT_KINDS = {
    "same_predicate_different_value": (
        "Two governing documents state different values for the same term, with "
        "nothing to break the tie"
    ),
    "temporal_precedence": (
        "A document acted on a value that had already been superseded when it was issued"
    ),
    "arithmetic_mismatch": (
        "A stated total does not equal the components the same document states"
    ),
}


@dataclass(frozen=True)
class ConflictCandidate:
    """A discrepancy found deterministically, before any model has an opinion.

    `detail` is computed, not generated — it states the arithmetic. If adjudication is
    skipped or the model is unavailable, this is still a usable finding, which is what
    lets the system degrade rather than fall over.
    """

    kind: str
    subject: str
    predicate: str
    a: FactView
    b: FactView
    detail: str
    # Which way the difference runs, stated by the comparator that measured it. The
    # adjudicator is shown this rather than left to infer it from the values, and its
    # explanation is checked against it — a live run produced a finding that reversed
    # the sign of a $60 under-billing and recommended requesting a refund.
    comparison: Comparison | None = None

    def pair_key(self) -> tuple:
        """Stable identity for deduplication, order-independent."""
        return (self.kind, *sorted([str(self.a.fact_id), str(self.b.fact_id)]))


def _magnitude(fact: FactView) -> Decimal | None:
    normalized = normalize(fact.predicate, fact.value_raw)
    return normalized.number if normalized else None


def _unit(fact: FactView) -> str | None:
    normalized = normalize(fact.predicate, fact.value_raw)
    return normalized.unit if normalized else None


def _comparable(a: FactView, b: FactView) -> bool:
    """Two facts may only be compared when their units agree.

    Without this, 30 days and 30 months compare equal and 30 days versus 24 months
    reads as a conflict. Unit mismatch means the comparison is meaningless, not that
    the values disagree.
    """
    na, nb = normalize(a.predicate, a.value_raw), normalize(b.predicate, b.value_raw)
    return na is not None and nb is not None and na.unit == nb.unit


# ---------------------------------------------------------------------------
# Comparator 1 — two governing documents, same term, different values
# ---------------------------------------------------------------------------


def find_same_predicate_different_value(facts: list[FactView]) -> list[ConflictCandidate]:
    """Genuine ambiguity: two documents that both govern, and disagree.

    Deliberately narrow. A later amendment stating a different value is *supersession*,
    which reconciliation already resolves — reporting that as a conflict would flag
    every amendment chain in the corpus as a problem, which is worse than useless.

    So this fires only when the ordering rules cannot separate them: same effective
    date (or both undated) and equal precedence. That is the case where the system
    genuinely cannot tell which value is in force, and a human must.
    """
    candidates: list[ConflictCandidate] = []
    grouped: dict[tuple[str, str, str | None], list[FactView]] = {}

    for fact in facts:
        if fact.governs:
            # Folded, exactly as reconcile folds them. Grouping here on the raw
            # strings meant `ARDENT FACILITIES MANAGEMENT LLC` and `Ardent Facilities
            # Management LLC` were two vendors to the comparators, and `out-of-hours`
            # and `Out of Hours` two scopes — so a rate card still fought itself across
            # documents even after reconciliation had been taught not to.
            grouped.setdefault(
                (
                    canonical_subject(fact.subject),
                    fact.predicate,
                    canonical_scope(fact.scope),
                ),
                [],
            ).append(fact)

    for (_folded, predicate, _scope), group in grouped.items():
        # Group on the folded name, report the written one — otherwise a finding reads
        # "ardent facilities management llc", which is a folding key leaking into text
        # a human is meant to act on.
        subject = group[0].subject
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if a.effective_date != b.effective_date or a.precedence != b.precedence:
                    continue  # resolvable by ordering; not a conflict
                if not _comparable(a, b):
                    continue

                ma, mb = _magnitude(a), _magnitude(b)
                if ma is None or mb is None or ma == mb:
                    continue

                candidates.append(
                    ConflictCandidate(
                        kind="same_predicate_different_value",
                        subject=subject,
                        predicate=predicate,
                        a=a,
                        b=b,
                        detail=(
                            f"{a.document_kind} states {a.value_raw!r} and "
                            f"{b.document_kind} states {b.value_raw!r} for {predicate}. "
                            f"Both carry the same effective date and precedence, so "
                            f"neither supersedes the other."
                        ),
                        comparison=compare(ma, mb, _unit(a)),
                    )
                )

    return candidates


# ---------------------------------------------------------------------------
# Comparator 2 — a document acted on a value that was already superseded
# ---------------------------------------------------------------------------


def find_temporal_precedence_violations(facts: list[FactView]) -> list[ConflictCandidate]:
    """The invoice that bills a rate an amendment already replaced.

    The important subtlety is *which* value to compare against: the one that governed
    on the observation's own date, not the one that governs today. Comparing against
    the current value would flag every historical invoice the moment a new amendment
    lands — a false-positive machine that trains people to ignore the output.
    """
    candidates: list[ConflictCandidate] = []
    by_term: dict[tuple[str, str, str | None], list[FactView]] = {}

    # Grouped by the term a fact speaks *about* (so an invoice's `invoice_rate` lands
    # alongside the agreement's `hourly_rate`) **and by scope** (so a partner-grade
    # observation is never measured against a litigation-support governing rate).
    #
    # Scope was honoured by `reconcile` and by comparator 1, but not here — the same
    # "folds in one place, not its peer" fault the vendor fix had. A live run reported a
    # managing-partner rate "$720 HIGHER" than agreed; that was $905 (managing partner,
    # observed) minus $185 (litigation support, governing) — two different grades
    # subtracted — and the direction fix printed the fabricated difference as fact. A
    # rate card fought itself through this comparator even after the others learned not
    # to. Facts sharing a governed term but not a scope are different rows, and a scope
    # that does not match on both sides means the two simply cannot be compared.
    for fact in facts:
        by_term.setdefault(
            (
                canonical_subject(fact.subject),
                governed_predicate(fact.predicate),
                canonical_scope(fact.scope),
            ),
            [],
        ).append(fact)

    # One finding per (document, term, scope). A document routinely states the same
    # value twice — an invoice gives its rate in the line-item table and again in prose,
    # so it is extracted as both `invoice_rate` and `hourly_rate` — and reporting both
    # produces two identical findings for one problem. Duplicate findings are not a
    # cosmetic issue: a reviewer working a queue has to read and dismiss each one, and
    # learns that the queue wastes their time. Scope is part of the marker because it is
    # part of the group: one invoice overbilling both the partner and associate lines is
    # two problems, and a marker without scope would swallow the second as a duplicate.
    reported: set[tuple] = set()

    for (folded_subject, predicate, scope), group in by_term.items():
        # Folded for grouping and for the dedupe marker, written for the report.
        subject = group[0].subject
        observations = [f for f in group if not f.governs and f.effective_date]

        for observation in observations:
            marker = (observation.document_id, folded_subject, predicate, scope)
            if marker in reported:
                continue
            governing = governing_value_at(group, observation.effective_date)
            if governing is None or not _comparable(observation, governing):
                continue

            observed, expected = _magnitude(observation), _magnitude(governing)
            if observed is None or expected is None or observed == expected:
                continue

            # Marked only once a candidate is actually raised, so a fact that was
            # skipped for being incomparable does not suppress a later one that is.
            reported.add(marker)

            since = (
                governing.effective_date.isoformat()
                if governing.effective_date
                else "an undated document"
            )
            candidates.append(
                ConflictCandidate(
                    kind="temporal_precedence",
                    subject=subject,
                    predicate=predicate,
                    a=observation,
                    b=governing,
                    detail=(
                        f"On {observation.effective_date.isoformat()} the "
                        f"{observation.document_kind} used {observation.value_raw!r}, "
                        f"but the value governing since {since} "
                        f"({governing.document_kind}) was {governing.value_raw!r}. "
                        f"Difference: {observed - expected}."
                    ),
                    comparison=compare(observed, expected, _unit(observation)),
                )
            )

    return candidates


# ---------------------------------------------------------------------------
# Comparator 3 — a stated total that its own components contradict
# ---------------------------------------------------------------------------

# Which predicate is the product of which others. Data, not code: a new arithmetic
# relationship is an entry here, not a new function.
ARITHMETIC_RELATIONS = [
    ("invoice_amount", ("invoice_rate", "invoice_hours")),
]


def _unambiguous(group: list[FactView], predicate: str) -> FactView | None:
    """The one fact stating this predicate on this line, or None if that is not settled.

    Facts accumulate: a document re-extracted across runs states each line several times
    over — the live corpus holds every Brightmoor line three times — and the old
    `{f.predicate: f for f in group}` silently kept whichever arrived last. That is
    harmless while the copies agree and a guess when they do not.

    So copies that agree on magnitude are one statement, and copies that disagree are no
    statement at all. Multiplying a rate one extraction read as $190 and another as $200
    would report a mismatch that is an artifact of the disagreement, not a fact about the
    bill — and a fabricated finding is the failure this comparator exists to avoid.
    """
    matches = [f for f in group if f.predicate == predicate]
    if not matches:
        return None
    magnitudes = {_magnitude(f) for f in matches}
    if len(magnitudes) != 1 or None in magnitudes:
        return None
    return matches[0]


def find_arithmetic_mismatches(facts: list[FactView]) -> list[ConflictCandidate]:
    """Internal inconsistency: a billed line whose own numbers do not multiply out.

    Scoped to a single document on purpose. Comparing a total in one document against
    components in another would be comparing different engagements and generating
    nonsense.

    Scoped to a single *line* for the same reason, one level down. A real invoice is
    many lines at several rates, and `rate x hours = amount` is an identity of a line,
    never of a document: the Talus invoice bills thirteen time-and-materials lines at
    three rates plus a subscription overage that is not time-based at all. Grouping by
    document alone kept one fact per predicate and subtracted a single line's product
    from the grand total — `$22,611` minus `$245 x 7.5` — reported as a $20,773.50
    discrepancy in the high band. Four such findings on the live corpus, every one an
    artifact of the grouping, and they crowded out the real arithmetic error.

    A line is identified by its qualifier, which reaches here as `scope`. Unqualified
    facts are the invoice's headline figures — a standard rate, total hours, a total
    folding in non-time charges — and those are checked only when the document states no
    lines at all, because then the triple genuinely is the whole bill. Once lines exist,
    the headline rate multiplied by the headline hours is not an identity anyone asserts,
    and treating it as one is how the fabrication started.
    """
    candidates: list[ConflictCandidate] = []
    by_line: dict[tuple[object, str | None], list[FactView]] = {}

    for fact in facts:
        by_line.setdefault((fact.document_id, canonical_scope(fact.scope)), []).append(fact)

    # Documents that itemise. Their unqualified figures summarise the lines rather than
    # standing alone, so no product is claimed over them.
    itemised = {document_id for (document_id, scope) in by_line if scope is not None}

    for (document_id, scope), group in by_line.items():
        if scope is None and document_id in itemised:
            continue

        for total_predicate, component_predicates in ARITHMETIC_RELATIONS:
            total = _unambiguous(group, total_predicate)
            components = [_unambiguous(group, p) for p in component_predicates]
            if total is None or any(c is None for c in components):
                continue

            magnitudes = [_magnitude(c) for c in components]
            total_magnitude = _magnitude(total)
            if total_magnitude is None or any(m is None for m in magnitudes):
                continue

            product = Decimal(1)
            for magnitude in magnitudes:
                product *= magnitude

            if product == total_magnitude:
                continue

            candidates.append(
                ConflictCandidate(
                    kind="arithmetic_mismatch",
                    subject=total.subject,
                    predicate=total_predicate,
                    a=total,
                    b=components[0],
                    detail=(
                        # Named, because a mismatch a reviewer cannot locate on the bill
                        # is a mismatch they cannot act on.
                        f"{'On line ' + repr(scope) + ', ' if scope else ''}"
                        f"{total_predicate} is stated as {total.value_raw!r}, but "
                        f"{' x '.join(f'{c.predicate}={c.value_raw!r}' for c in components)} "
                        f"gives {product}. Difference: {total_magnitude - product}."
                    ),
                    comparison=compare(total_magnitude, product, _unit(total)),
                )
            )

    return candidates


# ---------------------------------------------------------------------------


COMPARATORS = (
    find_same_predicate_different_value,
    find_temporal_precedence_violations,
    find_arithmetic_mismatches,
)


def detect(facts: list[FactView]) -> list[ConflictCandidate]:
    """Run every comparator and deduplicate.

    Returning an empty list is a first-class, expected outcome — a clean corpus really
    has no conflicts, and the graph uses emptiness to skip adjudication entirely rather
    than paying a model to confirm there is nothing to say.
    """
    seen: set[tuple] = set()
    candidates: list[ConflictCandidate] = []

    for comparator in COMPARATORS:
        for candidate in comparator(facts):
            key = candidate.pair_key()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)

    return candidates
