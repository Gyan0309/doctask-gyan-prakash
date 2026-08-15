"""Dependency-aware composition of the deliverable.

This is the load-bearing module. Everything the system claims about incremental
updates is decided here.

The rule: a section is re-derived **only** if the set of facts it depends on changed.
Every other section is carried forward by copying its previous content and hash
verbatim — not regenerated and compared, *copied*. That distinction is the whole
point. Regenerate-then-compare still pays the model cost, and still risks a
nondeterministic generator producing a different-but-equivalent string that shows up
as spurious churn.

Because a full run is just the case where no previous run exists (so every section is
invalid), there is one code path here, not two. The incremental logic is therefore
exercised by every run ever made, including the first, and cannot rot unobserved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from ledger.hashing import content_hash, stable_json
from ledger.models import SectionDependency, SectionVersion


@dataclass(frozen=True)
class SectionPlan:
    """What a section should contain this run, and what it derives from."""

    section_key: str
    kind: str
    payload: dict
    fact_ids: frozenset[UUID]

    def render(self) -> str:
        """Deterministic rendering. Dict ordering must never affect the hash, or
        'unchanged' becomes a coin flip across runs."""
        return stable_json(self.payload)


@dataclass
class RecompositionPlan:
    rederive: list[SectionPlan] = field(default_factory=list)
    carry_forward: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.rederive) + len(self.carry_forward)

    def summary(self) -> dict[str, int]:
        return {
            "sections_total": self.total,
            "sections_rederived": len(self.rederive),
            "sections_carried_forward": len(self.carry_forward),
        }


def load_recorded_dependencies(session: Session, section_keys: list[str]) -> dict[str, set[UUID]]:
    if not section_keys:
        return {}
    rows = session.execute(
        select(SectionDependency.section_key, SectionDependency.fact_id).where(
            SectionDependency.section_key.in_(section_keys)
        )
    ).all()
    recorded: dict[str, set[UUID]] = {}
    for key, fact_id in rows:
        recorded.setdefault(key, set()).add(fact_id)
    return recorded


def plan_recomposition(
    session: Session,
    desired: list[SectionPlan],
    *,
    prev_run_id: UUID | None,
) -> RecompositionPlan:
    """Decide which sections must be re-derived and which carry forward.

    A section is re-derived when any of these hold:
      - there is no previous run (the degenerate full-run case)
      - it has no version in the previous run (it is new)
      - its dependency set differs from what was recorded

    Note what is deliberately *not* a trigger: the content being different. Content is
    an output of composition, so consulting it would require composing first, which is
    the cost we are avoiding.
    """
    plan = RecompositionPlan()

    if prev_run_id is None:
        plan.rederive = list(desired)
        return plan

    previous_keys = set(
        session.execute(
            select(SectionVersion.section_key).where(SectionVersion.run_id == prev_run_id)
        )
        .scalars()
        .all()
    )
    recorded = load_recorded_dependencies(session, [s.section_key for s in desired])

    for section in desired:
        if section.section_key not in previous_keys:
            plan.rederive.append(section)
            continue
        if recorded.get(section.section_key, set()) != set(section.fact_ids):
            plan.rederive.append(section)
            continue
        plan.carry_forward.append(section.section_key)

    return plan


def apply_plan(
    session: Session,
    plan: RecompositionPlan,
    *,
    run_id: UUID,
    prev_run_id: UUID | None,
) -> list[SectionVersion]:
    """Write this run's section versions.

    Carried-forward sections copy the previous content and hash byte for byte. This is
    what makes "untouched sections are byte-identical" true by construction rather
    than by luck — there is no code path in which an untouched section is regenerated.
    """
    written: list[SectionVersion] = []

    previous: dict[str, SectionVersion] = {}
    if prev_run_id is not None:
        previous = {
            sv.section_key: sv
            for sv in session.execute(
                select(SectionVersion).where(SectionVersion.run_id == prev_run_id)
            ).scalars()
        }

    for section in plan.rederive:
        content = section.render()
        version = SectionVersion(
            section_key=section.section_key,
            run_id=run_id,
            content=content,
            content_hash=content_hash(content),
            prev_version_id=previous.get(section.section_key).id
            if section.section_key in previous
            else None,
            carried_forward=False,
        )
        session.add(version)
        written.append(version)

        # Rewrite the dependency map for this section. Delete-then-insert because a
        # fact that no longer contributes must stop invalidating it — leaving a stale
        # edge would cause phantom re-derivations forever, which looks like the
        # incremental logic simply not working.
        session.query(SectionDependency).filter(
            SectionDependency.section_key == section.section_key
        ).delete(synchronize_session=False)
        for fact_id in section.fact_ids:
            session.add(
                SectionDependency(section_key=section.section_key, fact_id=fact_id)
            )

    for key in plan.carry_forward:
        prior = previous[key]
        version = SectionVersion(
            section_key=key,
            run_id=run_id,
            content=prior.content,
            content_hash=prior.content_hash,  # copied, never recomputed
            prev_version_id=prior.id,
            carried_forward=True,
        )
        session.add(version)
        written.append(version)

    session.flush()
    return written
