"""Adjudication: deciding whether a detected discrepancy is a real contradiction.

The comparators in `conflicts.py` are exhaustive and literal. They will happily report
that a specialist rate differs from a standard rate, or that a document restates a term
in different words. Deciding which of those matter is genuine judgement, and that is
what the model is for.

What the model is explicitly *not* for: finding the discrepancies. That is arithmetic,
and arithmetic done by a language model is arithmetic you cannot test.

The model can only ever downgrade or explain — never invent. It receives a fixed list
of candidates and returns a verdict on each. There is no path by which adjudication
adds a conflict that the deterministic layer did not find, which means the recall of
this system is a property of code, not of a prompt.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from ledger.domain.conflicts import ConflictCandidate
from ledger.logging_config import get_logger, log
from ledger.metering import MeteredClient

logger = get_logger(__name__)

PROMPT_VERSION = "adjudicate-v1"

SEVERITIES = ["low", "medium", "high"]

ADJUDICATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "verdicts": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "is_real_conflict": {"type": "BOOLEAN"},
                    "severity": {"type": "STRING", "enum": SEVERITIES},
                    "explanation": {"type": "STRING"},
                },
                "required": ["index", "is_real_conflict", "severity", "explanation"],
            },
        }
    },
    "required": ["verdicts"],
}


@dataclass(frozen=True)
class Adjudication:
    candidate: ConflictCandidate
    is_real_conflict: bool
    severity: str
    explanation: str


def build_prompt(candidates: list[ConflictCandidate]) -> str:
    """One call for all candidates, not one per candidate.

    Batching is not only cheaper; it is more accurate. Several candidates often stem
    from one underlying situation — an amendment that changed a rate produces both a
    superseded value and a stale invoice — and a model shown them together can say so,
    where a model shown them one at a time cannot.
    """
    lines = []
    for index, candidate in enumerate(candidates):
        lines.append(
            f"[{index}] kind={candidate.kind} vendor={candidate.subject!r} "
            f"term={candidate.predicate!r}\n"
            f"     finding: {candidate.detail}\n"
            f"     source A: {candidate.a.document_kind}, "
            f"effective {candidate.a.effective_date or 'undated'}, "
            f"value {candidate.a.value_raw!r}\n"
            f"     source B: {candidate.b.document_kind}, "
            f"effective {candidate.b.effective_date or 'undated'}, "
            f"value {candidate.b.value_raw!r}"
        )

    return f"""You review discrepancies found in a set of vendor contract documents by
a deterministic checker. The arithmetic has already been done and is correct — do not
recompute it. Your job is judgement about meaning.

For each numbered candidate, decide:

- `is_real_conflict`: true if these two statements genuinely contradict each other.
  False if they are consistent once context is understood — for example a rate that
  applies to a different engagement, a value restated in different words, or a
  historical value that was correct when it was written.
- `severity`: low, medium or high. Judge by consequence. A billing discrepancy that
  costs money is high. An ambiguity in wording that changes nothing is low.
- `explanation`: one or two sentences a contract manager would find useful. State what
  the discrepancy is and what should happen about it. Do not restate the arithmetic.

Return a verdict for every candidate, using the index given.

Candidates:
{chr(10).join(lines)}"""


def adjudicate(
    candidates: list[ConflictCandidate], client: MeteredClient
) -> list[Adjudication]:
    """Judge each candidate. Never called with an empty list — see the graph."""
    if not candidates:
        return []

    completion = client.generate(
        build_prompt(candidates), stage="adjudicate", schema=ADJUDICATION_SCHEMA
    )

    try:
        payload = json.loads(completion.text)
    except json.JSONDecodeError:
        return _fallback(candidates, "adjudicator returned unparseable output")

    verdicts = {}
    for item in payload.get("verdicts", []):
        try:
            verdicts[int(item["index"])] = item
        except (KeyError, TypeError, ValueError):
            continue

    results: list[Adjudication] = []
    for index, candidate in enumerate(candidates):
        verdict = verdicts.get(index)

        if verdict is None:
            # A candidate the model did not answer for is reported, not dropped. The
            # deterministic layer already established the discrepancy is real; silence
            # from the adjudicator is not evidence against it, and discarding it here
            # would let a model make a finding disappear by omission.
            log(
                logger,
                logging.WARNING,
                "adjudicator returned no verdict for a candidate; reporting it unjudged",
                kind=candidate.kind,
                term=candidate.predicate,
            )
            results.append(
                Adjudication(
                    candidate=candidate,
                    is_real_conflict=True,
                    severity="medium",
                    explanation=(
                        f"{candidate.detail} (No adjudication was returned for this "
                        f"candidate, so it is reported as found.)"
                    ),
                )
            )
            continue

        severity = verdict.get("severity")
        results.append(
            Adjudication(
                candidate=candidate,
                is_real_conflict=bool(verdict.get("is_real_conflict", True)),
                severity=severity if severity in SEVERITIES else "medium",
                explanation=(verdict.get("explanation") or candidate.detail).strip(),
            )
        )

    return results


def _fallback(candidates: list[ConflictCandidate], reason: str) -> list[Adjudication]:
    """Degrade to the deterministic finding when the model cannot be understood.

    The comparators already proved the discrepancy exists and `detail` already states
    it in numbers. Losing the model costs the explanation and the severity judgement —
    it must not cost the finding.
    """
    log(
        logger,
        logging.WARNING,
        "adjudication unavailable, reporting raw candidates",
        reason=reason,
    )
    return [
        Adjudication(
            candidate=candidate,
            is_real_conflict=True,
            severity="medium",
            explanation=f"{candidate.detail} ({reason}.)",
        )
        for candidate in candidates
    ]
