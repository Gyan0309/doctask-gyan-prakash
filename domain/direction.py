"""Which way a difference runs, and whether the prose about it agrees.

The comparators compute a difference and then hand the pair to a model to explain in a
sentence a human will read. On a live run the adjudicator was given a partner rate
billed at $780 against a governing $840 — a vendor that *under*-billed by $60 — and
wrote "the billed hourly rate is significantly higher than the contractually agreed
rate. Request a refund or corrected invoice."

Every number in that finding was right. The sentence reversed the sign of a difference
the deterministic layer had already computed, and recommended chasing money nobody was
owed. That is the worst failure shape available to this system: not a miss, but a
confident instruction to take a wrong action, delivered in the one part of a finding a
human actually reads.

So direction is computed once, where the numbers are, and travels with the candidate:

  1. The comparator that did the arithmetic states the direction (`Comparison`).
  2. The prompt is *told* the direction rather than left to infer it.
  3. The returned explanation is checked against it, and an explanation that
     unambiguously contradicts it is withheld rather than shown.

Step 3 is the load-bearing one, because 2 is a request and 1 is already true. It is the
same rule the rest of this system runs on: the thing that checks is not the thing that
wrote.

The checker is a keyword test over English and it is not, and cannot be, complete. It
is deliberately biased towards silence: a claim it cannot read as clearly one direction
or the other passes through untouched. What it catches is the case that actually
occurred — a plain, confident statement pointing the wrong way.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

# How to name the two sides for each comparator, so the sentence says what was compared
# rather than "value A" and "value B". Data, not code: a new comparator is an entry.
_FRAMES: dict[str, tuple[str, str]] = {
    "temporal_precedence": ("the value the document used", "the value that governed"),
    "same_predicate_different_value": (
        "the first document's value",
        "the second document's value",
    ),
    "arithmetic_mismatch": ("the stated total", "its own components multiplied out"),
}

_DEFAULT_FRAME = ("the first value", "the second value")


@dataclass(frozen=True)
class Comparison:
    """The signed relationship between the two magnitudes in a candidate.

    `sign` is +1 when the value under scrutiny is the larger, -1 when it is the
    smaller, and this is stated from the perspective of side A in the candidate — which
    for every comparator is the value being questioned, not the reference.
    """

    sign: int
    delta: Decimal
    unit: str

    def as_payload(self) -> dict:
        """Serializable form. Candidates cross a LangGraph checkpoint as plain JSON,
        and a direction that does not survive that trip is a direction the adjudicator
        never sees."""
        return {"sign": self.sign, "delta": str(self.delta), "unit": self.unit}

    @classmethod
    def from_payload(cls, payload: object) -> Comparison | None:
        """Rebuild from state. Returns None for anything malformed or absent — an
        older checkpoint has no direction recorded, and that must degrade to "not
        checked" rather than to a fabricated sign."""
        if not isinstance(payload, dict):
            return None
        try:
            return cls(
                sign=int(payload["sign"]),
                delta=Decimal(str(payload["delta"])),
                unit=str(payload["unit"]),
            )
        except (KeyError, TypeError, ValueError, ArithmeticError):
            return None


def compare(a: Decimal | None, b: Decimal | None, unit: str | None) -> Comparison | None:
    """Compare two magnitudes that are already known to share a unit."""
    if a is None or b is None or unit is None:
        return None
    if a == b:
        return Comparison(sign=0, delta=Decimal(0), unit=unit)
    return Comparison(sign=1 if a > b else -1, delta=abs(a - b), unit=unit)


def _amount(comparison: Comparison) -> str:
    magnitude = comparison.delta.normalize()
    if comparison.unit == "USD":
        return f"${magnitude:f}"
    return f"{magnitude:f} {comparison.unit}"


def sentence(kind: str, comparison: Comparison | None) -> str:
    """The direction in one plain sentence, computed rather than generated.

    This is what goes into the prompt, and what replaces a contradicting explanation.
    It carries the consequence as well as the arithmetic for money, because "lower" and
    "no money is owed to us" are the same fact and only the second one stops a reviewer
    opening a refund request.
    """
    if comparison is None:
        return ""

    subject, reference = _FRAMES.get(kind, _DEFAULT_FRAME)

    if comparison.sign == 0:
        return f"{subject.capitalize()} and {reference} are equal in magnitude."

    relation = "HIGHER than" if comparison.sign > 0 else "LOWER than"
    text = (
        f"Direction, computed: {subject} is {relation} {reference}, "
        f"by {_amount(comparison)}."
    )

    if comparison.unit == "USD" and kind == "temporal_precedence":
        text += (
            " Money may be recoverable."
            if comparison.sign > 0
            else (
                " The vendor charged us less than agreed, so nothing is owed to us — "
                "a refund, credit or recovery is not the remedy here."
            )
        )
    return text


# Phrases that assert one value stands above another, and phrases that assert the
# reverse. Remedies count as claims: "request a refund" asserts an overcharge just as
# plainly as the word "higher" does, and it is the half that causes the wrong action.
_HIGHER = (
    r"higher",
    r"greater",
    r"exceed(?:s|ed|ing)?",
    r"above",
    r"more than",
    r"in excess",
    r"over-?bill(?:s|ed|ing)?",
    r"over-?charg(?:e|es|ed|ing)",
    r"over-?stat(?:e|es|ed|ing)",
    r"over-?paid",
    r"refund(?:s|ed)?",
    r"recover(?:s|ed|y)?",
    r"recoup(?:s|ed)?",
    r"claw-?back",
    r"too high",
    r"inflated",
)

_LOWER = (
    r"lower",
    r"less than",
    r"below",
    r"under-?bill(?:s|ed|ing)?",
    r"under-?charg(?:e|es|ed|ing)",
    r"under-?stat(?:e|es|ed|ing)",
    r"discount(?:s|ed)?",
    r"too low",
    r"shortfall",
    r"short-?paid",
)

# A window before the phrase, long enough to catch "is not higher" and "no more than"
# without reaching back into an unrelated clause.
_NEGATION = re.compile(r"\b(?:not|never|no|nor|n't|without)\b[^.;]{0,12}$", re.IGNORECASE)


def _asserted(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        for match in re.finditer(rf"\b{term}\b", text, flags=re.IGNORECASE):
            if _NEGATION.search(text[: match.start()]):
                # "not higher than" is a denial, not an assertion of the opposite.
                # Reading it as "lower" would be inventing a claim to then judge.
                continue
            return True
    return False


def claimed_sign(text: str) -> int | None:
    """The direction a piece of prose asserts, or None if it does not assert one.

    None covers three cases that must not be treated as claims: prose with no
    directional language, prose whose directional language is negated, and prose
    containing both directions at once — "billed above the amendment rate but below the
    original" is coherent and unjudgeable by keyword.
    """
    higher = _asserted(text or "", _HIGHER)
    lower = _asserted(text or "", _LOWER)
    if higher == lower:  # neither, or both
        return None
    return 1 if higher else -1


def contradicts(explanation: str, comparison: Comparison | None) -> bool:
    """Whether the prose asserts a direction the arithmetic rules out.

    Equal magnitudes are excluded: a candidate whose two values compare equal is not a
    difference anyone can describe backwards, and the comparators do not raise one.
    """
    if comparison is None or comparison.sign == 0:
        return False
    claimed = claimed_sign(explanation)
    return claimed is not None and claimed != comparison.sign
