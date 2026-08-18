"""Normalizing extracted values into something comparable.

This is where "the invoice bills a rate the amendment superseded" stops being a
sentence and becomes arithmetic. `$180.00` and `$180 per hour` and `180 USD/hr` must
all reduce to the same number, or the comparison that finds the contradiction never
fires.

Deliberately deterministic and model-free. Normalization is the layer everything
downstream trusts, so it must be exhaustively testable without a key — and a model
asked to "normalize this" will occasionally round, reformat, or helpfully correct a
value, which is precisely what must not happen to evidence.

The rule when parsing fails: return None and let the caller record the value as
unnormalized. A wrong number is far worse than an absent one, because a wrong number
still gets compared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

# Predicates whose values are money.
MONEY_PREDICATES = {
    "hourly_rate",
    "liability_cap",
    "annual_fees",
    "invoice_amount",
    "invoice_rate",
}

# Dimensionless counts. Their own unit so they can never be compared against money or
# durations by magnitude alone.
QUANTITY_PREDICATES = {"invoice_hours"}

# Predicates whose values are a count of days.
DAY_PREDICATES = {"payment_terms_days", "termination_notice_days"}

# Predicates whose values are a count of months.
MONTH_PREDICATES = {"auto_renew_months"}

PERCENT_PREDICATES = {"sla_credit_percent"}

_NUMBER = r"([0-9][0-9,]*(?:\.[0-9]+)?)"

# Written numbers appear in contracts constantly — "twenty-four (24) months" — but the
# digits are almost always right there in parentheses, so parsing the numeral is
# enough and a word-to-number table is not needed.
_MONTHS_PATTERNS = [
    rf"{_NUMBER}\s*(?:\(\s*[0-9]+\s*\))?\s*month",
    rf"\(\s*{_NUMBER}\s*\)\s*month",
]
_DAYS_PATTERNS = [
    rf"net\s*{_NUMBER}",
    rf"{_NUMBER}\s*(?:\(\s*[0-9]+\s*\))?\s*day",
    rf"\(\s*{_NUMBER}\s*\)\s*day",
]

_YEAR_PATTERNS = [rf"{_NUMBER}\s*year"]


@dataclass(frozen=True)
class NormalizedValue:
    """A comparable form of an extracted value.

    `number` is the canonical magnitude, `unit` names what it counts. Two facts are
    only ever compared when their units agree — comparing 30 days to 24 months by
    magnitude alone would silently generate nonsense conflicts.
    """

    number: Decimal
    unit: str

    def as_text(self) -> str:
        # Trailing zeros are stripped so 180.00 and 180 hash to the same section
        # content. Otherwise a purely cosmetic difference in the source reads as a
        # change to the register.
        quantized = self.number.normalize()
        return f"{quantized:f} {self.unit}".strip()


def _to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(raw.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _first_match(text: str, patterns: list[str]) -> Decimal | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _to_decimal(match.group(1))
    return None


def _bare_number(text: str) -> Decimal | None:
    """A value that is nothing but a number.

    Extraction frequently returns `"45"` rather than `"net 45 days"` — the model
    correctly reads the predicate as naming the unit and hands back only the
    magnitude. Refusing that was a real bug: a live run produced eight findings
    complaining it could not normalize `'45'`, `'24'`, `'60'` and so on, which made
    every duration term uncomparable and buried the review queue in noise.

    Safe because dispatch is on predicate. `payment_terms_days` means days; there is
    no ambiguity left for the string to resolve.
    """
    stripped = text.strip()
    return _to_decimal(stripped) if re.fullmatch(rf"{_NUMBER}", stripped) else None


# A written number with its numeral in parentheses and no unit word: "forty-five (45)",
# "twelve (12)", "two hundred and twenty five (225)".
#
# This shape arrived with a new extraction prompt, which returns the value without its
# trailing unit — stages 1 and 2 saw "net forty-five (45) days" and parsed it fine, and
# stage 3 saw "forty-five (45)" and could not. The consequence was silent and worse than
# a visible failure: Talus's payment-terms finding *disappeared* between runs, not
# because the terms became compliant but because the value stopped being comparable.
#
# Matched against the whole string, deliberately. An unanchored search would pull the
# `(3)` out of "the lesser of three (3) times the fees and five million dollars
# ($5,000,000)" and hand back 3 — which is the liability-cap fault this module's own
# docstring warns about, and turning a formula into a small number is exactly how a
# wrong value gets compared.
_WORD_THEN_NUMERAL = re.compile(rf"[a-z\s,\-]*\(\s*{_NUMBER}\s*\)", re.IGNORECASE)


def _parenthesised_numeral(text: str) -> Decimal | None:
    """The numeral from a word-and-numeral pair, when that is the entire value.

    Safe for the same reason `_bare_number` is safe: dispatch is on predicate, so the
    unit is already known and there is nothing left for the string to resolve. It stays
    strict about the digits themselves — an OCR-garbled "(6O)" or "(l2)" still fails,
    because a value we cannot read is one we must refuse rather than guess at.
    """
    match = _WORD_THEN_NUMERAL.fullmatch(text.strip())
    return _to_decimal(match.group(1)) if match else None


def normalize_money(raw: str) -> Decimal | None:
    """Extract a monetary magnitude. Currency is assumed USD and asserted nowhere —
    the corpus is single-currency, and inventing a currency we did not read would be
    exactly the kind of confident guess this system exists to avoid."""
    match = re.search(rf"\$?\s*{_NUMBER}", raw)
    return _to_decimal(match.group(1)) if match else None


def normalize_days(raw: str) -> Decimal | None:
    """Days, including 'net 30', a bare '30' and 'forty-five (45)'."""
    if (value := _first_match(raw, _DAYS_PATTERNS)) is not None:
        return value
    if (value := _bare_number(raw)) is not None:
        return value
    return _parenthesised_numeral(raw)


def normalize_months(raw: str) -> Decimal | None:
    """Months, converting from years where a term is stated that way."""
    if (months := _first_match(raw, _MONTHS_PATTERNS)) is not None:
        return months
    if (years := _first_match(raw, _YEAR_PATTERNS)) is not None:
        return years * 12
    if (value := _bare_number(raw)) is not None:
        return value
    return _parenthesised_numeral(raw)


def normalize_percent(raw: str) -> Decimal | None:
    match = re.search(rf"{_NUMBER}\s*%", raw)
    if match:
        return _to_decimal(match.group(1))
    if raw.strip().replace(".", "").isdigit():
        return _to_decimal(raw.strip())
    return _parenthesised_numeral(raw)


# Phrases that mean the value states a *computation* rather than an amount.
#
# A liability cap reading "the lesser of (i) three (3) times the fees charged on that
# matter and (ii) five million dollars ($5,000,000)" normalized to `Decimal('3')`,
# because the money parser takes the first digit run and that is the `(3)` in "three
# (3) times". The register displayed the clause honestly; the comparable magnitude
# behind it was nonsense. It stayed invisible only because `annual_fees` was missing, so
# the rule that would have used it returned "not checked" — the moment that fact
# appeared, LIAB-01 would have compared 3 against twice the annual fees and reported a
# **pass**. A liability cap certified compliant on a number that means nothing.
#
# The fix is not a better regex for finding the right digits. A value that is a formula
# is a different kind of thing from a value that is an amount, and no amount of digit
# hunting turns one into the other: "three times the fees" has no magnitude until you
# know the fees. So it is typed as unnormalizable and reported as such.
#
# Note this also refuses Talus's cap — "the greater of (a) two million dollars
# ($2,000,000) and ..." — which the old parser happened to read as $2,000,000. That was
# luck, not correctness: "the greater of $2,000,000 and twelve months of fees" is not
# $2,000,000, it is *at least* that, and comparing it as an equality is how a cap gets
# wrongly certified in the other direction.
#
# Deliberately narrow. "in aggregate" is not a marker, because "$2,500,000 in aggregate"
# is an amount; "aggregate of" is, because "the aggregate of the fees paid" is not.
_FORMULA_MARKERS = re.compile(
    r"\btimes\s+the\b"
    r"|\bmultiplied\s+by\b"
    r"|\bmultiple\s+of\b"
    r"|\b(?:greater|lesser|lower|higher)\s+of\b"
    r"|\baggregate\s+of\b"
    r"|\bpercent\s+of\b"
    r"|%\s*of\b",
    re.IGNORECASE,
)


def is_formula(raw: str) -> bool:
    """Whether a value expresses a computation instead of a quantity."""
    return bool(_FORMULA_MARKERS.search(raw or ""))


def why_not_comparable(predicate: str, raw: str) -> str:
    """Why a recorded value has no magnitude, in a phrase fit for a finding.

    Callers need this because "no number" has two very different causes and only one of
    them is a defect. A formula is a value we read correctly and cannot compare; an
    unreadable string is a value we failed to read. Reporting both as "missing" hides
    which one happened.
    """
    if not (raw or "").strip():
        return "empty"
    if is_formula(raw):
        return "a formula rather than an amount, so it has no magnitude to compare"
    if normalize(predicate, raw) is not None:
        return "comparable"
    return "not in a form that can be read as a number"


def normalize(predicate: str, raw: str) -> NormalizedValue | None:
    """Normalize a raw value according to what its predicate means.

    Dispatching on predicate rather than sniffing the string is deliberate: "$95" and
    "95 days" are indistinguishable to a general parser once the symbol is stripped,
    but never ambiguous once you know the predicate is a rate or a notice period.
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    # Checked once here rather than in each branch: every numeric predicate has the same
    # exposure, and a formula that slipped through one parser would be as wrong there as
    # in money.
    if is_formula(raw):
        return None

    if predicate in MONEY_PREDICATES:
        value = normalize_money(raw)
        return NormalizedValue(value, "USD") if value is not None else None

    if predicate in DAY_PREDICATES:
        value = normalize_days(raw)
        return NormalizedValue(value, "days") if value is not None else None

    if predicate in MONTH_PREDICATES:
        value = normalize_months(raw)
        return NormalizedValue(value, "months") if value is not None else None

    if predicate in PERCENT_PREDICATES:
        value = normalize_percent(raw)
        return NormalizedValue(value, "percent") if value is not None else None

    if predicate in QUANTITY_PREDICATES:
        # Chained on `is None` rather than with `or`: Decimal("0") is falsy, so `or`
        # discarded a legitimate zero and fell through to return None. A zero-hour line
        # then dropped out of the arithmetic comparator silently, which is the exact
        # failure mode this module exists to refuse.
        value = _bare_number(raw)
        if value is None:
            value = _first_match(raw, [rf"{_NUMBER}\s*hour"])
        if value is None:
            value = _parenthesised_numeral(raw)
        return NormalizedValue(value, "hours") if value is not None else None

    # Free-text predicates (governing_law). Case and spacing are normalized so
    # "State of Delaware" and "state of delaware" compare equal, but nothing else is
    # touched — collapsing further would start merging genuinely different values.
    return None


def normalize_text(raw: str) -> str:
    return " ".join((raw or "").split()).strip().lower()


_DATE_PATTERNS = [
    (r"(\d{4})-(\d{2})-(\d{2})", (1, 2, 3)),
    (r"(\d{2})/(\d{2})/(\d{4})", (3, 1, 2)),
]


def normalize_date(raw: str | None) -> date | None:
    """Parse an effective date. Ambiguous formats are refused rather than guessed.

    Notably absent: any attempt at DD/MM vs MM/DD inference. Getting that wrong
    silently reorders an amendment chain, which would make the system confidently
    report the wrong governing value — a failure far more damaging than declining to
    parse a date at all.
    """
    if not raw:
        return None
    for pattern, (y, m, d) in _DATE_PATTERNS:
        match = re.search(pattern, raw)
        if match:
            try:
                return date(
                    int(match.group(y)), int(match.group(m)), int(match.group(d))
                )
            except ValueError:
                return None
    return None
