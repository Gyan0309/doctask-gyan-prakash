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


def normalize_money(raw: str) -> Decimal | None:
    """Extract a monetary magnitude. Currency is assumed USD and asserted nowhere —
    the corpus is single-currency, and inventing a currency we did not read would be
    exactly the kind of confident guess this system exists to avoid."""
    match = re.search(rf"\$?\s*{_NUMBER}", raw)
    return _to_decimal(match.group(1)) if match else None


def normalize_days(raw: str) -> Decimal | None:
    """Days, including 'net 30' and a bare '30'."""
    if (value := _first_match(raw, _DAYS_PATTERNS)) is not None:
        return value
    return _bare_number(raw)


def normalize_months(raw: str) -> Decimal | None:
    """Months, converting from years where a term is stated that way."""
    if (months := _first_match(raw, _MONTHS_PATTERNS)) is not None:
        return months
    if (years := _first_match(raw, _YEAR_PATTERNS)) is not None:
        return years * 12
    return _bare_number(raw)


def normalize_percent(raw: str) -> Decimal | None:
    match = re.search(rf"{_NUMBER}\s*%", raw)
    if match:
        return _to_decimal(match.group(1))
    return _to_decimal(raw.strip()) if raw.strip().replace(".", "").isdigit() else None


def normalize(predicate: str, raw: str) -> NormalizedValue | None:
    """Normalize a raw value according to what its predicate means.

    Dispatching on predicate rather than sniffing the string is deliberate: "$95" and
    "95 days" are indistinguishable to a general parser once the symbol is stripped,
    but never ambiguous once you know the predicate is a rate or a notice period.
    """
    raw = (raw or "").strip()
    if not raw:
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
        value = _bare_number(raw) or _first_match(raw, [rf"{_NUMBER}\s*hour"])
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
