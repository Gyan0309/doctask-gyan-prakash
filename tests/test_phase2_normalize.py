"""Phase 2: normalization and reconciliation.

Pure logic, no database, no model, no key. This is the layer that turns "these
documents disagree" into arithmetic, so it earns exhaustive testing — and gets it
cheaply, because nothing here needs a network.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest

from domain.normalize import (
    normalize,
    normalize_date,
    normalize_days,
    normalize_money,
    normalize_months,
)
from domain.reconcile import FactView, governing_value_at, reconcile


class TestMoney:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("$195", 195),
            ("$195.00", 195),
            ("195", 195),
            ("$195 per hour", 195),
            ("$2,500,000", 2500000),
            ("$2,500,000 in aggregate", 2500000),
            ("USD 180.50", 180.50),
        ],
    )
    def test_money_forms_reduce_to_one_number(self, raw, expected) -> None:
        """Every one of these appears in the corpus. If they do not converge, the
        contradiction between an amendment and an invoice is never detected."""
        assert normalize_money(raw) == Decimal(str(expected))

    def test_formatting_differences_do_not_survive_normalization(self) -> None:
        """$180.00 and $180 must be the same value — otherwise a cosmetic difference
        in a source document reads as a change to the register."""
        a = normalize("hourly_rate", "$180.00")
        b = normalize("hourly_rate", "$180 per hour")
        assert a is not None and b is not None
        assert a.as_text() == b.as_text()

    def test_unparseable_money_returns_none_rather_than_guessing(self) -> None:
        assert normalize_money("to be agreed") is None
        assert normalize("hourly_rate", "market rate") is None

    def test_every_predicate_the_extractor_can_emit_normalizes_a_bare_number(self) -> None:
        """The regression that mattered. Extraction returns bare magnitudes, so any
        numeric predicate that cannot accept one is silently excluded from conflict
        detection — the failure is invisible until you notice the register is thin."""
        from domain.normalize import (
            DAY_PREDICATES,
            MONEY_PREDICATES,
            MONTH_PREDICATES,
            PERCENT_PREDICATES,
        )

        numeric = MONEY_PREDICATES | DAY_PREDICATES | MONTH_PREDICATES | PERCENT_PREDICATES
        for predicate in numeric:
            result = normalize(predicate, "30")
            assert result is not None, f"{predicate} cannot normalize a bare number"
            assert result.number == Decimal(30)


class TestDurations:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("net 30", 30),
            ("net 45 days", 45),
            ("30 days", 30),
            ("60 days written notice", 60),
            # A bare number is what extraction actually returns most of the time: the
            # model reads the predicate as naming the unit and hands back only the
            # magnitude. Rejecting these made every duration term uncomparable and
            # flooded the review queue — found in a live run, not in review.
            ("45", 45),
            ("15", 15),
        ],
    )
    def test_day_forms(self, raw, expected) -> None:
        assert normalize_days(raw) == Decimal(expected)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("24 months", 24),
            ("twenty-four (24) months", 24),
            ("36 months", 36),
            ("2 years", 24),  # converted, because the rule is written in months
            ("24", 24),  # bare number, unit implied by the predicate
        ],
    )
    def test_month_forms_including_written_numerals(self, raw, expected) -> None:
        """Contracts write 'twenty-four (24) months' constantly. The digits are always
        there, so parsing the numeral is enough."""
        assert normalize_months(raw) == Decimal(expected)

    def test_days_and_months_never_compare_by_magnitude_alone(self) -> None:
        """30 days and 30 months share a number and mean nothing alike. The unit is
        carried so a comparison can refuse to run."""
        days = normalize("payment_terms_days", "net 30")
        months = normalize("auto_renew_months", "30 months")
        assert days is not None and months is not None
        assert days.number == months.number
        assert days.unit != months.unit


class TestDates:
    def test_iso_dates_parse(self) -> None:
        assert normalize_date("2025-07-01") == date(2025, 7, 1)
        assert normalize_date("effective 2026-01-01") == date(2026, 1, 1)

    def test_ambiguous_and_missing_dates_return_none(self) -> None:
        """Refusing to guess is the point. Inferring DD/MM vs MM/DD wrongly reorders
        an amendment chain and makes the system confidently report a wrong value."""
        assert normalize_date(None) is None
        assert normalize_date("sometime next quarter") is None

    def test_impossible_date_is_refused_not_clamped(self) -> None:
        assert normalize_date("2025-02-30") is None


def _fact(predicate, value, kind, day=None, subject="Northwind", scope=None) -> FactView:
    return FactView(
        fact_id=uuid4(),
        predicate=predicate,
        subject=subject,
        value_raw=value,
        value_norm=None,
        unit=None,
        effective_date=day,
        document_id=uuid4(),
        document_kind=kind,
        scope=scope,
    )


class TestReconciliation:
    def test_the_later_amendment_governs(self) -> None:
        """The amendment chain, which is the core of the corpus."""
        msa = _fact("hourly_rate", "$180", "msa", date(2024, 3, 1))
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))

        resolution = reconcile([msa, amendment])[0]

        assert resolution.governing is amendment
        assert resolution.superseded == [msa]
        assert resolution.status == "agreed"

    def test_a_superseded_value_is_retained_not_deleted(self) -> None:
        """I4. The superseded value is exactly what a late invoice contradicts —
        deleting it would destroy the evidence for the system's best finding."""
        msa = _fact("hourly_rate", "$180", "msa", date(2024, 3, 1))
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))

        resolution = reconcile([msa, amendment])[0]
        assert "$180" in [f.value_raw for f in resolution.superseded]

    def test_an_invoice_never_governs_what_was_agreed(self) -> None:
        """An invoice records what was billed. Letting it govern would mean a supplier
        could change the contract by billing differently."""
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))
        invoice = _fact("invoice_rate", "$180", "invoice", date(2025, 9, 30))
        invoice = FactView(**{**invoice.__dict__, "predicate": "hourly_rate"})

        resolution = reconcile([amendment, invoice])[0]

        assert resolution.governing is amendment
        assert invoice in resolution.observations

    def test_a_sow_rate_does_not_compete_with_the_agreement_rate(self) -> None:
        """A $210 specialist rate for one project is not a contradiction of the $195
        standard rate. Treating it as one manufactures a conflict that does not exist —
        and a system that cries wolf gets ignored."""
        standard = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))
        specialist = _fact("hourly_rate", "$210", "sow", date(2025, 2, 1), scope="ALPHA")

        resolutions = reconcile([standard, specialist])

        assert len(resolutions) == 2, "scoped and unscoped terms resolve separately"
        values = {r.scope: r.governing.value_raw for r in resolutions}
        assert values[None] == "$195"
        assert values["ALPHA"] == "$210"

    def test_a_term_with_only_an_invoice_is_unsupported_not_asserted(self) -> None:
        """No agreement sets this term, only a bill records it. The register says
        unsupported rather than promoting what was billed into what was agreed."""
        invoice = _fact("hourly_rate", "$180", "invoice", date(2025, 9, 30))
        resolution = reconcile([invoice])[0]

        assert resolution.governing is None
        assert resolution.status == "unsupported"

    def test_undated_facts_cannot_displace_dated_ones(self) -> None:
        """A missing date must degrade to 'does not govern', never to 'governs
        everything'."""
        dated = _fact("liability_cap", "$2,500,000", "amendment", date(2026, 1, 1))
        undated = _fact("liability_cap", "$999", "amendment", None)

        assert reconcile([dated, undated])[0].governing is dated

    def test_precedence_breaks_a_date_tie(self) -> None:
        same_day = date(2025, 7, 1)
        msa = _fact("payment_terms_days", "net 45", "msa", same_day)
        amendment = _fact("payment_terms_days", "net 30", "amendment", same_day)

        assert reconcile([msa, amendment])[0].governing is amendment


class TestHistoricalGoverningValue:
    """Judging a September invoice needs the rate *in September*, not the rate now."""

    def test_the_value_in_force_on_a_past_date_is_recoverable(self) -> None:
        msa = _fact("hourly_rate", "$180", "msa", date(2024, 3, 1))
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))
        facts = [msa, amendment]

        assert governing_value_at(facts, date(2025, 1, 1)) is msa
        assert governing_value_at(facts, date(2025, 9, 30)) is amendment

    def test_a_future_amendment_does_not_apply_retroactively(self) -> None:
        """Otherwise every historical invoice is flagged the moment a new amendment
        lands — a false-positive machine."""
        msa = _fact("liability_cap", "$1,000,000", "msa", date(2024, 3, 1))
        future = _fact("liability_cap", "$2,500,000", "amendment", date(2026, 1, 1))

        assert governing_value_at([msa, future], date(2025, 9, 30)) is msa

    def test_no_governing_value_before_any_document_took_effect(self) -> None:
        msa = _fact("hourly_rate", "$180", "msa", date(2024, 3, 1))
        assert governing_value_at([msa], date(2023, 1, 1)) is None
