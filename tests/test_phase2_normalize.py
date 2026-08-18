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
            # A written number with its numeral and no unit word. This shape arrived
            # with a new extraction prompt and broke silently: Talus's payment-terms
            # finding vanished between two live runs, not because the terms changed but
            # because the value stopped being comparable.
            ("forty-five (45)", 45),
            ("thirty (30)", 30),
            ("ninety (90)", 90),
            ("net forty-five (45) days", 45),
        ],
    )
    def test_day_forms(self, raw, expected) -> None:
        assert normalize_days(raw) == Decimal(expected)

    @pytest.mark.parametrize(
        "raw",
        [
            # The liability-cap formula. An unanchored search for a parenthesised
            # numeral would return 3 from here, and a cap of 3 then gets compared
            # against real money — the fault this module's docstring warns about.
            "the lesser of three (3) times the fees and five million dollars ($5,000,000)",
            "three (3) times the fees charged on that matter",
            # OCR damage. A numeral we cannot read must stay refused rather than be
            # rescued into a plausible-looking wrong number.
            "si xty (6O)",
            "tvve1ve (l2)",
        ],
    )
    def test_a_numeral_inside_a_longer_phrase_is_still_refused(self, raw) -> None:
        """Accepting `word (N)` must not become accepting `(N)` anywhere. The whole
        value has to be the word-and-numeral pair, or it is not that shape."""
        assert normalize_days(raw) is None
        assert normalize_months(raw) is None

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("24 months", 24),
            ("twenty-four (24) months", 24),
            ("36 months", 36),
            ("2 years", 24),  # converted, because the rule is written in months
            ("24", 24),  # bare number, unit implied by the predicate
            ("twelve (12)", 12),  # numeral without its unit word
            ("sixty (60)", 60),
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


class TestTheSixValuesRunCb4ed630CouldNotRead:
    """The exact strings a live run reported as unnormalizable, kept verbatim.

    Their consequence is the reason this class exists rather than a line in the
    parametrize list above: three playbook checks stopped running for Talus and the
    only trace was three `low` severity notes. A finding that disappears looks
    identical to a finding that was never true.
    """

    @pytest.mark.parametrize(
        "predicate,raw,expected",
        [
            ("payment_terms_days", "forty-five (45)", 45),
            ("auto_renew_months", "twelve (12)", 12),
            ("termination_notice_days", "ninety (90)", 90),
            ("payment_terms_days", "thirty (30)", 30),
            ("termination_notice_days", "thirty (30)", 30),
        ],
    )
    def test_each_one_now_normalizes(self, predicate, raw, expected) -> None:
        result = normalize(predicate, raw)
        assert result is not None, f"{predicate}={raw!r} is still uncomparable"
        assert result.number == Decimal(expected)

    def test_a_zero_quantity_is_a_value_not_an_absence(self) -> None:
        """Found while fixing the above: the quantity branch chained with `or`, and
        Decimal("0") is falsy — so a zero-hour line normalized to None and dropped out
        of the arithmetic comparator without saying anything."""
        result = normalize("invoice_hours", "0")
        assert result is not None
        assert result.number == Decimal(0)


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
        # Scope is folded, so one column labelled differently by two documents is one
        # scope. The identifier is the folded form.
        values = {r.scope: r.governing.value_raw for r in resolutions}
        assert values[None] == "$195"
        assert values["alpha"] == "$210"

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


class TestARestatementDoesNotOverruleTheInstrument:
    """The renewal-notice trap, and the clearest case of a defence that did not work.

    `KIND_PRECEDENCE` already ranked `renewal_notice` *below* the MSA, with a comment
    saying a restatement must not become the governing source for the agreement's own
    terms. It became one anyway, because precedence is only a tie-break and `_sort_key`
    compares the effective date first.

    Measured live: Talus's payment terms resolved to `net forty-five (45) days` from the
    renewal notice, dated after Amendment No. 1, with the amendment's `net thirty (30)
    days` filed as superseded. The amendment was read correctly and then overruled by a
    courtesy letter repeating the old figure, because the letter was newer.
    """

    def test_a_later_notice_does_not_supersede_an_amendment(self) -> None:
        amendment = _fact("payment_terms_days", "net 30", "amendment", date(2025, 7, 1))
        notice = _fact("payment_terms_days", "net 45", "renewal_notice", date(2026, 5, 6))

        resolution = reconcile([amendment, notice])[0]

        assert resolution.governing is amendment
        assert resolution.governing.value_raw == "net 30"

    def test_the_restated_term_is_kept_as_an_observation(self) -> None:
        """Not discarded. A notice restating a term the agreement changed is exactly the
        discrepancy a reviewer should see — the corpus plants one deliberately, restating
        a 24-month renewal against an agreement that says 12."""
        agreement = _fact("auto_renew_months", "12 months", "msa", date(2023, 9, 1))
        notice = _fact("auto_renew_months", "24 months", "renewal_notice", date(2026, 5, 6))

        resolution = reconcile([agreement, notice])[0]

        assert resolution.governing is agreement
        assert [f.value_raw for f in resolution.observations] == ["24 months"]

    def test_a_term_only_a_notice_states_is_unsupported(self) -> None:
        notice = _fact("auto_renew_months", "24 months", "renewal_notice", date(2026, 5, 6))
        resolution = reconcile([notice])[0]

        assert resolution.governing is None
        assert resolution.status == "unsupported"


class TestAnUnidentifiedDocumentGovernsNothing:
    """A low rank is not a safeguard, because rank is only a tie-break.

    `unknown` was ranked -1, which read as safe and was not: `_sort_key` compares the
    effective date first, so a 2026 document nobody could identify beat the 2023 MSA it
    sat beside on date alone. That is how a data-protection addendum came to outrank the
    agreement it accompanied.
    """

    def test_a_later_unknown_document_does_not_beat_the_agreement(self) -> None:
        msa = _fact("liability_cap", "$2,000,000", "msa", date(2023, 6, 1))
        mystery = _fact("liability_cap", "$50,000", "unknown", date(2026, 4, 1))

        resolution = reconcile([msa, mystery])[0]

        assert resolution.governing is msa
        assert resolution.status == "agreed"

    def test_its_value_is_still_recorded_as_an_observation(self) -> None:
        """Not governing is not the same as being discarded. The value is exactly what
        a reviewer needs to see in order to decide what the document is."""
        msa = _fact("liability_cap", "$2,000,000", "msa", date(2023, 6, 1))
        mystery = _fact("liability_cap", "$50,000", "unknown", date(2026, 4, 1))

        resolution = reconcile([msa, mystery])[0]

        assert [f.value_raw for f in resolution.observations] == ["$50,000"]

    def test_a_term_only_an_unknown_document_states_is_unsupported(self) -> None:
        mystery = _fact("hourly_rate", "$999", "unknown", date(2026, 4, 1))
        resolution = reconcile([mystery])[0]

        assert resolution.governing is None
        assert resolution.status == "unsupported"

    def test_the_rank_still_loses_a_date_tie(self) -> None:
        """Belt and braces: the tie-break must agree with the stronger rule."""
        same_day = date(2025, 7, 1)
        msa = _fact("governing_law", "Delaware", "msa", same_day)
        mystery = _fact("governing_law", "Nevada", "unknown", same_day)

        assert reconcile([msa, mystery])[0].governing is msa


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
