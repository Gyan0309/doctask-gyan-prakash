"""Phase 3: conflict detection and adjudication.

The comparators are deterministic, so they can be tested exhaustively with no model and
no key — including, crucially, the cases where they must stay *silent*. A detector that
fires on everything is worthless in a way that is easy to miss, because every test
asserting "it found the thing" still passes.
"""

from __future__ import annotations

import json
from datetime import date
from uuid import uuid4

from ledger.domain.adjudicate import adjudicate, build_prompt
from ledger.domain.conflicts import (
    detect,
    find_arithmetic_mismatches,
    find_same_predicate_different_value,
    find_temporal_precedence_violations,
)
from ledger.domain.reconcile import FactView
from ledger.graph import route_after_detection
from ledger.providers.fake import FakeProvider


def _fact(
    predicate, value, kind, day=None, subject="Northwind", scope=None, document=None
) -> FactView:
    return FactView(
        fact_id=uuid4(),
        predicate=predicate,
        subject=subject,
        value_raw=value,
        value_norm=None,
        unit=None,
        effective_date=day,
        document_id=document or uuid4(),
        document_kind=kind,
        scope=scope,
    )


class TestTemporalPrecedence:
    """The headline conflict: an invoice billing a rate that was already superseded."""

    def test_an_invoice_billing_a_superseded_rate_is_caught(self) -> None:
        msa = _fact("hourly_rate", "$180", "msa", date(2024, 3, 1))
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))
        invoice = _fact("hourly_rate", "$180", "invoice", date(2025, 9, 30))

        found = find_temporal_precedence_violations([msa, amendment, invoice])

        assert len(found) == 1
        assert found[0].kind == "temporal_precedence"
        # The detail states the arithmetic, so the finding survives without a model.
        assert "$180" in found[0].detail and "$195" in found[0].detail

    def test_an_invoice_billing_the_correct_rate_is_silent(self) -> None:
        """Half the value of a detector is what it declines to report."""
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))
        invoice = _fact("hourly_rate", "$195", "invoice", date(2025, 9, 30))

        assert find_temporal_precedence_violations([amendment, invoice]) == []

    def test_an_invoice_predating_the_amendment_is_not_a_violation(self) -> None:
        """The invoice was right when it was written. Comparing against today's value
        would flag every historical document the moment an amendment lands."""
        msa = _fact("hourly_rate", "$180", "msa", date(2024, 3, 1))
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))
        early_invoice = _fact("hourly_rate", "$180", "invoice", date(2025, 1, 15))

        assert find_temporal_precedence_violations([msa, amendment, early_invoice]) == []

    def test_an_invoice_rate_is_compared_against_the_agreement_rate(self) -> None:
        """The regression that mattered most.

        An invoice's rate extracts as `invoice_rate` and an agreement's as
        `hourly_rate`. Grouped by predicate name alone the two never meet, so the
        central conflict of the whole domain was structurally undetectable — the
        comparator stayed silent no matter how wrong the invoice was. Every unit test
        still passed, because they all used one predicate name on both sides.
        """
        msa = _fact("hourly_rate", "$180", "msa", date(2024, 3, 1))
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))
        invoice = _fact("invoice_rate", "$180", "invoice", date(2025, 9, 30))

        found = find_temporal_precedence_violations([msa, amendment, invoice])

        assert len(found) == 1, "an invoice rate must be judged against the agreed rate"
        assert "$195" in found[0].detail

    def test_one_document_stating_a_value_twice_yields_one_finding(self) -> None:
        """An invoice gives its rate in the line-item table and again in prose, so it
        is extracted twice under different predicate names. That is one problem, not
        two — and a review queue that shows it twice teaches reviewers to skim."""
        doc = uuid4()
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))
        as_prose = _fact(
            "hourly_rate", "$180 per hour", "invoice", date(2025, 9, 30), document=doc
        )
        as_line_item = _fact(
            "invoice_rate", "$180.00", "invoice", date(2025, 9, 30), document=doc
        )

        found = find_temporal_precedence_violations([amendment, as_prose, as_line_item])
        assert len(found) == 1

    def test_two_different_invoices_each_get_their_own_finding(self) -> None:
        """Deduplication must not collapse genuinely separate documents."""
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))
        september = _fact(
            "invoice_rate", "$180", "invoice", date(2025, 9, 30), document=uuid4()
        )
        october = _fact(
            "invoice_rate", "$170", "invoice", date(2025, 10, 31), document=uuid4()
        )

        assert len(find_temporal_precedence_violations([amendment, september, october])) == 2

    def test_an_undated_observation_is_not_judged(self) -> None:
        """With no date there is no 'value in force at the time' to compare against.
        Guessing one would invent the conflict."""
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))
        undated = _fact("hourly_rate", "$180", "invoice", None)

        assert find_temporal_precedence_violations([amendment, undated]) == []


class TestSamePredicateDifferentValue:
    def test_two_equally_authoritative_documents_disagreeing_is_a_conflict(self) -> None:
        same_day = date(2025, 7, 1)
        a = _fact("liability_cap", "$1,000,000", "msa", same_day)
        b = _fact("liability_cap", "$2,000,000", "msa", same_day)

        found = find_same_predicate_different_value([a, b])
        assert len(found) == 1
        assert "neither supersedes" in found[0].detail

    def test_a_later_amendment_is_supersession_not_conflict(self) -> None:
        """The single most important false positive to avoid. Reporting this would
        flag every amendment chain in the corpus as a problem."""
        msa = _fact("hourly_rate", "$180", "msa", date(2024, 3, 1))
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))

        assert find_same_predicate_different_value([msa, amendment]) == []

    def test_a_scoped_sow_rate_does_not_conflict_with_the_agreement_rate(self) -> None:
        """$210 for specialist work is not a contradiction of $195 standard."""
        standard = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))
        specialist = _fact(
            "hourly_rate", "$210", "sow", date(2025, 7, 1), scope="ALPHA"
        )

        assert find_same_predicate_different_value([standard, specialist]) == []

    def test_identical_values_stated_twice_are_not_a_conflict(self) -> None:
        """Restating a term is normal in contracts. $180.00 and $180 are the same
        number, and only normalization makes that visible."""
        day = date(2024, 3, 1)
        a = _fact("hourly_rate", "$180", "msa", day)
        b = _fact("hourly_rate", "$180.00 per hour", "msa", day)

        assert find_same_predicate_different_value([a, b]) == []

    def test_different_vendors_are_never_compared(self) -> None:
        day = date(2024, 3, 1)
        one = _fact("hourly_rate", "$180", "msa", day, subject="Northwind")
        other = _fact("hourly_rate", "$95", "msa", day, subject="Cobalt")

        assert find_same_predicate_different_value([one, other]) == []


class TestArithmeticMismatch:
    def test_an_invoice_whose_total_does_not_multiply_out_is_caught(self) -> None:
        doc = uuid4()
        amount = _fact("invoice_amount", "$20,500.00", "invoice", document=doc)
        rate = _fact("invoice_rate", "$195.00", "invoice", document=doc)
        hours = _fact("invoice_hours", "100", "invoice", document=doc)

        found = find_arithmetic_mismatches([amount, rate, hours])

        assert len(found) == 1
        assert "19500" in found[0].detail.replace(",", "")

    def test_a_correct_invoice_is_silent(self) -> None:
        doc = uuid4()
        facts = [
            _fact("invoice_amount", "$21,600.00", "invoice", document=doc),
            _fact("invoice_rate", "$180.00", "invoice", document=doc),
            _fact("invoice_hours", "120", "invoice", document=doc),
        ]
        assert find_arithmetic_mismatches(facts) == []

    def test_components_from_different_documents_are_never_multiplied(self) -> None:
        """Two invoices' numbers are unrelated. Crossing them would generate a
        mismatch out of thin air."""
        facts = [
            _fact("invoice_amount", "$21,600.00", "invoice", document=uuid4()),
            _fact("invoice_rate", "$195.00", "invoice", document=uuid4()),
            _fact("invoice_hours", "100", "invoice", document=uuid4()),
        ]
        assert find_arithmetic_mismatches(facts) == []

    def test_an_incomplete_invoice_is_not_guessed_at(self) -> None:
        doc = uuid4()
        facts = [
            _fact("invoice_amount", "$21,600.00", "invoice", document=doc),
            _fact("invoice_rate", "$180.00", "invoice", document=doc),
        ]  # no hours
        assert find_arithmetic_mismatches(facts) == []


class TestUnitSafety:
    def test_values_in_different_units_are_never_compared(self) -> None:
        """30 days and 30 months share a number and mean nothing alike."""
        day = date(2025, 1, 1)
        days = _fact("payment_terms_days", "30", "msa", day)
        months = _fact("auto_renew_months", "30", "msa", day)

        assert detect([days, months]) == []


class TestCleanCorpus:
    def test_a_consistent_corpus_produces_no_candidates(self) -> None:
        facts = [
            _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1)),
            _fact("hourly_rate", "$195", "invoice", date(2025, 9, 30)),
            _fact("governing_law", "Delaware", "msa", date(2024, 3, 1)),
        ]
        assert detect(facts) == []

    def test_no_candidates_routes_past_adjudication(self) -> None:
        """Decision point 3. The clean path costs nothing because no model runs."""
        assert route_after_detection({"conflict_candidates": []}) == "compose"
        assert route_after_detection({}) == "compose"

    def test_any_candidate_routes_into_adjudication(self) -> None:
        assert route_after_detection({"conflict_candidates": [{"kind": "x"}]}) == "adjudicate"


class TestAdjudication:
    def _candidates(self):
        msa = _fact("hourly_rate", "$180", "msa", date(2024, 3, 1))
        amendment = _fact("hourly_rate", "$195", "amendment", date(2025, 7, 1))
        invoice = _fact("hourly_rate", "$180", "invoice", date(2025, 9, 30))
        return find_temporal_precedence_violations([msa, amendment, invoice])

    class _Client:
        def __init__(self, provider):
            self._provider = provider

        def generate(self, prompt, *, stage, schema=None, cheap=False):
            return self._provider.generate(prompt, schema=schema, cheap=cheap)

    def test_the_model_can_dismiss_a_candidate(self) -> None:
        candidates = self._candidates()
        client = self._Client(
            FakeProvider(
                scripted={
                    "You review discrepancies": json.dumps(
                        {
                            "verdicts": [
                                {
                                    "index": 0,
                                    "is_real_conflict": False,
                                    "severity": "low",
                                    "explanation": "Retainer billed under separate terms.",
                                }
                            ]
                        }
                    )
                }
            )
        )
        results = adjudicate(candidates, client)

        assert len(results) == 1
        assert results[0].is_real_conflict is False

    def test_a_candidate_the_model_ignores_is_still_reported(self) -> None:
        """Silence is not evidence against a discrepancy the arithmetic already
        proved. A model must not be able to delete a finding by omission."""
        candidates = self._candidates()
        client = self._Client(
            FakeProvider(scripted={"You review discrepancies": json.dumps({"verdicts": []})})
        )
        results = adjudicate(candidates, client)

        assert len(results) == 1
        assert results[0].is_real_conflict is True
        assert "No adjudication was returned" in results[0].explanation

    def test_unparseable_output_degrades_to_the_deterministic_finding(self) -> None:
        candidates = self._candidates()
        client = self._Client(
            FakeProvider(scripted={"You review discrepancies": "))) not json"})
        )
        results = adjudicate(candidates, client)

        assert len(results) == 1
        assert results[0].is_real_conflict is True
        # The arithmetic survives the model being useless.
        assert "$180" in results[0].explanation

    def test_an_invalid_severity_falls_back_rather_than_propagating(self) -> None:
        candidates = self._candidates()
        client = self._Client(
            FakeProvider(
                scripted={
                    "You review discrepancies": json.dumps(
                        {
                            "verdicts": [
                                {
                                    "index": 0,
                                    "is_real_conflict": True,
                                    "severity": "catastrophic",
                                    "explanation": "x",
                                }
                            ]
                        }
                    )
                }
            )
        )
        assert adjudicate(candidates, client)[0].severity == "medium"

    def test_adjudication_is_never_called_with_nothing_to_judge(self) -> None:
        provider = FakeProvider()
        assert adjudicate([], self._Client(provider)) == []
        assert provider.calls == [], "a clean corpus must cost zero model calls"

    def test_all_candidates_go_in_one_call(self) -> None:
        """Batched because several candidates often share one root cause, and a model
        shown them together can say so."""
        candidates = self._candidates() * 3
        prompt = build_prompt(candidates)

        assert prompt.count("kind=temporal_precedence") == 3
        assert "[0]" in prompt and "[2]" in prompt
