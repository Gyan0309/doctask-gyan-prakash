"""The direction of a difference, and the prose that describes it.

Built from a live defect rather than from imagination. Run `cb4ed630` produced a finding
whose numbers were correct and whose sentence was backwards: a partner rate billed at
$780 against a governing $840 — an under-billing — reported as "significantly higher
than the contractually agreed rate. Request a refund or corrected invoice."

The tests below fix both halves of that: that the direction reaches the adjudicator as a
computed fact, and that an explanation contradicting it does not reach a human.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal
from uuid import uuid4

from domain.adjudicate import adjudicate, build_prompt
from domain.conflicts import (
    find_arithmetic_mismatches,
    find_same_predicate_different_value,
    find_temporal_precedence_violations,
)
from domain.direction import Comparison, claimed_sign, compare, contradicts, sentence
from domain.reconcile import FactView
from providers.fake import FakeProvider


def _fact(predicate, value, kind, day=None, subject="Brightmoor", document=None):
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
        scope=None,
    )


class _Client:
    def __init__(self, provider):
        self._provider = provider

    def generate(self, prompt, *, stage, schema=None, cheap=False):
        return self._provider.generate(prompt, schema=schema, cheap=cheap)


def _scripted(explanation: str, severity: str = "high"):
    return _Client(
        FakeProvider(
            scripted={
                "You review discrepancies": json.dumps(
                    {
                        "verdicts": [
                            {
                                "index": 0,
                                "is_real_conflict": True,
                                "severity": severity,
                                "explanation": explanation,
                            }
                        ]
                    }
                )
            }
        )
    )


def _the_live_defect():
    """The actual Brightmoor pair: billed $780.00 against a governing $840."""
    agreement = _fact("hourly_rate", "$840", "msa", date(2026, 1, 1))
    invoice = _fact("invoice_rate", "$780.00", "invoice", date(2026, 5, 31))
    found = find_temporal_precedence_violations([agreement, invoice])
    assert len(found) == 1, "the under-billing must still be detected"
    return found


class TestTheComparatorsMeasureDirection:
    """A difference is measured where the magnitudes are, not inferred downstream."""

    def test_an_under_billing_carries_a_negative_sign(self) -> None:
        candidate = _the_live_defect()[0]

        assert candidate.comparison is not None
        assert candidate.comparison.sign == -1
        assert candidate.comparison.delta == Decimal("60.00")
        assert candidate.comparison.unit == "USD"

    def test_an_over_billing_carries_a_positive_sign(self) -> None:
        agreement = _fact("hourly_rate", "$840", "msa", date(2026, 1, 1))
        invoice = _fact("invoice_rate", "$900", "invoice", date(2026, 5, 31))

        candidate = find_temporal_precedence_violations([agreement, invoice])[0]

        assert candidate.comparison.sign == 1
        assert candidate.comparison.delta == Decimal("60")

    def test_two_governing_documents_are_measured_too(self) -> None:
        a = _fact("payment_terms_days", "net 30", "msa", date(2026, 1, 1))
        b = _fact("payment_terms_days", "net 45", "msa", date(2026, 1, 1))

        candidate = find_same_predicate_different_value([a, b])[0]

        assert candidate.comparison.sign == -1
        assert candidate.comparison.unit == "days"

    def test_an_arithmetic_mismatch_is_measured_against_its_own_product(self) -> None:
        """The product exists only inside the comparator, which is why direction is
        recorded there rather than recomputed from the candidate's two facts."""
        document = uuid4()
        total = _fact("invoice_amount", "$1,000", "invoice", document=document)
        rate = _fact("invoice_rate", "$100", "invoice", document=document)
        hours = _fact("invoice_hours", "8", "invoice", document=document)

        candidate = find_arithmetic_mismatches([total, rate, hours])[0]

        assert candidate.comparison.sign == 1  # billed 1000, components give 800
        assert candidate.comparison.delta == Decimal("200")

    def test_an_incomparable_pair_records_no_direction(self) -> None:
        """A value that would not normalize must yield no sign at all. Defaulting to
        zero would read as "these are equal", which is a claim we cannot make."""
        assert compare(None, Decimal(5), "USD") is None
        assert compare(Decimal(5), None, "USD") is None
        assert compare(Decimal(5), Decimal(5), None) is None


class TestTheAdjudicatorIsToldWhichWayItRuns:
    def test_the_prompt_states_the_computed_direction(self) -> None:
        prompt = build_prompt(_the_live_defect())

        assert "Direction, computed:" in prompt
        assert "LOWER than" in prompt

    def test_the_prompt_names_the_consequence_for_money(self) -> None:
        """"Lower" and "no money is owed to us" are the same fact, and only the second
        stops a reviewer opening a refund request."""
        prompt = build_prompt(_the_live_defect())

        assert "refund, credit or recovery is not the remedy" in prompt

    def test_an_over_billing_says_money_may_be_recoverable(self) -> None:
        agreement = _fact("hourly_rate", "$840", "msa", date(2026, 1, 1))
        invoice = _fact("invoice_rate", "$900", "invoice", date(2026, 5, 31))

        prompt = build_prompt(find_temporal_precedence_violations([agreement, invoice]))

        assert "HIGHER than" in prompt
        assert "Money may be recoverable" in prompt


class TestAnInvertedExplanationIsWithheld:
    """The load-bearing test. The prompt is a request; this is the guarantee."""

    def test_the_exact_live_wording_is_rejected(self) -> None:
        candidates = _the_live_defect()
        client = _scripted(
            "The billed hourly rate is significantly higher than the contractually "
            "agreed rate. Request a refund or corrected invoice."
        )

        result = adjudicate(candidates, client)[0]

        assert result.explanation_withheld is True
        assert "significantly higher" not in result.explanation
        assert "LOWER than" in result.explanation
        # A refund is now named only to rule it out, which is the opposite of the
        # instruction the reviewer would otherwise have acted on.
        assert "refund, credit or recovery is not the remedy" in result.explanation

    def test_the_finding_itself_survives_the_rejection(self) -> None:
        """Withholding wording must not withhold the discrepancy. The comparators
        proved it; only the sentence was wrong."""
        result = adjudicate(_the_live_defect(), _scripted("Request a refund."))[0]

        assert result.is_real_conflict is True
        assert "$780.00" in result.explanation and "$840" in result.explanation

    def test_a_correct_explanation_is_left_alone(self) -> None:
        """The good case from the same live run, which must pass through untouched."""
        candidates = _the_live_defect()
        client = _scripted(
            "The invoice rate is lower than the contract rate, which is favorable to "
            "the company. Confirm if this was a one-time discount.",
            severity="low",
        )

        result = adjudicate(candidates, client)[0]

        assert result.explanation_withheld is False
        assert result.explanation.startswith("The invoice rate is lower")
        assert result.severity == "low"

    def test_an_inverted_over_billing_explanation_is_rejected_too(self) -> None:
        """Symmetry matters: calling a real overcharge a discount would talk a
        reviewer out of a finding that is costing money."""
        agreement = _fact("hourly_rate", "$840", "msa", date(2026, 1, 1))
        invoice = _fact("invoice_rate", "$900", "invoice", date(2026, 5, 31))
        candidates = find_temporal_precedence_violations([agreement, invoice])

        result = adjudicate(candidates, _scripted("Billed below the agreed rate."))[0]

        assert result.explanation_withheld is True
        assert "HIGHER than" in result.explanation

    def test_the_severity_judgement_is_kept(self) -> None:
        """Only the prose was demonstrably wrong. Discarding the severity as well
        would be punishing a judgement we have no evidence against."""
        result = adjudicate(_the_live_defect(), _scripted("Request a refund.", "low"))[0]

        assert result.severity == "low"

    def test_a_candidate_with_no_measured_direction_is_never_second_guessed(self) -> None:
        """No direction means not checked, not "assume zero". An explanation must not
        be withheld on the basis of a comparison that was never made."""
        candidates = _the_live_defect()
        stripped = [
            type(candidates[0])(
                kind=candidates[0].kind,
                subject=candidates[0].subject,
                predicate=candidates[0].predicate,
                a=candidates[0].a,
                b=candidates[0].b,
                detail=candidates[0].detail,
                comparison=None,
            )
        ]

        result = adjudicate(stripped, _scripted("Request a refund."))[0]

        assert result.explanation_withheld is False


class TestReadingADirectionalClaim:
    """The checker errs towards silence. Everything it cannot read plainly, it passes."""

    def test_plain_claims_are_read(self) -> None:
        assert claimed_sign("billed above the agreed rate") == 1
        assert claimed_sign("the rate exceeds the contract") == 1
        assert claimed_sign("this is below the agreed figure") == -1
        assert claimed_sign("the vendor under-billed us") == -1

    def test_a_remedy_is_a_claim(self) -> None:
        """"Request a refund" asserts an overcharge as plainly as the word "higher",
        and it is the half that produces the wrong action."""
        assert claimed_sign("Request a refund or a corrected invoice.") == 1
        assert claimed_sign("Recover the difference from the vendor.") == 1

    def test_prose_with_no_direction_is_not_a_claim(self) -> None:
        assert claimed_sign("The two documents disagree; ask the vendor.") is None
        assert claimed_sign("") is None

    def test_a_negated_claim_is_not_treated_as_its_opposite(self) -> None:
        """"Not higher" denies a direction rather than asserting the other one.
        Reading it as "lower" would invent a claim in order to judge it."""
        assert claimed_sign("The billed rate is not higher than agreed.") is None

    def test_both_directions_at_once_is_left_alone(self) -> None:
        """"Above the amendment but below the original" is coherent, and no keyword
        test can adjudicate it. Silence is the honest answer."""
        assert (
            claimed_sign("Billed above the amendment rate but below the original.")
            is None
        )

    def test_equal_magnitudes_are_never_contradicted(self) -> None:
        assert contradicts("anything at all", Comparison(0, Decimal(0), "USD")) is False

    def test_an_unmeasured_comparison_contradicts_nothing(self) -> None:
        assert contradicts("Request a refund.", None) is False


class TestTheComputedSentence:
    def test_a_missing_comparison_produces_no_sentence(self) -> None:
        assert sentence("temporal_precedence", None) == ""

    def test_non_money_units_are_named(self) -> None:
        text = sentence("same_predicate_different_value", Comparison(-1, Decimal(15), "days"))

        assert "15 days" in text
        assert "$" not in text

    def test_the_refund_warning_is_scoped_to_money(self) -> None:
        """A notice period stated 15 days short is not a refund question, and pasting
        a money remedy onto it would be noise."""
        text = sentence("temporal_precedence", Comparison(-1, Decimal(15), "days"))

        assert "refund" not in text


class TestDirectionSurvivesTheCheckpoint:
    """Candidates cross a LangGraph checkpoint as JSON. A direction that does not
    survive the trip is a direction the adjudicator never sees."""

    def test_a_round_trip_preserves_the_measurement(self) -> None:
        original = Comparison(-1, Decimal("60.00"), "USD")

        restored = Comparison.from_payload(json.loads(json.dumps(original.as_payload())))

        assert restored == original

    def test_a_checkpoint_written_before_this_existed_degrades_to_unchecked(self) -> None:
        assert Comparison.from_payload(None) is None
        assert Comparison.from_payload({}) is None
        assert Comparison.from_payload({"sign": "sideways"}) is None
        assert Comparison.from_payload("not a dict") is None
