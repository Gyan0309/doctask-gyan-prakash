"""Phase 2: classification and its escalation branch.

Floor 1 asks that decisions genuinely change the path. The way to show that is not to
assert the happy path works — it is to force the *alternate* branch and prove the
graph went somewhere else.
"""

from __future__ import annotations

import json

import pytest

from domain.classify import (
    DOCUMENT_KINDS,
    KIND_PRECEDENCE,
    Classification,
    accepted_kind,
    build_prompt,
    classify_document,
    escalation_reason,
    names_our_organisation,
)
from providers.fake import FakeProvider
from services.graph import route_after_classify


class _Client:
    """Minimal MeteredClient stand-in: this suite is about routing, not metering."""

    def __init__(self, provider):
        self._provider = provider

    def generate(self, prompt, *, stage, schema=None, cheap=False):
        return self._provider.generate(prompt, schema=schema, cheap=cheap)


def _client_returning(payload: dict) -> _Client:
    return _Client(FakeProvider(scripted={"untrusted_document": json.dumps(payload)}))


class TestClassification:
    def test_a_confident_classification_is_returned_intact(self) -> None:
        client = _client_returning(
            {
                "kind": "amendment",
                "vendor": "Northwind Analytics",
                "confidence": 0.94,
                "fits_kind": True,
                "reasoning": "amends an existing MSA by reference",
            }
        )
        result = classify_document("Amendment No. 1 ...", "a.md", client)

        assert result.kind == "amendment"
        assert result.confidence == pytest.approx(0.94)
        assert result.fits_kind is True

    def test_an_unrecognised_kind_becomes_unknown_at_zero_confidence(self) -> None:
        """An invented kind must not flow into precedence calculations, where it would
        silently rank last instead of being questioned."""
        client = _client_returning(
            {"kind": "purchase_order", "vendor": "X", "confidence": 0.99, "reasoning": ""}
        )
        result = classify_document("...", "a.md", client)

        assert result.kind == "unknown"
        assert result.confidence == 0.0

    def test_unparseable_output_fails_toward_escalation(self) -> None:
        """The safe direction to fail is 'ask a human', never 'assume msa'."""
        client = _Client(FakeProvider(scripted={"untrusted_document": "not json at all"}))
        result = classify_document("...", "a.md", client)

        assert result.kind == "unknown"
        assert result.confidence == 0.0

    def test_confidence_is_clamped(self) -> None:
        """A model returning 1.5 would otherwise clear every threshold forever,
        including ones deliberately set high."""
        client = _client_returning(
            {"kind": "msa", "vendor": "X", "confidence": 1.5, "reasoning": ""}
        )
        assert classify_document("...", "a.md", client).confidence == 1.0


class TestEscalationActuallyFires:
    """Built from the run that proved it never did.

    Four live runs, 29 documents, `escalations=0` — including five documents that were
    none of the six kinds the classifier knows. Every one was forced into the nearest
    category at 0.8 to 1.0 confidence, because "how confident are you?" is answered
    relative to the options offered and there was no way to answer "none of these".
    """

    THRESHOLD = 0.75

    def _fits(self, kind: str, confidence: float, fits: bool) -> Classification:
        return Classification(
            kind=kind,
            vendor="Talus Cloud Services",
            confidence=confidence,
            reasoning="",
            fits_kind=fits,
        )

    def test_a_well_fitting_confident_document_does_not_escalate(self) -> None:
        assert escalation_reason(self._fits("msa", 0.95, True), self.THRESHOLD) is None

    @pytest.mark.parametrize(
        "kind,confidence,document",
        [
            ("amendment", 0.95, "talus-dpa.md — a data-protection addendum"),
            ("invoice", 0.95, "talus-credit-note-2026-02.md — a credit note"),
            ("invoice", 0.8, "ardent-interim-application-07.txt"),
            ("amendment", 0.9, "vendor-cover-note-meridian.md — a stranger's letter"),
            ("sow", 1.0, "talus-order-form-001.md — an order form"),
        ],
    )
    def test_each_document_that_slipped_through_now_escalates(
        self, kind, confidence, document
    ) -> None:
        """High confidence must not be able to override a poor fit. Every one of these
        cleared the 0.75 threshold comfortably and was processed on a guess."""
        reason = escalation_reason(self._fits(kind, confidence, False), self.THRESHOLD)

        assert reason is not None, f"{document} still slips through"
        assert "nearest available kind" in reason

    def test_a_confident_unknown_no_longer_sails_through(self) -> None:
        """The hole this had regardless of fit: the model saying "I cannot type this"
        at 0.9 cleared the threshold and was processed at precedence -1."""
        reason = escalation_reason(self._fits("unknown", 0.9, True), self.THRESHOLD)

        assert reason is not None
        assert "declined to identify" in reason

    def test_low_confidence_still_escalates(self) -> None:
        reason = escalation_reason(self._fits("msa", 0.4, True), self.THRESHOLD)

        assert reason is not None and "below the 0.75 threshold" in reason

    def test_a_missing_fit_answer_is_read_as_a_poor_fit(self) -> None:
        """An older cached response has no such field. Defaulting it to True would
        reinstate silent acceptance for exactly the documents this catches."""
        client = _client_returning(
            {"kind": "amendment", "vendor": "X", "confidence": 0.99, "reasoning": ""}
        )
        result = classify_document("A data processing addendum...", "dpa.md", client)

        assert result.fits_kind is False
        assert escalation_reason(result, self.THRESHOLD) is not None

    def test_a_non_boolean_fit_answer_is_not_believed(self) -> None:
        client = _client_returning(
            {
                "kind": "msa",
                "vendor": "X",
                "confidence": 0.99,
                "fits_kind": "yes, definitely",
                "reasoning": "",
            }
        )
        assert classify_document("...", "a.md", client).fits_kind is False

    def test_the_prompt_asks_the_question_that_can_answer_none_of_these(self) -> None:
        prompt = build_prompt("Data Processing Addendum", "dpa.md")

        assert "fits_kind" in prompt
        # The distinction the whole fix turns on, stated in the prompt rather than left
        # for the model to infer from a confidence scale.
        assert "nearest answer" in prompt
        assert "a credit note is not an invoice" in prompt


class TestPrecedence:
    def test_an_amendment_outranks_the_agreement_it_amends(self) -> None:
        assert KIND_PRECEDENCE["amendment"] > KIND_PRECEDENCE["msa"]

    def test_an_invoice_ranks_below_everything_that_governs(self) -> None:
        """An invoice records what was billed; it must never set what was agreed."""
        for governing in ("msa", "amendment", "sow", "renewal_notice"):
            assert KIND_PRECEDENCE["invoice"] < KIND_PRECEDENCE[governing]


class TestRouting:
    """The branch itself. Both directions must be reachable, or it is not a decision."""

    def test_no_escalations_routes_straight_to_extraction(self) -> None:
        assert route_after_classify({"escalations": []}) == "extract"
        assert route_after_classify({}) == "extract"

    def test_any_escalation_diverts_the_graph_to_a_human(self) -> None:
        state = {"escalations": [{"document": "mystery.md", "confidence": 0.3}]}
        assert route_after_classify(state) == "escalate"

    def test_the_escalate_node_exists_in_the_compiled_graph(self) -> None:
        """Routing to a node that was never added fails at runtime, not at import."""
        from services.graph import build_graph

        nodes = set(build_graph().get_graph().nodes)
        assert {"classify", "escalate", "extract"} <= nodes


class TestADocumentThatIsNotOurs:
    """`kestrel-scan-ocr.md` is a subcontractor's agreement between two other companies.

    It was read correctly, classified `msa` at 0.95 with the right vendor, and silently
    added a vendor to the register that is not a counterparty at all. Nothing anywhere
    asked whether the document was ours.
    """

    OURS = "This Agreement is between Ridgeline Industries Ltd and Talus Cloud Services."
    THEIRS = "This Agreement is between Kestrel Refrigeration and Brand Cold Storage."

    def test_a_document_naming_us_passes(self) -> None:
        assert names_our_organisation(self.OURS, "Ridgeline Industries Ltd") is True

    def test_a_document_between_other_parties_does_not(self) -> None:
        assert names_our_organisation(self.THEIRS, "Ridgeline Industries Ltd") is False

    def test_case_and_spacing_do_not_decide_it(self) -> None:
        shouted = "RIDGELINE  INDUSTRIES LTD signs below."
        assert names_our_organisation(shouted, "Ridgeline Industries Ltd") is True

    def test_no_configured_organisation_means_not_checked(self) -> None:
        """None, not False. "We did not check" and "this is not ours" are different
        answers, and reporting the second for the first would flag every document in a
        deployment that never configured a name."""
        assert names_our_organisation(self.THEIRS, "") is None
        assert names_our_organisation(self.THEIRS, "   ") is None

    def test_an_empty_document_is_not_ours(self) -> None:
        assert names_our_organisation("", "Ridgeline Industries Ltd") is False

    def test_a_shared_first_word_is_not_a_match(self) -> None:
        """Load-bearing, not incidental. This corpus holds both "Meridian Retail Group"
        (a client) and "Meridian Interconnect BV" (a stranger who sent a fraudulent
        novation letter). Matching on a token would pass the forgery."""
        forgery = "Meridian Interconnect BV hereby novates the agreement."

        assert names_our_organisation(forgery, "Meridian Retail Group") is False

    def test_the_client_itself_still_matches(self) -> None:
        genuine = "This Agreement is between Meridian Retail Group and Northwind."

        assert names_our_organisation(genuine, "Meridian Retail Group") is True


class TestTheHumanIsAskedAnAnswerableQuestion:
    def test_none_of_these_is_one_of_the_options(self) -> None:
        """The escalation used to offer five governing kinds and no way out, which put
        the human in the same trap as the classifier: obliged to name a nearest kind for
        a document that is none of them. Answering "amendment" for a data-protection
        addendum is how one came to outrank the agreement it accompanied."""
        from services.graph import ESCALATION_OPTIONS

        assert "unknown" in ESCALATION_OPTIONS
        assert set(ESCALATION_OPTIONS) == set(DOCUMENT_KINDS)

    def test_every_option_says_whether_it_governs(self) -> None:
        """Two of the six answers silently strip a document's authority, and nothing at
        the gate said so.

        Live consequence, from an independent re-test: a rate revision letter escalated,
        and the model's own displayed reasoning argued it was "a unilateral notice rather
        than a bilateral contract amendment". Following that reasoning the reviewer
        answered `renewal_notice` — which is non-governing, so the 2026 rate card became
        observation-only and the register went on asserting 2024 rates. Correctly-billed
        2026 invoices then produced two `high` "money may be recoverable" findings
        against a vendor that had done nothing wrong.

        The reviewer was not careless. The gate offered a reasonable-sounding answer and
        concealed the one consequence that mattered.
        """
        from services.graph import escalation_options

        options = {o["kind"]: o for o in escalation_options()}

        assert set(options) == set(DOCUMENT_KINDS), "every answer must be described"
        assert options["renewal_notice"]["governs"] is False
        assert options["unknown"]["governs"] is False
        assert options["invoice"]["governs"] is False
        assert options["msa"]["governs"] is True
        assert options["amendment"]["governs"] is True
        assert options["sow"]["governs"] is True

    def test_what_an_option_does_is_derived_not_restated(self) -> None:
        """`OBSERVATIONAL_KINDS` decides what governs. If the gate keeps its own copy of
        that list, the two drift and the gate goes back to lying — quietly, and only for
        the kind someone forgot to update. This is the same class of fault as the vendor
        fold reaching `reconcile` but not `detect`: one truth, stated twice.
        """
        from domain.reconcile import OBSERVATIONAL_KINDS
        from services.graph import escalation_options

        for option in escalation_options():
            assert option["governs"] is (option["kind"] not in OBSERVATIONAL_KINDS)

    def test_every_option_carries_a_human_readable_effect(self) -> None:
        """The flag is for machines; the sentence is for the person answering. A new kind
        added without one leaves a blank where the consequence should be, so the suite
        fails rather than the gate."""
        from services.graph import escalation_options

        for option in escalation_options():
            assert option["description"].strip(), option["kind"]
            assert option["effect"].strip(), option["kind"]
            # The non-governing half is the half that surprised a reviewer, so it has to
            # be stated in words and not left to a boolean nobody reads.
            if not option["governs"]:
                assert "govern" in option["effect"].lower()

    def test_a_valid_answer_is_accepted(self) -> None:
        assert accepted_kind("amendment") == "amendment"
        assert accepted_kind("unknown") == "unknown"

    def test_case_and_padding_are_tolerated(self) -> None:
        """A human typed it, or a form sent it. Neither is a reason to lose the answer."""
        assert accepted_kind(" Amendment ") == "amendment"

    @pytest.mark.parametrize("answer", ["purchase_order", "", "  ", None, 7, ["msa"]])
    def test_anything_else_is_refused_rather_than_stored(self, answer) -> None:
        """Stored, it would reach `KIND_PRECEDENCE.get(kind, -1)` and take the default
        — a typo silently deciding what governs. Refusing leaves the document
        escalated, which is the loud failure instead of the quiet one."""
        assert accepted_kind(answer) is None
