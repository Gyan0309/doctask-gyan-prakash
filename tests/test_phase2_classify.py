"""Phase 2: classification and its escalation branch.

Floor 1 asks that decisions genuinely change the path. The way to show that is not to
assert the happy path works — it is to force the *alternate* branch and prove the
graph went somewhere else.
"""

from __future__ import annotations

import json

import pytest

from ledger.domain.classify import KIND_PRECEDENCE, classify_document
from ledger.graph import route_after_classify
from ledger.providers.fake import FakeProvider


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
                "reasoning": "amends an existing MSA by reference",
            }
        )
        result = classify_document("Amendment No. 1 ...", "a.md", client)

        assert result.kind == "amendment"
        assert result.confidence == pytest.approx(0.94)

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
        from ledger.graph import build_graph

        nodes = set(build_graph().get_graph().nodes)
        assert {"classify", "escalate", "extract"} <= nodes
