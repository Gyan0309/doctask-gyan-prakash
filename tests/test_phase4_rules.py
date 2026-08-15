"""Phase 4: the rules engine (Stage A).

Deterministic, so tested exhaustively with no model and no key — including the cases
where a rule must NOT fire, which is where a compliance report earns or loses its
credibility.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest

from ledger.domain.reconcile import FactView, reconcile
from ledger.domain.rules import (
    RuleError,
    evaluate,
    evaluate_expression,
    load_rules,
)

PLAYBOOK = Path(__file__).resolve().parents[1] / "rules" / "playbook.yaml"


def _fact(predicate, value, kind="msa", subject="Northwind", day=None, scope=None):
    return FactView(
        fact_id=uuid4(),
        predicate=predicate,
        subject=subject,
        value_raw=value,
        value_norm=None,
        unit=None,
        effective_date=day or date(2024, 1, 1),
        document_id=uuid4(),
        document_kind=kind,
        scope=scope,
    )


class TestPlaybookLoading:
    def test_the_shipped_playbook_loads(self) -> None:
        ruleset_id, rules = load_rules(PLAYBOOK)
        assert ruleset_id
        assert len(rules) >= 5
        assert {r.code for r in rules} >= {"LIAB-01", "PAY-01", "RENEW-01", "LAW-01"}

    def test_an_unknown_rule_kind_is_refused_at_load_time(self, tmp_path) -> None:
        """A new *kind* is a code change. Accepting one silently would mean shipping a
        rule that never fires, which looks exactly like a rule that passes."""
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "ruleset_id: x\nrules:\n  - code: A\n    kind: vibes_based\n"
            "    predicate: hourly_rate\n    severity: low\n",
            encoding="utf-8",
        )
        with pytest.raises(RuleError, match="not one of"):
            load_rules(bad)

    def test_a_rule_missing_its_bound_is_refused(self, tmp_path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "ruleset_id: x\nrules:\n  - code: A\n    kind: numeric_max\n"
            "    predicate: payment_terms_days\n    severity: low\n",
            encoding="utf-8",
        )
        with pytest.raises(RuleError, match="config.max"):
            load_rules(bad)

    def test_duplicate_codes_are_refused(self, tmp_path) -> None:
        """Two rules with one code means one of them is unreachable."""
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "ruleset_id: x\nrules:\n"
            "  - {code: A, kind: presence_required, predicate: governing_law, severity: low}\n"
            "  - {code: A, kind: presence_required, predicate: hourly_rate, severity: low}\n",
            encoding="utf-8",
        )
        with pytest.raises(RuleError, match="duplicate"):
            load_rules(bad)


class TestExpressionEvaluation:
    def test_a_bound_referring_to_another_fact(self) -> None:
        result = evaluate_expression("2 * annual_fees", {"annual_fees": Decimal(600000)})
        assert result == Decimal(1200000)

    def test_plain_numbers(self) -> None:
        assert evaluate_expression("30", {}) == Decimal(30)

    def test_a_missing_input_yields_none_rather_than_zero(self) -> None:
        """Treating an absent fact as zero would make every bound trivially breached."""
        assert evaluate_expression("2 * annual_fees", {}) is None

    def test_a_dangling_operator_is_refused(self) -> None:
        """Returning the partial total would evaluate the rule against a bound its
        author never wrote."""
        assert evaluate_expression("2 *", {}) is None

    def test_arbitrary_python_is_not_executed(self) -> None:
        """A playbook is configuration. Configuration that can execute Python is a
        remote code execution hole wearing a friendly name."""
        assert evaluate_expression("__import__('os').system('echo pwned')", {}) is None


class TestRuleEvaluation:
    def setup_method(self) -> None:
        _, self.rules = load_rules(PLAYBOOK)

    def _violations(self, facts):
        return evaluate(self.rules, reconcile(facts))

    def test_payment_terms_worse_than_net_30_is_flagged(self) -> None:
        found = self._violations([_fact("payment_terms_days", "60")])
        assert [v.rule.code for v in found] == ["PAY-01"] or "PAY-01" in [
            v.rule.code for v in found
        ]

    def test_payment_terms_of_exactly_net_30_passes(self) -> None:
        """Boundary. `numeric_max: 30` means 30 is acceptable, not one short of it."""
        found = self._violations([_fact("payment_terms_days", "30")])
        assert "PAY-01" not in [v.rule.code for v in found]

    def test_a_notice_period_shorter_than_the_minimum_is_flagged(self) -> None:
        found = self._violations([_fact("termination_notice_days", "15")])
        assert "NOTICE-01" in [v.rule.code for v in found]

    def test_a_liability_cap_above_twice_annual_fees_is_flagged(self) -> None:
        """The cross-fact rule: the bound is computed from another fact entirely."""
        found = self._violations(
            [
                _fact("annual_fees", "$600,000"),
                _fact("liability_cap", "$2,500,000"),
            ]
        )
        assert "LIAB-01" in [v.rule.code for v in found]

    def test_a_liability_cap_within_the_bound_passes(self) -> None:
        found = self._violations(
            [
                _fact("annual_fees", "$600,000"),
                _fact("liability_cap", "$1,000,000"),
            ]
        )
        assert "LIAB-01" not in [v.rule.code for v in found]

    def test_an_uncomputable_bound_is_reported_not_silently_skipped(self) -> None:
        """No annual_fees means LIAB-01 cannot run. A silent skip is indistinguishable
        from a pass, and 'we checked' would then be a lie."""
        found = self._violations([_fact("liability_cap", "$2,500,000")])
        liab = [v for v in found if v.rule.code == "LIAB-01"]
        assert len(liab) == 1
        assert "could not be evaluated" in liab[0].explanation

    def test_a_missing_required_term_is_flagged(self) -> None:
        found = self._violations([_fact("hourly_rate", "$195")])
        assert "LAW-01" in [v.rule.code for v in found]

    def test_a_present_required_term_passes(self) -> None:
        found = self._violations([_fact("governing_law", "State of Delaware")])
        assert "LAW-01" not in [v.rule.code for v in found]

    def test_rules_run_against_the_governing_value_not_superseded_ones(self) -> None:
        """The most important false positive to avoid. Checking superseded values
        reports violations an amendment already fixed — true, useless, and the fastest
        way to make a compliance report ignorable."""
        facts = [
            _fact("payment_terms_days", "45", "msa", day=date(2024, 3, 1)),
            _fact("payment_terms_days", "30", "amendment", day=date(2025, 7, 1)),
        ]
        assert "PAY-01" not in [v.rule.code for v in self._violations(facts)]

    def test_a_sow_is_not_judged_against_agreement_level_rules(self) -> None:
        """The playbook governs the agreement. Applying it to a scoped engagement
        reaches somewhere it was never meant to."""
        facts = [
            _fact("payment_terms_days", "60", "sow", scope="ALPHA"),
            _fact("governing_law", "Delaware"),
        ]
        assert "PAY-01" not in [v.rule.code for v in self._violations(facts)]

    def test_each_vendor_is_judged_separately(self) -> None:
        facts = [
            _fact("payment_terms_days", "30", subject="Northwind"),
            _fact("payment_terms_days", "60", subject="Cobalt"),
        ]
        offenders = {v.subject for v in self._violations(facts) if v.rule.code == "PAY-01"}
        assert offenders == {"Cobalt"}

    def test_a_fully_compliant_vendor_produces_no_violations(self) -> None:
        facts = [
            _fact("payment_terms_days", "30"),
            _fact("auto_renew_months", "12"),
            _fact("termination_notice_days", "60"),
            _fact("governing_law", "State of Delaware"),
            _fact("annual_fees", "$600,000"),
            _fact("liability_cap", "$1,000,000"),
        ]
        assert self._violations(facts) == []
