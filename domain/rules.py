"""The rules engine — Stage A of the examine movement.

A contract playbook expressed as YAML. Adding a rule is a data change; adding a rule
*kind* is a code change, and four kinds cover the playbook. That line is the whole
point of the design: fifty rules cost nothing, but a fifty-first *kind* of rule should
make you stop and think.

Every rule here is deterministic and model-free. A rule that says "payment terms must
be net 30 or better" is arithmetic, and arithmetic evaluated by a language model is
arithmetic you cannot test. The model's judgement is spent on conflicts, where it is
genuinely needed.

`numeric_bound` evaluates a small expression against other facts about the same vendor
— "liability cap must not exceed 2x annual fees" needs the annual fees. The expression
grammar is deliberately tiny: a number, a predicate name, `*`, `+`, `-`. Not eval().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from domain.normalize import normalize, why_not_comparable
from domain.reconcile import Resolution

RULE_KINDS = {"numeric_max", "numeric_min", "numeric_bound", "presence_required"}
SEVERITIES = {"low", "medium", "high"}


class RuleError(ValueError):
    """A malformed playbook. Raised at load time, loudly.

    Rules are configuration, and configuration that is silently ignored is worse than
    configuration that fails: a rule nobody notices is broken looks exactly like a rule
    that passes.
    """


@dataclass(frozen=True)
class Rule:
    code: str
    kind: str
    predicate: str
    description: str
    config: dict[str, Any]
    severity: str


@dataclass(frozen=True)
class RuleViolation:
    rule: Rule
    subject: str
    observed: str | None
    explanation: str
    fact_id: object | None = None
    # False when the rule could not be run at all, as opposed to run and breached.
    checked: bool = True

    @property
    def severity(self) -> str:
        """The severity to report this at, which is not always the rule's own.

        Three of seven `high` findings in a live run were variations of "LIAB-01 could
        not be evaluated ... Not checked." The honesty was right and the severity was
        wrong: "I could not check this" is not the same class of event as "this is
        breached", and putting it at the rule's own severity pushed three non-findings
        to the top of the review queue. A reviewer who sees that twice learns to skim
        the high band, which costs more than the rule buys.

        Reported at `low` instead. The coverage gap it represents is real, but it is a
        property of the corpus rather than of the contract, so it belongs in the run's
        metrics — `examine` counts these separately — rather than at the top of a queue
        ordered by consequence.
        """
        return self.rule.severity if self.checked else "low"


def load_rules(path: Path) -> tuple[str, list[Rule]]:
    """Load and validate a playbook. Every field is checked here rather than at
    evaluation time, so a typo surfaces at startup instead of as a rule that quietly
    never fires."""
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    ruleset_id = document.get("ruleset_id")
    if not ruleset_id:
        raise RuleError(f"{path.name}: missing 'ruleset_id'")

    rules: list[Rule] = []
    seen: set[str] = set()

    for index, entry in enumerate(document.get("rules") or []):
        where = f"{path.name} rule #{index + 1}"

        code = entry.get("code")
        if not code:
            raise RuleError(f"{where}: missing 'code'")
        if code in seen:
            raise RuleError(f"{where}: duplicate code {code!r}")
        seen.add(code)

        kind = entry.get("kind")
        if kind not in RULE_KINDS:
            raise RuleError(
                f"{where} ({code}): kind {kind!r} is not one of {sorted(RULE_KINDS)}. "
                f"A new kind is a code change, not a config change."
            )

        severity = entry.get("severity", "medium")
        if severity not in SEVERITIES:
            raise RuleError(f"{where} ({code}): severity {severity!r} is not valid")

        if not entry.get("predicate"):
            raise RuleError(f"{where} ({code}): missing 'predicate'")

        config = entry.get("config") or {}
        if kind == "numeric_max" and "max" not in config:
            raise RuleError(f"{where} ({code}): numeric_max needs config.max")
        if kind == "numeric_min" and "min" not in config:
            raise RuleError(f"{where} ({code}): numeric_min needs config.min")
        if kind == "numeric_bound" and "expr" not in config:
            raise RuleError(f"{where} ({code}): numeric_bound needs config.expr")

        rules.append(
            Rule(
                code=code,
                kind=kind,
                predicate=entry["predicate"],
                description=entry.get("description", ""),
                config=config,
                severity=severity,
            )
        )

    return ruleset_id, rules


# ---------------------------------------------------------------------------
# Expression evaluation for numeric_bound
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(?:(?P<number>\d+(?:\.\d+)?)|(?P<name>[a-z_]+)|(?P<op>[*+\-]))")


def evaluate_expression(expr: str, values: dict[str, Decimal]) -> Decimal | None:
    """Evaluate a tiny arithmetic expression over named facts.

    Supports numbers, predicate names, `*`, `+`, `-`, left to right. Deliberately not
    `eval()`: a playbook is configuration, and configuration that can execute arbitrary
    Python is a remote code execution hole wearing a friendly name. It is also not a
    general expression parser, because operator precedence nobody asked for is a bug
    generator — `2 * annual_fees` is the shape these rules actually need.

    Returns None when a referenced predicate is unknown, so a rule that cannot be
    evaluated is reported as inapplicable rather than silently passing.
    """
    tokens: list[str] = []
    position = 0
    while position < len(expr):
        match = _TOKEN.match(expr, position)
        if not match:
            return None
        position = match.end()
        tokens.append(match.group(match.lastgroup))

    if not tokens:
        return None

    def resolve(token: str) -> Decimal | None:
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            return Decimal(token)
        return values.get(token)

    total = resolve(tokens[0])
    if total is None:
        return None

    index = 1
    while index + 1 < len(tokens):
        operator, operand = tokens[index], resolve(tokens[index + 1])
        if operand is None:
            return None

        if operator == "*":
            total *= operand
        elif operator == "+":
            total += operand
        elif operator == "-":
            total -= operand
        else:
            return None
        index += 2

    # A dangling operator ("2 *") means the expression is malformed. Returning the
    # partial total would evaluate a rule against a bound its author never wrote.
    return total if index == len(tokens) else None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _magnitude(resolution: Resolution) -> Decimal | None:
    if resolution.governing is None:
        return None
    normalized = normalize(resolution.predicate, resolution.governing.value_raw)
    return normalized.number if normalized else None


def evaluate(rules: list[Rule], resolutions: list[Resolution]) -> list[RuleViolation]:
    """Apply every rule to every vendor's governing values.

    Rules run against the *reconciled* value, not against every fact ever extracted.
    Checking superseded values would report violations that an amendment already fixed
    — technically true, entirely useless, and the fastest way to make a compliance
    report ignorable.
    """
    violations: list[RuleViolation] = []

    by_vendor: dict[str, dict[str, Resolution]] = {}
    for resolution in resolutions:
        # Scoped resolutions (a SOW's own terms) are excluded: the playbook governs the
        # agreement, and flagging a specialist rate against an agreement-level rule
        # would be applying a rule where it was never meant to reach.
        if resolution.scope is None:
            by_vendor.setdefault(resolution.subject, {})[resolution.predicate] = resolution

    for vendor, resolutions_by_predicate in sorted(by_vendor.items()):
        magnitudes = {
            predicate: value
            for predicate, resolution in resolutions_by_predicate.items()
            if (value := _magnitude(resolution)) is not None
        }

        for rule in rules:
            resolution = resolutions_by_predicate.get(rule.predicate)

            if rule.kind == "presence_required":
                if resolution is None or resolution.governing is None:
                    violations.append(
                        RuleViolation(
                            rule=rule,
                            subject=vendor,
                            observed=None,
                            explanation=(
                                f"{rule.description}. No supported value for "
                                f"{rule.predicate} was found for {vendor}."
                            ),
                        )
                    )
                continue

            if resolution is None or resolution.governing is None:
                # A numeric rule with nothing to check is not a violation. Reporting
                # one would mean every vendor fails every rule about a term they
                # simply do not have.
                continue

            observed = magnitudes.get(rule.predicate)
            if observed is None:
                # A value exists but has no magnitude — a liability cap written as a
                # formula, or a string the normalizer refused. This used to `continue`
                # silently, and a silent skip is indistinguishable from a pass: Talus
                # dropped out of three playbook checks between two runs and the register
                # gave no sign of it. Now it is reported, at "not checked" severity, and
                # says which of the two happened.
                raw = resolution.governing.value_raw
                violations.append(
                    RuleViolation(
                        rule=rule,
                        subject=vendor,
                        observed=raw,
                        explanation=(
                            f"{rule.code} could not be evaluated for {vendor}: the "
                            f"recorded {rule.predicate} {raw!r} is "
                            f"{why_not_comparable(rule.predicate, raw)}. Not checked."
                        ),
                        fact_id=resolution.governing.fact_id,
                        checked=False,
                    )
                )
                continue

            limit: Decimal | None
            comparison: str

            if rule.kind == "numeric_max":
                limit, comparison = Decimal(str(rule.config["max"])), "exceeds"
                breached = observed > limit
            elif rule.kind == "numeric_min":
                limit, comparison = Decimal(str(rule.config["min"])), "falls short of"
                breached = observed < limit
            else:  # numeric_bound
                limit = evaluate_expression(str(rule.config["expr"]), magnitudes)
                comparison = "exceeds"
                if limit is None:
                    # The bound could not be computed — usually a missing input fact.
                    # Reported, because a rule that cannot run is a gap in coverage and
                    # a silent skip looks exactly like a pass.
                    violations.append(
                        RuleViolation(
                            rule=rule,
                            subject=vendor,
                            observed=resolution.governing.value_raw,
                            explanation=(
                                f"{rule.code} could not be evaluated for {vendor}: "
                                f"the bound {rule.config['expr']!r} needs a value this "
                                f"corpus does not supply. Not checked."
                            ),
                            fact_id=resolution.governing.fact_id,
                            checked=False,
                        )
                    )
                    continue
                breached = observed > limit

            if breached:
                violations.append(
                    RuleViolation(
                        rule=rule,
                        subject=vendor,
                        observed=resolution.governing.value_raw,
                        explanation=(
                            f"{rule.description}. {vendor} has {rule.predicate} of "
                            f"{resolution.governing.value_raw} ({observed}), which "
                            f"{comparison} the limit of {limit}."
                        ),
                        fact_id=resolution.governing.fact_id,
                    )
                )

    return violations
