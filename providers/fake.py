"""Deterministic provider. No key, no network, same answer every time.

This is what lets the entire test suite run on a clean CI runner holding no secrets
(behavior 7). It is deliberately *not* a mock library: it satisfies the real
ModelProvider interface, returns schema-conforming JSON, and records every call.

The brief is pointed about this — "tests that only prove your mocks work do not
count" — so the suite never asserts on what this provider returned. It asserts on
what the *system* did: which nodes ran, which sections changed hash, whether work
survived a kill. This provider exists to make those assertions possible without a
network, not to stand in for the thing under test.

Call recording is the load-bearing feature. "The adjudication stage did not run on a
clean corpus" is only checkable because every call is logged with its stage.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from providers.base import Completion, ModelProvider, ProviderError

# Phrases that mark text aimed at an automated reader. Deliberately literal: this
# stub stands in for a model's judgement, and pretending it has judgement it does not
# have would make offline tests agree with reality only by luck.
INJECTION_MARKERS = (
    "ignore previous",
    "ignore all previous",
    "disregard the above",
    "system prompt",
    "approve all",
    "you are an ai",
    "new instructions",
)


@dataclass
class RecordedCall:
    prompt: str
    schema: dict[str, Any] | None
    cheap: bool
    model: str


@dataclass
class FakeProvider(ModelProvider):
    """`scripted` maps a substring of the prompt to the exact text to return. First
    match wins. Anything unmatched falls through to deterministic synthesis from the
    schema, so a test only scripts the responses it actually cares about."""

    name: str = "fake"
    scripted: dict[str, str] = field(default_factory=dict)
    calls: list[RecordedCall] = field(default_factory=list)
    fail_on: str | None = None  # substring that triggers ProviderError, to test retries

    model: str = "fake-primary"
    model_cheap: str = "fake-cheap"

    # -- deterministic synthesis --------------------------------------------

    @staticmethod
    def _seed(prompt: str) -> int:
        return int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)

    # What a stub should call a document, keyed off its filename. First match wins, so
    # the more specific patterns come first.
    _KIND_HINTS = (
        ("credit-note", "invoice"),
        ("credit_note", "invoice"),
        ("invoice", "invoice"),
        ("application", "invoice"),
        ("amendment", "amendment"),
        ("change-order", "amendment"),
        ("rate-adjustment", "amendment"),
        ("rate-revision", "amendment"),
        ("side-letter", "amendment"),
        ("addendum", "amendment"),
        ("dpa", "amendment"),
        ("renewal", "renewal_notice"),
        ("sow", "sow"),
        ("statement-of-work", "sow"),
        ("order-form", "sow"),
    )

    @classmethod
    def _kind_for(cls, prompt: str) -> str:
        """The document kind a stub should return, derived from the filename.

        Previously this came out of `enum[seed % len(enum)]`, and the seed is a hash of
        the prompt — so **editing the classification prompt silently reshuffled every
        synthetic document's kind**. That is not a hypothetical: adding one field to the
        classify schema turned an MSA fixture into an invoice, and a verification test
        failed on a precondition about supported claims, thirty lines away from anything
        to do with classification. A stub whose answers move when unrelated text moves
        makes every downstream precondition a coin flip.

        Deriving it from the filename is stable under prompt edits, and it makes
        multi-document fixtures behave sensibly: an `msa.md` plus an `invoice.md`
        actually produce a governing value and an observation that can contradict it.

        `msa` is the default because the clean path is what a stub should produce
        unprompted. A test that wants another kind names its file accordingly, which
        also puts the intent in the fixture rather than in a hash.
        """
        match = re.search(r'<untrusted_document name="([^"]*)"', prompt)
        name = (match.group(1) if match else prompt).lower()
        for hint, kind in cls._KIND_HINTS:
            if hint in name:
                return kind
        return "msa"

    @classmethod
    def _predicates_for(cls, prompt: str, enum: list[str]) -> list[str]:
        """The predicates a stub may emit for the document it is pretending to read.

        The stub has to be internally consistent, or it manufactures facts the real
        system is right to refuse. Billing predicates are only legitimate on a document
        that bills, so a stub calling a file `msa.md` and then emitting `invoice_hours`
        for it produced facts that were correctly dropped downstream — leaving a run with
        no facts, no register and no sections, and half a dozen tests failing on
        preconditions about claims.

        Filtered rather than remapped: the choice within the allowed set stays
        seed-driven, so a test asserting on an exact predicate still gets a stable
        answer.
        """
        # Imported here rather than at module scope: the rule belongs to the domain, and
        # a stub is not the place to keep a second copy of it that can drift.
        from domain.extract import predicate_allowed

        kind = cls._kind_for(prompt)
        allowed = [p for p in enum if predicate_allowed(p, kind)]
        return allowed or list(enum)

    @staticmethod
    def _source_lines(prompt: str) -> list[str]:
        """Pull the document text back out of the prompt's untrusted-data envelope."""
        start = prompt.find(">", prompt.find("<untrusted_document"))
        end = prompt.find("</untrusted_document>")
        if start == -1 or end == -1 or end <= start:
            return []
        body = prompt[start + 1 : end]
        return [ln.strip() for ln in body.splitlines() if len(ln.strip()) > 12]

    def _synthesize(
        self, schema: dict[str, Any], prompt: str, field_name: str = "", root_prompt: str = ""
    ) -> Any:
        """Build a value satisfying `schema`. Same prompt and schema always produce
        the same value, so a test can assert on exact output without pinning a
        fixture file."""
        seed = self._seed(prompt)
        kind = str(schema.get("type", "STRING")).upper()
        root_prompt = root_prompt or prompt

        if kind == "OBJECT":
            out = {}
            for key, sub in (schema.get("properties") or {}).items():
                out[key] = self._synthesize(sub, f"{prompt}:{key}", key, root_prompt)
            return out
        if kind == "ARRAY":
            item = schema.get("items", {"type": "STRING"})

            # Injection reporting must reflect the document, not the dice. Returning
            # synthetic entries here made every ordinary contract raise a high-severity
            # "instruction-like text" finding — noise that would bury the one case that
            # matters and make the Phase 6 injection test pass for the wrong reason.
            if field_name == "instruction_like_spans":
                return [
                    line
                    for line in self._source_lines(root_prompt)
                    if any(marker in line.lower() for marker in INJECTION_MARKERS)
                ]

            return [
                self._synthesize(item, f"{prompt}:{i}", field_name, root_prompt)
                for i in range(1 + seed % 2)
            ]
        if kind in ("NUMBER", "INTEGER"):
            # Confidently high, and deterministically so.
            #
            # This spanned 0.70–0.99 for one revision, which straddled the 0.75
            # escalation threshold — so roughly a fifth of synthetic documents routed
            # to the human gate at random, and every test downstream of classification
            # became flaky for reasons having nothing to do with what it tested.
            #
            # The default path a stub produces should be the clean one. A test that
            # wants the escalation branch scripts a low confidence explicitly, which
            # also makes the intent visible in the test rather than emergent.
            if field_name == "confidence":
                return round(0.90 + (seed % 10) / 100, 2)
            return seed % 1000
        if kind == "BOOLEAN":
            # Same reasoning as `confidence` above, and the same trap: `seed % 2` would
            # mark half of all synthetic documents as an ill-fitting kind and route them
            # to the human gate at random, making every test downstream of
            # classification flaky for reasons unrelated to what it tests. A test that
            # wants the escalation branch scripts the poor fit explicitly.
            if field_name == "fits_kind":
                return True
            return seed % 2 == 0
        if enum := schema.get("enum"):
            if field_name == "kind" and "msa" in enum:
                return self._kind_for(root_prompt)
            if field_name == "predicate":
                enum = self._predicates_for(root_prompt, enum)
            return enum[seed % len(enum)]

        # `quote` and `value_raw` must both come from the SAME line of the real source.
        #
        # The quote makes the citation resolvable; the value must then be findable in
        # the chunk that quote resolved to, because Stage C verification re-checks
        # exactly that. Deriving them from independent seeds produced facts whose
        # values appeared nowhere in the document — so every offline run failed its own
        # verification and blocked, which is Stage C working correctly on a stub that
        # was fabricating evidence.
        #
        # Both therefore key off the parent object's prompt rather than their own field
        # name, so they agree on which line they are describing.
        if field_name in ("quote", "value_raw"):
            lines = self._source_lines(root_prompt)
            if lines:
                parent = prompt.rsplit(":", 1)[0]
                line = lines[self._seed(parent) % len(lines)]
                if field_name == "quote":
                    return line
                return self._number_in(line) or line[:40]

        if field_name == "effective_date":
            return f"202{4 + seed % 3}-{1 + seed % 12:02d}-01"
        if field_name in ("subject", "vendor"):
            return f"Vendor{seed % 4}"

        return f"fake-{seed:08x}"

    @staticmethod
    def _number_in(line: str) -> str | None:
        """The first number in a line, kept in the form the document wrote it."""
        import re

        match = re.search(r"\$?\d[\d,]*(?:\.\d+)?", line)
        return match.group(0) if match else None

    # -- interface -----------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        schema: dict[str, Any] | None = None,
        cheap: bool = False,
        temperature: float = 0.0,
    ) -> Completion:
        model = self.model_cheap if cheap else self.model
        self.calls.append(RecordedCall(prompt=prompt, schema=schema, cheap=cheap, model=model))

        if self.fail_on and self.fail_on in prompt:
            raise ProviderError(f"FakeProvider deliberate failure on {self.fail_on!r}")

        for needle, response in self.scripted.items():
            if needle in prompt:
                return Completion(text=response, model=model, tokens_in=len(prompt) // 4)

        text = (
            json.dumps(self._synthesize(schema, prompt))
            if schema is not None
            else f"fake-response-{self._seed(prompt):08x}"
        )
        return Completion(
            text=text,
            model=model,
            tokens_in=len(prompt) // 4,
            tokens_out=len(text) // 4,
        )

    def healthcheck(self) -> dict[str, Any]:
        return {"provider": self.name, "models": {"primary": self.model}, "offline": True}

    # -- test helpers --------------------------------------------------------

    def calls_matching(self, needle: str) -> list[RecordedCall]:
        return [c for c in self.calls if needle in c.prompt]

    def reset(self) -> None:
        self.calls.clear()
