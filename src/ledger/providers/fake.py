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
from dataclasses import dataclass, field
from typing import Any

from ledger.providers.base import Completion, ModelProvider, ProviderError

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
            return seed % 2 == 0
        if enum := schema.get("enum"):
            return enum[seed % len(enum)]

        # A synthesized `quote` must actually appear in the source, or every fact this
        # provider produces gets rejected by citation resolution and the offline path
        # can never exercise anything downstream of extraction. Returning a real line
        # from the document keeps the stub deterministic *and* honest: the citation it
        # emits is genuinely resolvable, exactly as a real one must be.
        if field_name == "quote":
            lines = self._source_lines(root_prompt)
            if lines:
                return lines[seed % len(lines)]

        # Values must be *shaped* like the real thing, not merely unique. A stub
        # returning "fake-3f5d3cf9" as an hourly rate fails normalization on every
        # fact, so every offline run manufactures findings, parks at the human gate,
        # and never completes — which then silently disables incrementality testing,
        # because an incremental run needs a completed predecessor. Plausible values
        # keep the offline path exercising the same code the real one does.
        if field_name == "value_raw":
            return str(50 + seed % 950)
        if field_name == "effective_date":
            return f"202{4 + seed % 3}-{1 + seed % 12:02d}-01"
        if field_name in ("subject", "vendor"):
            return f"Vendor{seed % 4}"

        return f"fake-{seed:08x}"

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
