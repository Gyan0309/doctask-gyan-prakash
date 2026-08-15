"""Phase 0: prove the scaffold is actually wired, not merely present.

Every test here runs with no API key, no network, and no database.
"""

from __future__ import annotations

import json

import pytest

from ledger.config import Settings
from ledger.providers import FakeProvider, build_provider
from ledger.providers.base import ProviderError


def test_provider_is_selected_by_configuration_not_code() -> None:
    """'Configuration over code' has to mean something testable. Switching providers
    is an environment change; no import or edit is involved."""
    assert build_provider(Settings(llm_provider="fake")).name == "fake"
    assert build_provider(
        Settings(llm_provider="gemini", gemini_api_key="not-a-real-key")
    ).name == "gemini"


def test_unknown_provider_fails_loudly_and_names_the_alternatives() -> None:
    with pytest.raises(ProviderError) as exc:
        build_provider(Settings(llm_provider="chatgtp"))
    # An error that only says "unknown provider" makes you go read the source.
    assert "gemini" in str(exc.value) and "fake" in str(exc.value)


def test_gemini_provider_constructs_without_network() -> None:
    """Constructing a provider must never do I/O — otherwise importing the app on a
    machine with no key hangs or crashes instead of starting and explaining itself."""
    provider = build_provider(Settings(llm_provider="gemini", gemini_api_key="k"))
    assert provider.name == "gemini"


def test_missing_key_is_reported_at_construction_not_at_first_call() -> None:
    with pytest.raises(ProviderError) as exc:
        build_provider(Settings(llm_provider="gemini", gemini_api_key=""))
    assert "LLM_PROVIDER=fake" in str(exc.value)  # tells you how to proceed


class TestFakeProvider:
    def test_output_conforms_to_the_requested_schema(self) -> None:
        provider = FakeProvider()
        schema = {
            "type": "OBJECT",
            "properties": {
                "vendor": {"type": "STRING"},
                "rate": {"type": "NUMBER"},
                "conflicted": {"type": "BOOLEAN"},
            },
            "required": ["vendor", "rate", "conflicted"],
        }
        parsed = json.loads(provider.generate("extract terms", schema=schema).text)

        assert set(parsed) == {"vendor", "rate", "conflicted"}
        assert isinstance(parsed["rate"], (int, float))
        assert isinstance(parsed["conflicted"], bool)

    def test_same_input_gives_same_output(self) -> None:
        """Determinism is what allows exact assertions later without pinning fixture
        files that rot."""
        a, b = FakeProvider(), FakeProvider()
        schema = {"type": "OBJECT", "properties": {"x": {"type": "NUMBER"}}}
        assert a.generate("p", schema=schema).text == b.generate("p", schema=schema).text

    def test_every_call_is_recorded(self) -> None:
        """The recording is what makes 'the adjudication stage never ran' checkable.
        Without it, a skipped stage and an executed one look identical from outside."""
        provider = FakeProvider()
        provider.generate("classify this document")
        provider.generate("adjudicate this conflict", cheap=True)

        assert len(provider.calls) == 2
        assert provider.calls[1].cheap is True
        assert len(provider.calls_matching("adjudicate")) == 1
        assert provider.calls_matching("summarize") == []

    def test_can_be_told_to_fail_so_retry_paths_are_reachable(self) -> None:
        """Error branches that cannot be triggered are error branches that were never
        tested. The provider can be armed to fail on demand."""
        provider = FakeProvider(fail_on="poison")
        provider.generate("this one is fine")
        with pytest.raises(ProviderError):
            provider.generate("this one contains poison")


def test_settings_defaults_are_safe_for_a_keyless_clone() -> None:
    settings = Settings(_env_file=None)
    assert settings.classify_confidence_threshold == 0.75
    assert settings.extract_max_retries == 2
    assert "postgresql+psycopg://" in settings.database_url
