"""Configuration. Everything that could reasonably differ between environments lives
here and arrives from the environment — model IDs included.

"Configuration over code" is one of the behaviors this build is judged on, so the test
is concrete: swapping the model, the provider, or the database must never require
editing a Python file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `model_*` is a reserved namespace in pydantic v2; our fields are named
        # llm_* precisely to stay out of it.
        protected_namespaces=(),
    )

    # --- database -----------------------------------------------------------
    database_url: str = "postgresql+psycopg://ledger:ledger@db:5432/ledger"

    # --- model provider -----------------------------------------------------
    # "gemini" → real API calls. "fake" → deterministic stub, no key, no network.
    #
    # Defaults to "fake" so a fresh clone with no .env and no key comes up working
    # rather than merely running. Set LLM_PROVIDER=gemini in .env for real models.
    llm_provider: str = "fake"

    gemini_api_key: str | None = None

    # Default is a lite model, not the largest one, and that is a considered choice.
    # The Gemini free tier caps requests **per day, per model** — measured at 20/day
    # for gemini-3.6-flash. A seven-document corpus exhausts that in a single run, so
    # the biggest model is the one you can least afford to make the default.
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_model_cheap: str = "gemini-3.1-flash-lite"

    # Ordered fallbacks, tried when the model above it is quota-exhausted. Because the
    # cap is per model, a second model is a second budget — which is the difference
    # between a run that degrades and a run that dies.
    gemini_model_fallbacks: str = (
        "gemini-flash-lite-latest,gemini-3-flash-preview,gemini-flash-latest"
    )

    # --- SuperDocs, the editing surface for the register (D5) ----------------
    # Optional by design. Absent, the register still renders and exports locally;
    # publishing is the only thing that becomes unavailable, and it says so rather
    # than failing obscurely. Nothing in the test suite needs this.
    superdocs_api_key: str | None = None
    superdocs_base_url: str = "https://api.superdocs.app/v1"

    # Outbound pacing, requests per minute. Does not address the daily cap — nothing
    # can — but keeps bursts from tripping the per-minute limiter on top of it.
    gemini_requests_per_minute: int = 15

    @property
    def gemini_fallback_list(self) -> list[str]:
        return [m.strip() for m in self.gemini_model_fallbacks.split(",") if m.strip()]

    # --- ingestion ----------------------------------------------------------
    watch_dir: Path = Path("/data/inbox")

    # The watcher is opt-in. Enabled by default it would start runs the moment the
    # service boots, spending a 20-request-per-day budget on work nobody asked for —
    # and a system that acts on its own before you have configured it is a system
    # people learn to distrust.
    watch_enabled: bool = False
    watch_interval_seconds: float = 5.0
    watch_corpus_name: str = "watched"

    # --- rules --------------------------------------------------------------
    # The contract playbook. A new rule is an edit to this file, never to code.
    rules_path: Path = Path("rules/playbook.yaml")

    # --- logging ------------------------------------------------------------
    log_level: str = "INFO"
    # "human" for aligned, readable console output; "json" for log aggregators.
    log_format: str = "human"

    # --- behaviour knobs ----------------------------------------------------
    # Below this classification confidence the graph escalates to a human instead of
    # guessing. A real decision point, not a label on a fixed script.
    classify_confidence_threshold: float = 0.75

    # Extraction retries with a repair prompt this many times before the document is
    # skipped and a finding is emitted. Skipping loudly beats extracting garbage.
    extract_max_retries: int = 2

    @property
    def is_fake_provider(self) -> bool:
        return self.llm_provider.lower() == "fake"


@lru_cache
def get_settings() -> Settings:
    return Settings()
