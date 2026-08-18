"""Phase 5: the folder watcher.

Every test here is synchronous. `poll_once` is a pure function over a directory and a
state object, so none of this waits on a timer — a watcher that can only be tested by
sleeping is a watcher that is barely tested.
"""

from __future__ import annotations

from pathlib import Path

from services.watcher import Watcher, WatchState, list_documents, poll_once


class _Recorder:
    """Stands in for service.start_run, recording what it was asked to process."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *, corpus_name: str, document_paths: list[str]) -> dict:
        self.calls.append({"corpus_name": corpus_name, "paths": document_paths})
        return {"run_id": f"run-{len(self.calls)}"}


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body, encoding="utf-8")
    return path


class TestListing:
    def test_only_supported_formats_are_listed(self, tmp_path) -> None:
        _write(tmp_path, "a.md", "# contract")
        _write(tmp_path, "notes.docx", "unsupported")
        _write(tmp_path, ".DS_Store", "junk")

        assert set(list_documents(tmp_path)) == {"a.md"}

    def test_a_missing_directory_is_empty_not_an_error(self, tmp_path) -> None:
        """The watched directory may not exist yet on a fresh deployment. That is a
        reason to find nothing, not to crash the service on startup."""
        assert list_documents(tmp_path / "nope") == {}

    def test_files_are_identified_by_content_not_name(self, tmp_path) -> None:
        _write(tmp_path, "a.md", "same text")
        _write(tmp_path, "b.md", "same text")
        found = list_documents(tmp_path)
        assert found["a.md"] == found["b.md"]


class TestChangeDetection:
    def test_a_new_document_triggers_a_run(self, tmp_path) -> None:
        state, runner = WatchState(), _Recorder()
        _write(tmp_path, "msa.md", "The hourly rate is $195 per hour.")

        result = poll_once(tmp_path, state, corpus_name="c", start_run=runner)

        assert result.added == ["msa.md"]
        assert result.triggered is True
        assert len(runner.calls) == 1

    def test_a_second_poll_with_nothing_new_does_not_trigger(self, tmp_path) -> None:
        """The property that makes a 5-second poll affordable. Without it the system
        re-runs forever and exhausts a daily quota in minutes."""
        state, runner = WatchState(), _Recorder()
        _write(tmp_path, "msa.md", "The hourly rate is $195 per hour.")

        poll_once(tmp_path, state, corpus_name="c", start_run=runner)
        second = poll_once(tmp_path, state, corpus_name="c", start_run=runner)

        assert second.triggered is False
        assert len(runner.calls) == 1, "an unchanged directory must cost nothing"

    def test_an_edited_document_triggers_again(self, tmp_path) -> None:
        state, runner = WatchState(), _Recorder()
        _write(tmp_path, "msa.md", "The hourly rate is $195 per hour.")
        poll_once(tmp_path, state, corpus_name="c", start_run=runner)

        _write(tmp_path, "msa.md", "The hourly rate is $210 per hour.")
        result = poll_once(tmp_path, state, corpus_name="c", start_run=runner)

        assert result.modified == ["msa.md"]
        assert len(runner.calls) == 2

    def test_rewriting_a_file_with_identical_content_does_not_trigger(self, tmp_path) -> None:
        """Detection is by content hash, not mtime. Saving a file without changing it
        is not a change, and a touch-triggered run would burn quota for nothing."""
        state, runner = WatchState(), _Recorder()
        body = "The hourly rate is $195 per hour."
        _write(tmp_path, "msa.md", body)
        poll_once(tmp_path, state, corpus_name="c", start_run=runner)

        _write(tmp_path, "msa.md", body)  # same bytes, new mtime
        result = poll_once(tmp_path, state, corpus_name="c", start_run=runner)

        assert result.triggered is False
        assert len(runner.calls) == 1

    def test_a_removed_document_is_reported_but_does_not_trigger(self, tmp_path) -> None:
        """A vanished file does not retract the obligations it established. Dropping
        its facts would silently rewrite history."""
        state, runner = WatchState(), _Recorder()
        _write(tmp_path, "msa.md", "The hourly rate is $195 per hour.")
        poll_once(tmp_path, state, corpus_name="c", start_run=runner)

        (tmp_path / "msa.md").unlink()
        result = poll_once(tmp_path, state, corpus_name="c", start_run=runner)

        assert result.removed == ["msa.md"]
        assert result.triggered is False
        assert len(runner.calls) == 1


class TestWhatTheRunReceives:
    def test_the_whole_corpus_is_passed_not_only_the_new_file(self, tmp_path) -> None:
        """A new amendment changes the meaning of documents already ingested, so a run
        over the delta alone would produce a register that contradicts itself."""
        state, runner = WatchState(), _Recorder()

        _write(tmp_path, "msa.md", "The hourly rate is $180 per hour.")
        poll_once(tmp_path, state, corpus_name="c", start_run=runner)

        _write(tmp_path, "amendment.md", "The hourly rate is $195 per hour.")
        poll_once(tmp_path, state, corpus_name="c", start_run=runner)

        second_call_files = {Path(p).name for p in runner.calls[1]["paths"]}
        assert second_call_files == {"msa.md", "amendment.md"}


class TestPriming:
    """Priming asks "what has been ingested", which only the corpus can answer.

    It used to ask the filesystem, which gets the restart case right and the
    fresh-deployment case wrong in the worst direction: an inbox full of documents that
    were never processed is marked as already seen, and nothing ever processes them.
    """

    def test_documents_already_ingested_do_not_re_run(self, tmp_path) -> None:
        """A restart must not re-process the whole directory as though it just arrived —
        on a per-day quota that spends the budget on completed work, and reporting
        thirty-six new documents when none arrived is false besides."""
        runner = _Recorder()
        _write(tmp_path, "msa.md", "The hourly rate is $195 per hour.")
        _write(tmp_path, "sow.md", "The hourly rate is $210 per hour.")
        ingested = list_documents(tmp_path)

        watcher = Watcher(
            tmp_path, corpus_name="c", start_run=runner, known=lambda _c: ingested
        )
        watcher.prime()

        assert runner.calls == []
        result = watcher.poll()
        assert result.triggered is False
        assert result.added == [], "nothing arrived, so nothing may be reported as added"

    def test_an_inbox_that_was_never_ingested_is_processed(self, tmp_path) -> None:
        """The case filesystem priming got wrong. A fresh deployment whose inbox already
        holds documents must process them, not mark them seen and go quiet."""
        runner = _Recorder()
        _write(tmp_path, "msa.md", "The hourly rate is $195 per hour.")

        watcher = Watcher(
            tmp_path, corpus_name="c", start_run=runner, known=lambda _c: {}
        )
        watcher.prime()

        assert watcher.poll().triggered is True
        assert len(runner.calls) == 1

    def test_an_edited_document_is_seen_as_modified(self, tmp_path) -> None:
        """Priming carries hashes, not just names, so a file changed while the service
        was down is picked up rather than mistaken for one already handled."""
        runner = _Recorder()
        _write(tmp_path, "msa.md", "The hourly rate is $195 per hour.")
        ingested = list_documents(tmp_path)

        watcher = Watcher(
            tmp_path, corpus_name="c", start_run=runner, known=lambda _c: ingested
        )
        watcher.prime()
        _write(tmp_path, "msa.md", "The hourly rate is $265 per hour.")

        result = watcher.poll()

        assert result.triggered is True
        assert result.modified == ["msa.md"]

    def test_with_no_source_of_truth_it_fails_towards_doing_the_work(
        self, tmp_path
    ) -> None:
        """Priming empty re-processes; priming full silently skips. Only one of those is
        recoverable by looking at the output."""
        runner = _Recorder()
        _write(tmp_path, "msa.md", "The hourly rate is $195 per hour.")

        watcher = Watcher(tmp_path, corpus_name="c", start_run=runner)
        watcher.prime()

        assert watcher.poll().triggered is True

    def test_a_document_arriving_after_priming_still_triggers(self, tmp_path) -> None:
        runner = _Recorder()
        _write(tmp_path, "msa.md", "The hourly rate is $195 per hour.")
        ingested = list_documents(tmp_path)

        watcher = Watcher(
            tmp_path, corpus_name="c", start_run=runner, known=lambda _c: ingested
        )
        watcher.prime()
        _write(tmp_path, "amendment.md", "The hourly rate is $210 per hour.")

        assert watcher.poll().triggered is True
        assert len(runner.calls) == 1


class TestResilience:
    def test_a_failing_run_does_not_stop_the_watcher(self, tmp_path) -> None:
        """A watcher that dies on one bad run stops watching forever, and the symptom
        — nothing happening — looks identical to nothing arriving."""

        def _explode(**_kwargs):
            raise RuntimeError("model unavailable")

        watcher = Watcher(tmp_path, corpus_name="c", start_run=_explode)
        watcher.prime()
        _write(tmp_path, "msa.md", "The hourly rate is $195 per hour.")

        # _loop swallows and continues; poll() itself propagates, which is what a
        # caller driving it by hand should see.
        try:
            watcher.poll()
        except RuntimeError:
            pass

        # The failed file was still marked seen, so the next poll is quiet rather than
        # retrying forever against a model that is down.
        assert watcher.poll().triggered is False
