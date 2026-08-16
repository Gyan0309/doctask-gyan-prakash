"""The folder watcher: documents arrive, and the system wakes up.

Structured so the interesting part is testable without waiting for anything.
`poll_once()` is a pure, synchronous function that reports what it found and optionally
starts a run; the background loop is a thin `while True` around it. Tests call
`poll_once()` directly and finish in milliseconds — a watcher that can only be tested
by sleeping is a watcher that is barely tested.

The location sits behind this module rather than being assumed. Today it is a local
directory, which is legible on camera: dropping a PDF into a folder needs no
explanation. Object storage would be a different `list_documents()` and nothing else.

Detection is by **content hash, not mtime**. A file touched but unchanged must not
trigger a run, and a file changed in place without its timestamp moving must. Ingestion
already dedupes on hash, so a spurious trigger is harmless — but it would still burn
model quota and fill the run history with noise, and on a 20-requests-per-day budget
that is not a small thing.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from domain.ingest import SUPPORTED_SUFFIXES
from utils.hashing import bytes_hash
from utils.logging_config import get_logger, log

logger = get_logger(__name__)


@dataclass
class WatchState:
    """What the watcher has already seen, keyed by filename."""

    seen: dict[str, str] = field(default_factory=dict)

    def diff(self, current: dict[str, str]) -> tuple[list[str], list[str], list[str]]:
        """Return (added, modified, removed) filenames."""
        added = sorted(set(current) - set(self.seen))
        removed = sorted(set(self.seen) - set(current))
        modified = sorted(
            name
            for name, digest in current.items()
            if name in self.seen and self.seen[name] != digest
        )
        return added, modified, removed


def list_documents(directory: Path) -> dict[str, str]:
    """Filename → content hash, for every supported document in the directory.

    Unsupported files are ignored silently *here* and reported by ingestion if they are
    ever explicitly submitted. The watcher is not the right place to complain about a
    stray `.DS_Store`.
    """
    if not directory.is_dir():
        return {}

    found: dict[str, str] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        try:
            found[path.name] = bytes_hash(path.read_bytes())
        except OSError as exc:
            # A file mid-copy is common and transient: it will be picked up on the next
            # poll once the write completes. Logged at DEBUG so a large import does not
            # fill the log with alarming lines about files that are simply still arriving.
            log(
                logger,
                logging.DEBUG,
                "skipping unreadable file",
                file=path.name,
                error=type(exc).__name__,
            )
    return found


@dataclass
class PollResult:
    added: list[str]
    modified: list[str]
    removed: list[str]
    run_id: str | None = None

    @property
    def triggered(self) -> bool:
        return bool(self.added or self.modified)


def poll_once(
    directory: Path,
    state: WatchState,
    *,
    corpus_name: str,
    start_run=None,
) -> PollResult:
    """Look once, and start a run if anything arrived or changed.

    The run is given the **whole** directory, not only the new file. That is not
    laziness: a new amendment changes the meaning of documents already ingested, so a
    run over the delta alone would produce a register that contradicts itself. Passing
    everything costs nothing, because ingestion skips documents it has already seen by
    hash and extraction reuses their facts — the incremental saving comes from the
    dependency map, not from withholding inputs.
    """
    current = list_documents(directory)
    added, modified, removed = state.diff(current)

    if removed:
        # Reported, not acted on. A vanished file does not retract the obligations it
        # established, and quietly dropping its facts would silently rewrite history.
        log(
            logger,
            logging.WARNING,
            "documents disappeared from the watched directory; their facts are retained",
            files=", ".join(removed),
        )

    state.seen = current

    if not (added or modified):
        return PollResult(added=added, modified=modified, removed=removed)

    log(
        logger,
        logging.INFO,
        "change detected in watched directory",
        added=", ".join(added) or "-",
        modified=", ".join(modified) or "-",
        corpus=corpus_name,
    )

    if start_run is None:
        return PollResult(added=added, modified=modified, removed=removed)

    result = start_run(
        corpus_name=corpus_name,
        document_paths=[str(directory / name) for name in sorted(current)],
    )
    return PollResult(
        added=added, modified=modified, removed=removed, run_id=result.get("run_id")
    )


class Watcher:
    """Background polling loop. A thin wrapper; the logic lives in `poll_once`."""

    def __init__(
        self,
        directory: Path,
        *,
        corpus_name: str,
        interval_seconds: float = 5.0,
        start_run=None,
    ) -> None:
        self.directory = directory
        self.corpus_name = corpus_name
        self.interval = interval_seconds
        self._start_run = start_run
        self._state = WatchState()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def prime(self) -> None:
        """Record what is already there without triggering a run.

        Called at startup so restarting the service does not re-process the entire
        directory as though it had just arrived — which, on a free tier measured in
        requests per day, would exhaust the budget on work already done.
        """
        self._state.seen = list_documents(self.directory)
        log(
            logger,
            logging.INFO,
            "watcher primed with existing documents",
            directory=str(self.directory),
            documents=len(self._state.seen),
        )

    def poll(self) -> PollResult:
        return poll_once(
            self.directory,
            self._state,
            corpus_name=self.corpus_name,
            start_run=self._start_run,
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception as exc:
                # A watcher that dies on one bad run stops watching forever, and the
                # symptom — nothing happening — looks identical to nothing arriving.
                log(
                    logger,
                    logging.ERROR,
                    "poll failed; watcher continues",
                    error=type(exc).__name__,
                    detail=str(exc)[:300],
                )
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self.prime()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ledger-watcher")
        self._thread.start()
        log(
            logger,
            logging.INFO,
            "watcher started",
            directory=str(self.directory),
            interval_s=self.interval,
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
