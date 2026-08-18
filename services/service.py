"""The service layer.

REST lives on top of this, and the MCP surface will too. Neither contains logic, so
the two cannot drift apart — there is nothing to drift *from*. That is also what makes
behavior 4 honest: the machine path and the human path are the same code, not two
implementations that happen to agree today.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from psycopg_pool import ConnectionPool
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.config import get_settings
from database.db import session_scope
from domain.changes import build_ledger
from models import (
    Chunk,
    Claim,
    ClaimCitation,
    Corpus,
    Decision,
    Document,
    Fact,
    Finding,
    Run,
    SectionVersion,
    StageMetric,
)
from services.graph import build_graph
from utils.logging_config import get_logger, log, run_context, timed
from utils.paths import basename

logger = get_logger(__name__)

_pool: ConnectionPool | None = None
_checkpointer: PostgresSaver | None = None
_lock = threading.Lock()


def _sync_dsn() -> str:
    """LangGraph's checkpointer takes a raw psycopg DSN, not a SQLAlchemy URL."""
    return get_settings().database_url.replace("postgresql+psycopg://", "postgresql://")


def get_checkpointer() -> PostgresSaver:
    """Process-wide checkpointer over a connection pool.

    A pool rather than a single connection because runs park at the human gate for
    arbitrary lengths of time, and concurrent runs must not queue behind each other —
    behavior 9 requires they stay genuinely isolated.

    `autocommit=True` is required: PostgresSaver issues CREATE TABLE during setup()
    and expects its writes to land immediately rather than sit in a transaction the
    caller controls.
    """
    global _pool, _checkpointer
    with _lock:
        if _checkpointer is None:
            _pool = ConnectionPool(
                conninfo=_sync_dsn(),
                min_size=1,
                max_size=10,
                kwargs={"autocommit": True, "prepare_threshold": 0},
                open=True,
            )
            _checkpointer = PostgresSaver(_pool)
            _checkpointer.setup()
    return _checkpointer


def reset_checkpointer() -> None:
    """Tests point at a different database than the process opened."""
    global _pool, _checkpointer
    with _lock:
        if _pool is not None:
            _pool.close()
        _pool = None
        _checkpointer = None


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def ensure_corpus(name: str) -> UUID:
    """Get or create a corpus by name, safely under concurrency.

    Insert-if-absent then read back, rather than check-then-insert. Two runs starting
    together on the same corpus both saw it as absent, both inserted, and one died on
    `corpus_name_key` — the same race as document ingestion, in the very first thing a
    run does. Found by the behavior-9 test, which is the only thing that exercises two
    runs starting at once.
    """
    with session_scope() as session:
        inserted = session.execute(
            pg_insert(Corpus)
            .values(name=name)
            .on_conflict_do_nothing(index_elements=["name"])
            .returning(Corpus.id)
        ).scalar_one_or_none()

        if inserted is not None:
            return inserted

        return session.execute(
            select(Corpus.id).where(Corpus.name == name)
        ).scalar_one()


def start_run(
    *,
    corpus_name: str,
    document_paths: list[str | Path],
    thread_id: str | None = None,
    on_started: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Start a run and drive it until it completes or parks at the gate.

    `thread_id` is the isolation boundary. It defaults to the run id, so two runs
    never share checkpoint state even on the same corpus.
    """
    corpus_id = ensure_corpus(corpus_name)

    with session_scope() as session:
        previous = session.execute(
            select(Run)
            .where(Run.corpus_id == corpus_id, Run.status == "completed")
            .order_by(Run.started_at.desc())
        ).scalars().first()
        prev_run_id = str(previous.id) if previous else None

        run = Run(
            id=uuid4(),
            corpus_id=corpus_id,
            parent_run_id=previous.id if previous else None,
            status="running",
            mode="incremental",  # a full run is the degenerate case, not a mode
        )
        session.add(run)
        session.flush()
        run_id = str(run.id)

    # Announce the id before the graph runs, so a caller that does not want to wait out
    # the whole run still has something to poll. A run takes as long as the model takes;
    # holding an HTTP request open for it is a choice the caller should get to make.
    if on_started is not None:
        on_started(run_id)

    graph = build_graph(get_checkpointer())
    config = {"configurable": {"thread_id": thread_id or run_id}}

    with run_context(run_id):
        log(
            logger,
            logging.INFO,
            "run starting",
            corpus=corpus_name,
            documents=len(document_paths),
            # Logged explicitly because it decides whether this is a full or
            # incremental run, and that is the first thing worth knowing when the
            # section counts later look surprising.
            previous_run=prev_run_id or "none (full run)",
            thread_id=thread_id or run_id,
        )
        with timed(logger, "run", run_id=run_id):
            # `durability="sync"` is not the default, and behavior 2 does not survive
            # the default.
            #
            # LangGraph persists checkpoints asynchronously unless told otherwise: the
            # next node starts while the previous one's checkpoint is still being
            # written. SIGKILL then takes whatever had not landed, and how much that is
            # depends on how fast the machine is — the same kill left CI resumable at
            # `compose` and a developer laptop resumable at `ingest`, having thrown away
            # four stages that had already run and already been metered.
            #
            # "Resumes at the last node boundary" is the floor. A boundary that only
            # usually survives is not one, and the failure is invisible: the run does
            # finish, just by re-doing work it claims not to re-do. Sync costs one
            # round-trip per node on a twelve-node graph.
            state = graph.invoke(
                {
                    "run_id": run_id,
                    "corpus_id": str(corpus_id),
                    "prev_run_id": prev_run_id,
                    "document_paths": [str(p) for p in document_paths],
                },
                config,
                durability="sync",
            )

        result = _describe(run_id, state, config)
        log(logger, logging.INFO, "run returned to caller", status=result["status"])
        return result


def resume_run(
    *,
    run_id: str,
    thread_id: str | None = None,
    decisions: dict[str, str],
    actor: str = "human",
) -> dict[str, Any]:
    """Resume a run parked at the gate with per-item verdicts."""
    graph = build_graph(get_checkpointer())
    config = {"configurable": {"thread_id": thread_id or run_id}}

    with run_context(run_id):
        approved = sum(1 for v in decisions.values() if v == "approved")
        log(
            logger,
            logging.INFO,
            "resuming run with human decisions",
            approved=approved,
            rejected=len(decisions) - approved,
        )
        with timed(logger, "resume", run_id=run_id):
            # Always an envelope, never the bare mapping.
            #
            # `Command(resume={})` does not resume: an empty dict is falsy, LangGraph
            # reads that as "no value supplied", and re-raises the interrupt. The run
            # then re-enters the gate forever — so a reviewer who rejects everything,
            # or approves nothing, hangs the run with no error anywhere. Wrapping the
            # decisions in a dict that always has keys makes the resume value truthy
            # by construction.
            state = graph.invoke(
                Command(resume={"decisions": decisions, "actor": actor}),
                config,
                # Same reason as `start_run`. A reviewer's verdicts are the least
                # reproducible thing in the system — nothing re-derives them — so the
                # node that records them is the last one that should be checkpointed on
                # a best-effort basis.
                durability="sync",
            )
        return _describe(run_id, state, config)


class ResumeFailed(RuntimeError):
    """A resume was attempted and could not complete — most often a source document
    that has moved since the run started. The run stays resumable."""


def known_document_hashes(corpus_name: str) -> dict[str, str]:
    """Filename → content hash for every document already ingested into a corpus.

    The watcher's "what have I seen" lives in memory, so a restart made every file in the
    inbox look new: a poll after restarting reported "36 added" when nothing had been
    added, and started a run for it. The run itself was harmless — documents dedupe on
    hash and facts are reused, so it cost no model calls — but the report was false, and a
    system that says thirty-six documents arrived when none did is the failure mode this
    project keeps finding.

    `prime()` existed for exactly this and read the *filesystem*, which is the wrong
    source and has its own bug: on a genuinely fresh deployment it marks an inbox full of
    unprocessed documents as already seen, and they are never ingested at all. The
    database is the right source, because "have I seen this" is a question about what was
    ingested, not about what is on disk. Both cases then come out correct — after a
    restart nothing is new, and on a fresh deployment everything is.
    """
    with session_scope() as session:
        corpus = session.execute(
            select(Corpus).where(Corpus.name == corpus_name)
        ).scalar_one_or_none()
        if corpus is None:
            return {}
        rows = session.execute(
            select(Document.uri, Document.sha256).where(Document.corpus_id == corpus.id)
        ).all()
    return {basename(uri): digest for uri, digest in rows}


def mark_interrupted_runs() -> int:
    """At startup, no run can still be in flight — so any row claiming otherwise lied.

    Called from the API's startup hook. A run is driven by an in-process graph, so if
    this process is only now booting, nothing is executing the runs the database still
    marks `running`: their process died. Left alone the row says `running` forever, and
    an API reporting work in progress that nothing is progressing is the same class of
    untruth as a false success.

    Safe because the service runs as a single uvicorn process (no `--workers`). Under
    multiple workers this would need a heartbeat instead, since one worker booting
    would say nothing about another worker's live runs.
    """
    with session_scope() as session:
        orphans = session.execute(
            select(Run).where(Run.status == "running")
        ).scalars().all()

        for run in orphans:
            run.status = "interrupted"
            # The stage it died in is kept as `stage_detail` — it is the most useful
            # thing to know about an interrupted run — but `current_stage` is cleared,
            # because nothing is currently executing it.
            if run.current_stage:
                run.stage_detail = f"died during {run.current_stage}"
            run.current_stage = None

        count = len(orphans)

    if count:
        log(
            logger,
            logging.WARNING,
            "runs found still marked running at startup; their process died",
            runs=count,
            note="resumable via POST /runs/{id}/resume",
        )
    return count


def resume_interrupted(run_id: str) -> dict[str, Any]:
    """Continue a run whose process was killed, from its last checkpoint.

    Distinct from `resume_run`, which answers a human gate. This one supplies no value
    — LangGraph re-enters the graph at the node after the last one that checkpointed,
    which is exactly the seam a crash lands on. Nothing completed is recomputed.
    """
    with session_scope() as session:
        run = session.get(Run, UUID(run_id))
        if run is None:
            raise KeyError(run_id)
        if run.status not in ("interrupted", "running"):
            raise ValueError(
                f"run {run_id} is {run.status}; only an interrupted run can be resumed"
            )
        run.status = "running"

    graph = build_graph(get_checkpointer())
    config = {"configurable": {"thread_id": run_id}}

    with run_context(run_id):
        log(logger, logging.INFO, "resuming interrupted run from its last checkpoint")
        try:
            with timed(logger, "resume_interrupted", run_id=run_id):
                # Same reason as `start_run`, and it matters most here: a resumed run is
                # one that has already been killed once, and there is no reason to
                # assume it will not be killed again mid-recovery.
                state = graph.invoke(None, config, durability="sync")
        except Exception as exc:
            # Put the run back where it was. Without this a failed resume leaves the
            # row saying `running` — recreating exactly the ghost this operation exists
            # to clear, and making the second attempt look like a live run.
            #
            # `interrupted` rather than `failed` because the usual cause is a source
            # document that moved: the run is still resumable once it is back, and
            # marking it failed would throw away recoverable work.
            with session_scope() as session:
                run = session.get(Run, UUID(run_id))
                if run is not None:
                    run.status = "interrupted"
            log(
                logger,
                logging.ERROR,
                "resume failed; run left interrupted and still resumable",
                error=type(exc).__name__,
                detail=str(exc)[:300],
            )
            raise ResumeFailed(f"{type(exc).__name__}: {exc}") from exc

        result = _describe(run_id, state, config)
        log(logger, logging.INFO, "interrupted run resumed", status=result["status"])
        return result


def _describe(run_id: str, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    graph = build_graph(get_checkpointer())
    snapshot = graph.get_state(config)
    awaiting = bool(snapshot.next) and "gate" in snapshot.next

    # The escalation gate, surfaced the same way the review gate is.
    #
    # This only read `gate` interrupts, which was survivable while the escalation branch
    # never fired: it had not fired once across four runs and twenty-nine documents. The
    # moment it did, a run parked there reported plain `running` with nothing pending and
    # no question anywhere — the API had a human gate the API could not answer. A branch
    # that becomes reachable has to become answerable in the same change.
    awaiting_classification = bool(snapshot.next) and "escalate" in snapshot.next

    pending: list[dict[str, Any]] = []
    escalations: list[dict[str, Any]] = []
    options: list[str] = []
    option_effects: list[dict[str, Any]] = []
    for task in snapshot.tasks:
        for intr in getattr(task, "interrupts", ()) or ():
            value = getattr(intr, "value", None)
            if not isinstance(value, dict):
                continue
            if value.get("kind") == "classify_documents":
                escalations = value.get("documents", [])
                options = value.get("options", [])
                option_effects = value.get("option_effects", [])
            else:
                pending = value.get("findings", [])

    # A blocked run reports blocked, never "completed". I5: a success message must
    # mean the output is genuinely in the state claimed.
    if state.get("verification_passed") is False:
        status = "blocked_by_verification"
    elif awaiting:
        status = "awaiting_review"
    elif awaiting_classification:
        status = "awaiting_classification"
    else:
        status = state.get("status", "unknown")

    return {
        "run_id": run_id,
        "status": status,
        "awaiting_review": awaiting,
        "awaiting_classification": awaiting_classification,
        "escalations": escalations,
        "classification_options": options,
        "classification_option_effects": option_effects,
        "verification": state.get("verification", {}),
        "pending_findings": pending,
        "plan": state.get("plan_summary", {}),
        "counts": {
            "documents": len(state.get("document_ids", [])),
            "facts": len(state.get("fact_ids", [])),
            "findings": len(state.get("findings", [])),
        },
    }


def get_run(run_id: str) -> dict[str, Any]:
    with session_scope() as session:
        run = session.get(Run, UUID(run_id))
        if run is None:
            raise KeyError(run_id)

        stages = session.execute(
            select(StageMetric).where(StageMetric.run_id == run.id)
        ).scalars().all()

        result = {
            "run_id": run_id,
            "status": run.status,
            "mode": run.mode,
            # Where the run is right now. Stage metrics are only written when a stage
            # *finishes*, so without this a two-minute run reports "running" and one
            # completed line for ninety seconds, and a caller cannot tell work from a
            # hang.
            "current_stage": run.current_stage,
            "stage_detail": run.stage_detail,
            "started_at": run.started_at.isoformat(),
            "ended_at": run.ended_at.isoformat() if run.ended_at else None,
            "stages": [
                {
                    "stage": s.stage,
                    "skipped": s.skipped,
                    "tokens_in": s.tokens_in,
                    "tokens_out": s.tokens_out,
                    "cost_usd": round(s.cost_usd, 6),
                    "cache_hits": s.cache_hits,
                    "cache_misses": s.cache_misses,
                }
                for s in stages
            ],
        }

    # The findings a reviewer needs to see live only in the graph checkpoint while a
    # run sits at the gate — they aren't written to the `finding` table until commit.
    # Without this, the only way to ever see them was the single synchronous response
    # that started or resumed the run; a page reload or a second client asking "what's
    # pending on this run?" got "running" and nothing else. That is the exact request
    # the review UI makes on every load, so the gate was unreachable outside the tab
    # that happened to trigger it.
    result["awaiting_review"] = False
    result["pending_findings"] = []
    result["awaiting_classification"] = False
    result["escalations"] = []
    result["classification_options"] = []
    result["classification_option_effects"] = []

    # Read whenever the run is unfinished, not only when the status column says
    # `awaiting_review`. The escalation gate does not set a status of its own — it pauses
    # mid-pipeline with the row still reading `running` — so gating this read on the
    # status would leave the same blind spot the review gate had: a live question that no
    # page reload could ever find.
    if result["status"] in ("running", "awaiting_review"):
        graph = build_graph(get_checkpointer())
        snapshot = graph.get_state({"configurable": {"thread_id": run_id}})
        nxt = set(snapshot.next or ())
        for task in snapshot.tasks:
            for intr in getattr(task, "interrupts", ()) or ():
                value = getattr(intr, "value", None)
                if not isinstance(value, dict):
                    continue
                if value.get("kind") == "classify_documents" and "escalate" in nxt:
                    result["awaiting_classification"] = True
                    result["status"] = "awaiting_classification"
                    result["escalations"] = value.get("documents", [])
                    result["classification_options"] = value.get("options", [])
                    result["classification_option_effects"] = value.get(
                        "option_effects", []
                    )
                elif "gate" in nxt:
                    result["awaiting_review"] = True
                    result["pending_findings"] = value.get("findings", [])

    return result


def get_deliverable(run_id: str) -> dict[str, Any]:
    """The register for one run, with hashes.

    Hashes are exposed deliberately: they let a reviewer verify the untouched-section
    claim themselves rather than taking our word for it.
    """
    with session_scope() as session:
        versions = session.execute(
            select(SectionVersion)
            .where(SectionVersion.run_id == UUID(run_id))
            .order_by(SectionVersion.section_key)
        ).scalars().all()

        return {
            "run_id": run_id,
            "sections": [
                {
                    "section_key": v.section_key,
                    "content": v.content,
                    "content_hash": v.content_hash,
                    "carried_forward": v.carried_forward,
                }
                for v in versions
            ],
            "carried_forward": sum(1 for v in versions if v.carried_forward),
            "rederived": sum(1 for v in versions if not v.carried_forward),
        }


def get_changes(run_id: str) -> dict[str, Any]:
    """What changed in this run, and because of which document.

    The other half of the incrementality claim: `get_deliverable` shows the hashes,
    this shows the causal chain from an arriving document to the rows it moved.
    """
    with session_scope() as session:
        ledger = build_ledger(session, UUID(run_id))
        return {
            **ledger.summary(),
            "sections": [
                {
                    "section_key": change.section_key,
                    "status": change.status,
                    "content_hash": change.content_hash,
                    "previous_hash": change.previous_hash,
                    "caused_by": change.caused_by,
                }
                for change in ledger.changes
            ],
        }


def list_runs(limit: int = 25) -> list[dict[str, Any]]:
    """Recent runs, newest first — what a reviewer lands on."""
    with session_scope() as session:
        rows = session.execute(
            select(Run, Corpus.name)
            .join(Corpus, Corpus.id == Run.corpus_id)
            .order_by(Run.started_at.desc())
            .limit(limit)
        ).all()

        return [
            {
                "run_id": str(run.id),
                "corpus": corpus_name,
                "status": run.status,
                "mode": run.mode,
                "current_stage": run.current_stage,
                "stage_detail": run.stage_detail,
                "started_at": run.started_at.isoformat(),
                "ended_at": run.ended_at.isoformat() if run.ended_at else None,
            }
            for run, corpus_name in rows
        ]


def get_provenance(run_id: str, section_key: str) -> dict[str, Any]:
    """Where a register row's value actually came from.

    This is I1 made visible. The register asserts "the hourly rate is $195"; this
    returns the document, the exact sentence, and the character span it was read from —
    so a reviewer can check the claim rather than trust it.

    Returning the surrounding chunk text as well as the quote is deliberate: a quote on
    its own is easy to agree with, and the point of provenance is to let someone
    disagree.
    """
    with session_scope() as session:
        version = session.execute(
            select(SectionVersion).where(
                SectionVersion.run_id == UUID(run_id),
                SectionVersion.section_key == section_key,
            )
        ).scalar_one_or_none()

        if version is None:
            raise KeyError(f"{section_key} in run {run_id}")

        claims = session.execute(
            select(Claim).where(Claim.section_version_id == version.id)
        ).scalars().all()

        cited = []
        for claim in claims:
            rows = session.execute(
                select(Fact, Document, Chunk)
                .join(ClaimCitation, ClaimCitation.fact_id == Fact.id)
                .join(Document, Document.id == Fact.document_id)
                .outerjoin(Chunk, Chunk.id == Fact.chunk_id)
                .where(ClaimCitation.claim_id == claim.id)
            ).all()

            for fact, document, chunk in rows:
                cited.append(
                    {
                        "fact_id": str(fact.id),
                        "predicate": fact.predicate,
                        "value": fact.value_raw,
                        "effective_date": (
                            fact.effective_date.isoformat() if fact.effective_date else None
                        ),
                        "document": basename(document.uri),
                        "document_kind": document.kind,
                        "confidence": fact.confidence,
                        # The passage itself. Without it a citation is a filename, and a
                        # filename proves nothing.
                        "passage": chunk.text if chunk else None,
                        "char_start": chunk.char_start if chunk else None,
                        "char_end": chunk.char_end if chunk else None,
                    }
                )

        return {
            "run_id": run_id,
            "section_key": section_key,
            "content": version.content,
            "content_hash": version.content_hash,
            "carried_forward": version.carried_forward,
            "claims": [{"text": c.text, "status": c.status} for c in claims],
            "citations": cited,
        }


def get_decisions(run_id: str) -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.execute(
            select(Decision, Finding)
            .join(Finding, Finding.id == Decision.target_id)
            .where(Decision.run_id == UUID(run_id))
        ).all()
        return [
            {
                "decision_id": str(d.id),
                "verdict": d.verdict,
                "actor": d.actor,
                "severity": f.severity,
                "explanation": f.explanation,
                "decided_at": d.created_at.isoformat(),
            }
            for d, f in rows
        ]
