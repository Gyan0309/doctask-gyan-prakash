"""The service layer.

REST lives on top of this, and the MCP surface will too. Neither contains logic, so
the two cannot drift apart — there is nothing to drift *from*. That is also what makes
behavior 4 honest: the machine path and the human path are the same code, not two
implementations that happen to agree today.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from psycopg_pool import ConnectionPool
from sqlalchemy import select

from ledger.config import get_settings
from ledger.db import session_scope
from ledger.domain.changes import build_ledger
from ledger.graph import build_graph
from ledger.logging_config import get_logger, log, run_context, timed
from ledger.models import Corpus, Decision, Finding, Run, SectionVersion, StageMetric

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
    with session_scope() as session:
        existing = session.execute(
            select(Corpus).where(Corpus.name == name)
        ).scalar_one_or_none()
        if existing is not None:
            return existing.id
        corpus = Corpus(name=name)
        session.add(corpus)
        session.flush()
        return corpus.id


def start_run(
    *,
    corpus_name: str,
    document_paths: list[str | Path],
    thread_id: str | None = None,
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
            state = graph.invoke(
                {
                    "run_id": run_id,
                    "corpus_id": str(corpus_id),
                    "prev_run_id": prev_run_id,
                    "document_paths": [str(p) for p in document_paths],
                },
                config,
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
                Command(resume={"decisions": decisions, "actor": actor}), config
            )
        return _describe(run_id, state, config)


def _describe(run_id: str, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    graph = build_graph(get_checkpointer())
    snapshot = graph.get_state(config)
    awaiting = bool(snapshot.next) and "gate" in snapshot.next

    pending: list[dict[str, Any]] = []
    if awaiting:
        for task in snapshot.tasks:
            for intr in getattr(task, "interrupts", ()) or ():
                value = getattr(intr, "value", None)
                if isinstance(value, dict):
                    pending = value.get("findings", [])

    # A blocked run reports blocked, never "completed". I5: a success message must
    # mean the output is genuinely in the state claimed.
    if state.get("verification_passed") is False:
        status = "blocked_by_verification"
    elif awaiting:
        status = "awaiting_review"
    else:
        status = state.get("status", "unknown")

    return {
        "run_id": run_id,
        "status": status,
        "awaiting_review": awaiting,
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

        return {
            "run_id": run_id,
            "status": run.status,
            "mode": run.mode,
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
