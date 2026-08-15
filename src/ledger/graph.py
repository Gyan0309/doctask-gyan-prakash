"""The run graph.

Phase 1 walks the shortest complete path — ingest, extract, compose, human gate,
commit — with the three things D6 says are rewrites if deferred already wired:
citations, section hashes, and the metered client.

Later phases insert stages between these. They do not restructure them, because the
gate and the commit are the parts everything else has to fit around.

State holds only JSON-serializable values. LangGraph checkpoints state after every
node, so an ORM object in here would either fail to serialize or — worse — round-trip
into a detached instance that raises on first attribute access, hours later, on
resume.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any, TypedDict
from uuid import UUID

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ledger.db import session_scope
from ledger.domain.compose import SectionPlan, apply_plan, plan_recomposition
from ledger.domain.extract import extract_from_chunk
from ledger.domain.ingest import UnsupportedFormat, load_document
from ledger.logging_config import get_logger, log, run_context, stage_context
from ledger.metering import MeteredClient
from ledger.models import Chunk, Decision, Document, Fact, Finding, Run
from ledger.providers import build_provider


def _merge(existing: list, incoming: list) -> list:
    return (existing or []) + (incoming or [])


class RunState(TypedDict, total=False):
    run_id: str
    corpus_id: str
    prev_run_id: str | None
    document_paths: list[str]
    document_ids: Annotated[list[str], _merge]
    fact_ids: Annotated[list[str], _merge]
    findings: Annotated[list[dict[str, Any]], _merge]
    plan_summary: dict[str, int]
    decisions: dict[str, str]
    status: str


logger = get_logger(__name__)

EXTRACTOR_VERSION = "extract-v1"


def _client(session, run_id: str) -> MeteredClient:
    return MeteredClient(build_provider(), session, UUID(run_id))


def node(stage: str):
    """Bind logging context and announce entry/exit for a graph node.

    Applied as a decorator so no node can forget. Every line a node emits — including
    lines from deep inside the provider or the ORM — is then attributable to a
    specific run and stage, which is what makes concurrent runs debuggable at all.
    """

    def decorate(fn):
        def wrapper(state: RunState) -> dict[str, Any]:
            with run_context(state.get("run_id"), stage), stage_context(stage):
                log(logger, logging.INFO, f"{stage}: entering")
                try:
                    result = fn(state)
                except GraphBubbleUp:
                    # interrupt() pauses the graph by raising. That is the human gate
                    # working exactly as designed, so it must not be logged as a
                    # failure — otherwise every run that waits for a reviewer files an
                    # ERROR, and real errors stop being visible in the noise.
                    log(logger, logging.INFO, f"{stage}: paused, awaiting human input")
                    raise
                except Exception as exc:
                    log(
                        logger,
                        logging.ERROR,
                        f"{stage}: failed",
                        error=type(exc).__name__,
                        detail=str(exc)[:300],
                    )
                    raise
                log(
                    logger,
                    logging.INFO,
                    f"{stage}: done",
                    **{
                        k: len(v) if isinstance(v, (list, dict)) else v
                        for k, v in (result or {}).items()
                        if k != "decisions"
                    },
                )
                return result

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorate


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


@node("ingest")
def ingest(state: RunState) -> dict[str, Any]:
    """Read each document, hash it, chunk it with real offsets.

    A file we cannot read becomes a finding and the run continues. One unreadable
    document must not take down a corpus of eleven — and the operator needs to know
    it was skipped, which a silent `continue` would not tell them.
    """
    document_ids: list[str] = []
    findings: list[dict[str, Any]] = []

    with session_scope() as session:
        for raw_path in state["document_paths"]:
            path = Path(raw_path)
            try:
                loaded = load_document(path)
            except (UnsupportedFormat, ValueError) as exc:
                log(
                    logger,
                    logging.WARNING,
                    "document skipped, run continues",
                    document=path.name,
                    reason=str(exc)[:200],
                )
                findings.append(
                    {
                        "severity": "medium",
                        "target_kind": "document",
                        "explanation": f"{path.name} skipped: {exc}",
                    }
                )
                continue

            existing = (
                session.query(Document)
                .filter(
                    Document.corpus_id == UUID(state["corpus_id"]),
                    Document.sha256 == loaded.sha256,
                )
                .one_or_none()
            )
            if existing is not None:
                # Identical content already ingested. Not an error — re-dropping a
                # file is a normal thing for a human to do, and it must be a no-op.
                log(
                    logger,
                    logging.INFO,
                    "document already ingested, reusing",
                    document=path.name,
                    sha256=loaded.sha256[:12],
                )
                document_ids.append(str(existing.id))
                continue

            document = Document(
                corpus_id=UUID(state["corpus_id"]),
                uri=loaded.uri,
                sha256=loaded.sha256,
                mime=loaded.mime,
            )
            session.add(document)
            session.flush()

            for raw in loaded.chunks:
                session.add(
                    Chunk(
                        document_id=document.id,
                        ordinal=raw.ordinal,
                        text=raw.text,
                        char_start=raw.char_start,
                        char_end=raw.char_end,
                        page=raw.page,
                    )
                )
            session.flush()
            document_ids.append(str(document.id))

    return {"document_ids": document_ids, "findings": findings}


@node("extract")
def extract(state: RunState) -> dict[str, Any]:
    """Extract typed facts, keeping only those with resolvable citations."""
    fact_ids: list[str] = []
    findings: list[dict[str, Any]] = []

    with session_scope() as session:
        client = _client(session, state["run_id"])

        for document_id in state.get("document_ids", []):
            document = session.get(Document, UUID(document_id))
            if document is None:
                continue

            # A document's identity is its content hash, and extraction for a given
            # extractor_version is deterministic. So facts already extracted from this
            # exact document are still valid, and re-extracting would do three bad
            # things: pay for the model again, and — because each pass minted fresh
            # fact UUIDs — make every dependent section look changed, which silently
            # destroys incrementality. Reuse is not an optimisation here; it is what
            # makes "an update costs like an update" true at all.
            already = (
                session.query(Fact)
                .filter(
                    Fact.document_id == document.id,
                    Fact.extractor_version == EXTRACTOR_VERSION,
                )
                .all()
            )
            if already:
                log(
                    logger,
                    logging.INFO,
                    "facts already extracted for this document; model not called",
                    document=Path(document.uri).name,
                    facts=len(already),
                )
                fact_ids.extend(str(f.id) for f in already)
                continue

            chunks = (
                session.query(Chunk)
                .filter(Chunk.document_id == document.id)
                .order_by(Chunk.ordinal)
                .all()
            )
            name = Path(document.uri).name

            for chunk in chunks:
                from ledger.domain.ingest import RawChunk

                raw = RawChunk(
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                )
                result = extract_from_chunk(raw, name, client)

                for fact in result.facts:
                    row = Fact(
                        document_id=document.id,
                        chunk_id=chunk.id,
                        predicate=fact.predicate,
                        subject=fact.subject,
                        value_raw=fact.value_raw,
                        value_norm=fact.value_raw,
                        unit=fact.unit,
                        confidence=fact.confidence,
                        extractor_version=EXTRACTOR_VERSION,
                    )
                    session.add(row)
                    session.flush()
                    fact_ids.append(str(row.id))

                for rejection in result.rejections:
                    findings.append(
                        {
                            "severity": "low",
                            "target_kind": "fact",
                            "explanation": (
                                f"{name}: dropped {rejection.predicate} "
                                f"({rejection.value_raw!r}) — {rejection.reason}"
                            ),
                        }
                    )

                if result.rejections:
                    log(
                        logger,
                        logging.INFO,
                        "facts dropped for unresolvable citations",
                        document=name,
                        kept=len(result.facts),
                        dropped=len(result.rejections),
                    )

                for span in result.instruction_like_spans:
                    # I3: the attack is converted into a reportable observation.
                    log(
                        logger,
                        logging.WARNING,
                        "instruction-like text found in a source document; reported, not followed",
                        document=name,
                        span=span[:120],
                    )
                    findings.append(
                        {
                            "severity": "high",
                            "target_kind": "document",
                            "explanation": (
                                f"{name} contains text addressed at an automated "
                                f"system, which was reported and not acted on: {span[:200]!r}"
                            ),
                        }
                    )

        client.record_stage("extract")

    return {"fact_ids": fact_ids, "findings": findings}


@node("compose")
def compose(state: RunState) -> dict[str, Any]:
    """Build the register, re-deriving only sections whose dependencies changed."""
    with session_scope() as session:
        facts = (
            session.query(Fact)
            .filter(Fact.id.in_([UUID(f) for f in state.get("fact_ids", [])]))
            .all()
            if state.get("fact_ids")
            else []
        )

        grouped: dict[tuple[str, str], list[Fact]] = {}
        for fact in facts:
            grouped.setdefault((fact.subject, fact.predicate), []).append(fact)

        desired = [
            SectionPlan(
                section_key=f"{subject}::{predicate}",
                kind="register_row",
                payload={
                    "vendor": subject,
                    "term": predicate,
                    "value": sorted(f.value_raw for f in rows)[0],
                    "sources": sorted(str(f.document_id) for f in rows),
                },
                fact_ids=frozenset(f.id for f in rows),
            )
            for (subject, predicate), rows in sorted(grouped.items())
        ]

        prev = UUID(state["prev_run_id"]) if state.get("prev_run_id") else None
        plan = plan_recomposition(session, desired, prev_run_id=prev)
        apply_plan(session, plan, run_id=UUID(state["run_id"]), prev_run_id=prev)

        summary = plan.summary()

        # The line that makes the incremental claim auditable from a log tail: how
        # many sections this run actually re-derived versus carried forward untouched.
        log(
            logger,
            logging.INFO,
            "recomposition planned",
            full_run=prev is None,
            **summary,
        )

    return {"plan_summary": summary}


@node("gate")
def gate(state: RunState) -> dict[str, Any]:
    """The human gate.

    interrupt() halts the graph and persists it. The run can sit here indefinitely —
    across a restart, a crash, or a weekend — and resume on Command(resume=...).

    A run with nothing to review skips the gate entirely rather than presenting an
    empty form. Stopping a human to approve nothing trains them to click through
    without reading, which quietly destroys the value of having a gate at all.
    """
    findings = state.get("findings", [])
    if not findings:
        log(logger, logging.INFO, "nothing to review; gate skipped rather than shown empty")
        return {"status": "no_review_required"}

    log(logger, logging.INFO, "presenting findings for human review", pending=len(findings))

    decisions = interrupt(
        {
            "kind": "review_findings",
            "run_id": state["run_id"],
            "findings": [{"index": i, **f} for i, f in enumerate(findings)],
        }
    )
    return {"decisions": decisions or {}, "status": "reviewed"}


@node("commit")
def commit(state: RunState) -> dict[str, Any]:
    """Persist findings with their verdicts, and close the run.

    Rejected findings are recorded as rejected, never deleted. A gate whose rejections
    vanish cannot answer "why is this not in the register?" six weeks later.
    """
    decisions = state.get("decisions", {}) or {}

    with session_scope() as session:
        run = session.get(Run, UUID(state["run_id"]))

        for index, payload in enumerate(state.get("findings", [])):
            verdict = decisions.get(str(index), "approved")
            finding = Finding(
                run_id=UUID(state["run_id"]),
                severity=payload.get("severity", "low"),
                status="approved" if verdict == "approved" else "rejected",
                target_kind=payload.get("target_kind", "document"),
                explanation=payload.get("explanation", ""),
            )
            session.add(finding)
            session.flush()

            session.add(
                Decision(
                    run_id=UUID(state["run_id"]),
                    kind="finding",
                    target_id=finding.id,
                    verdict="approved" if verdict == "approved" else "rejected",
                    actor=decisions.get("_actor", "human"),
                )
            )

        if run is not None:
            run.status = "completed"

        approved = sum(1 for v in decisions.values() if v == "approved")
        log(
            logger,
            logging.INFO,
            "run committed",
            findings=len(state.get("findings", [])),
            approved=approved,
            rejected=len(decisions) - approved if decisions else 0,
        )

    return {"status": "completed"}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    builder = StateGraph(RunState)

    builder.add_node("ingest", ingest)
    builder.add_node("extract", extract)
    builder.add_node("compose", compose)
    builder.add_node("gate", gate)
    builder.add_node("commit", commit)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "extract")
    builder.add_edge("extract", "compose")
    builder.add_edge("compose", "gate")
    builder.add_edge("gate", "commit")
    builder.add_edge("commit", END)

    return builder.compile(checkpointer=checkpointer)
