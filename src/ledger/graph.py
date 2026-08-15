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

from ledger.config import get_settings
from ledger.db import session_scope
from ledger.domain.classify import classify_document
from ledger.domain.compose import SectionPlan, apply_plan, plan_recomposition
from ledger.domain.extract import extract_from_document
from ledger.domain.ingest import RawChunk, UnsupportedFormat, load_document
from ledger.domain.normalize import normalize, normalize_date
from ledger.domain.reconcile import FactView, reconcile
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
    # Documents whose classification was too uncertain to act on. Non-empty routes the
    # graph to the escalation node instead of straight to extraction.
    escalations: Annotated[list[dict[str, Any]], _merge]
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


@node("classify")
def classify(state: RunState) -> dict[str, Any]:
    """Identify what each document is, and how sure we are.

    The confidence is not decoration. Below the configured threshold the document is
    escalated to a human rather than processed on a guess — because document kind sets
    precedence, and a mis-ranked document produces a register that is confidently
    wrong. Confidently wrong is the worst outcome available to this system.
    """
    settings = get_settings()
    escalations: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []

    with session_scope() as session:
        client = _client(session, state["run_id"])

        for document_id in state.get("document_ids", []):
            document = session.get(Document, UUID(document_id))
            if document is None or document.kind is not None:
                # Already classified in an earlier run. Same reasoning as fact reuse:
                # the document's identity is its content hash, so the answer cannot
                # have changed, and re-asking would pay for it again.
                continue

            name = Path(document.uri).name
            text = "\n\n".join(
                c.text
                for c in session.query(Chunk)
                .filter(Chunk.document_id == document.id)
                .order_by(Chunk.ordinal)
                .all()
            )

            result = classify_document(text, name, client)
            document.kind = result.kind
            document.kind_confidence = result.confidence
            document.vendor = result.vendor or None

            if result.confidence < settings.classify_confidence_threshold:
                log(
                    logger,
                    logging.WARNING,
                    "classification below threshold, escalating to a human",
                    document=name,
                    kind=result.kind,
                    confidence=result.confidence,
                    threshold=settings.classify_confidence_threshold,
                )
                escalations.append(
                    {
                        "document_id": str(document.id),
                        "document": name,
                        "proposed_kind": result.kind,
                        "confidence": result.confidence,
                        "reasoning": result.reasoning,
                    }
                )
                findings.append(
                    {
                        "severity": "medium",
                        "target_kind": "document",
                        "explanation": (
                            f"{name}: classified as {result.kind!r} with confidence "
                            f"{result.confidence:.2f}, below the "
                            f"{settings.classify_confidence_threshold} threshold. "
                            f"Model's reasoning: {result.reasoning}"
                        ),
                    }
                )
            else:
                log(
                    logger,
                    logging.INFO,
                    "document classified",
                    document=name,
                    kind=result.kind,
                    vendor=result.vendor,
                    confidence=result.confidence,
                )

        client.record_stage("classify")

    return {"escalations": escalations, "findings": findings}


def route_after_classify(state: RunState) -> str:
    """A real branch: uncertain documents take a different path through the graph.

    Returns a node name, so the routing is visible in the compiled graph rather than
    hidden inside a node as an if-statement.
    """
    return "escalate" if state.get("escalations") else "extract"


@node("escalate")
def escalate(state: RunState) -> dict[str, Any]:
    """Ask a human what an unrecognised document is.

    A separate node rather than a flag on the gate, because this question arrives
    *before* extraction and its answer changes what gets extracted. Folding it into
    the final review would mean extracting on a guess and asking afterwards, which is
    the wrong order.
    """
    escalations = state.get("escalations", [])
    log(logger, logging.INFO, "awaiting human classification", documents=len(escalations))

    answers = interrupt(
        {
            "kind": "classify_documents",
            "run_id": state["run_id"],
            "documents": escalations,
            "options": ["msa", "amendment", "sow", "invoice", "renewal_notice"],
        }
    )

    applied = 0
    with session_scope() as session:
        for document_id, chosen_kind in (answers or {}).items():
            document = session.get(Document, UUID(document_id))
            if document is None or not chosen_kind:
                continue
            document.kind = chosen_kind
            # A human answer is definitive; the threshold no longer applies to it.
            document.kind_confidence = 1.0
            applied += 1

    log(logger, logging.INFO, "human classifications applied", documents=applied)
    return {"status": "classified_by_human"}


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

            raw_chunks = [
                RawChunk(
                    ordinal=c.ordinal,
                    text=c.text,
                    char_start=c.char_start,
                    char_end=c.char_end,
                )
                for c in chunks
            ]
            chunk_by_ordinal = {c.ordinal: c for c in chunks}

            # One call per document, not per chunk. A term stated across a paragraph
            # break is invisible to a model shown only one side of it.
            result = extract_from_document(raw_chunks, name, client)

            for fact in result.facts:
                # Normalize here, deterministically. `value_raw` keeps what the
                # document literally said so a conflict can quote it verbatim;
                # `value_norm` is what comparisons run on. Keeping both is what
                # lets the system prove a contradiction *and* show its source.
                normalized = normalize(fact.predicate, fact.value_raw)
                source_chunk = chunk_by_ordinal.get(fact.chunk_ordinal)

                row = Fact(
                    document_id=document.id,
                    chunk_id=source_chunk.id if source_chunk else None,
                    predicate=fact.predicate,
                    subject=fact.subject,
                    value_raw=fact.value_raw,
                    value_norm=normalized.as_text() if normalized else None,
                    unit=normalized.unit if normalized else fact.unit,
                    effective_date=normalize_date(fact.effective_date),
                    confidence=fact.confidence,
                    extractor_version=EXTRACTOR_VERSION,
                )
                session.add(row)
                session.flush()
                fact_ids.append(str(row.id))

                if normalized is None and fact.predicate != "governing_law":
                    # Reported, not silently tolerated: an unnormalized numeric
                    # value cannot participate in conflict detection, so it is a
                    # gap in coverage that a human should see rather than a
                    # cosmetic issue.
                    findings.append(
                        {
                            "severity": "low",
                            "target_kind": "fact",
                            "explanation": (
                                f"{name}: {fact.predicate} value "
                                f"{fact.value_raw!r} could not be normalized, so "
                                f"it cannot be compared against other documents."
                            ),
                        }
                    )

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

            # Reported unconditionally. This sat nested under the rejection branch for
            # one revision, which meant a document whose citations all resolved could
            # smuggle instruction-like text past reporting entirely — the injection
            # defence silently disabled by an unrelated success.
            for span in result.instruction_like_spans:
                # I3: the attack is converted into a reportable observation.
                log(
                    logger,
                    logging.WARNING,
                    "instruction-like text in a source document; reported, not followed",
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

        # Reconcile before composing. Without this the register would list every
        # value any document ever stated for a term, which is a pile of paper, not a
        # register. Reconciliation picks the one that governs and keeps the rest as
        # evidence.
        documents = {
            d.id: d
            for d in session.query(Document)
            .filter(Document.corpus_id == UUID(state["corpus_id"]))
            .all()
        }

        views = []
        for f in facts:
            document = documents.get(f.document_id)
            kind = (document.kind if document else None) or "unknown"

            views.append(
                FactView(
                    fact_id=f.id,
                    predicate=f.predicate,
                    # Vendor comes from the document, not the fact. See the comment on
                    # Document.vendor — per-fact subjects drift and fragment the
                    # register, which stops amendments from superseding the terms they
                    # amend.
                    subject=(
                        document.vendor if document and document.vendor else f.subject
                    ),
                    value_raw=f.value_raw,
                    value_norm=None,
                    unit=f.unit,
                    effective_date=f.effective_date,
                    document_id=f.document_id,
                    document_kind=kind,
                    # A Statement of Work binds its own engagement. Scoping by document
                    # keeps a specialist rate from competing with the standard rate —
                    # without it a $210 SOW rate appears to supersede the $195
                    # agreement rate, manufacturing a contradiction that is not real.
                    scope=str(f.document_id) if kind == "sow" else None,
                )
            )

        desired = []
        for resolution in reconcile(views):
            governing = resolution.governing
            contributing = (
                [governing] if governing else []
            ) + resolution.superseded + resolution.observations

            desired.append(
                SectionPlan(
                    section_key=resolution.key(),
                    kind="register_row",
                    payload={
                        "vendor": resolution.subject,
                        "term": resolution.predicate,
                        "value": governing.value_raw if governing else None,
                        "normalized": governing.unit if governing else None,
                        "effective_date": (
                            governing.effective_date.isoformat()
                            if governing and governing.effective_date
                            else None
                        ),
                        "governing_source": (
                            governing.document_kind if governing else None
                        ),
                        "status": resolution.status,
                        # Superseded values stay visible. They are what a late invoice
                        # contradicts, and hiding them would hide the finding.
                        "superseded": [f.value_raw for f in resolution.superseded],
                        "observed": [f.value_raw for f in resolution.observations],
                    },
                    fact_ids=frozenset(f.fact_id for f in contributing),
                )
            )

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
    builder.add_node("classify", classify)
    builder.add_node("escalate", escalate)
    builder.add_node("extract", extract)
    builder.add_node("compose", compose)
    builder.add_node("gate", gate)
    builder.add_node("commit", commit)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "classify")

    # Decision point 1: uncertain classification takes a different path. Declared as a
    # conditional edge so it is part of the graph's shape, not an if-statement buried
    # in a node where nothing can observe it.
    builder.add_conditional_edges(
        "classify",
        route_after_classify,
        {"escalate": "escalate", "extract": "extract"},
    )
    builder.add_edge("escalate", "extract")
    builder.add_edge("extract", "compose")
    builder.add_edge("compose", "gate")
    builder.add_edge("gate", "commit")
    builder.add_edge("commit", END)

    return builder.compile(checkpointer=checkpointer)
