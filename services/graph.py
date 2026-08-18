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
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict
from uuid import UUID

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.config import get_settings
from database.db import session_scope
from domain.adjudicate import adjudicate
from domain.classify import (
    DOCUMENT_KINDS,
    accepted_kind,
    classify_document,
    escalation_reason,
    names_our_organisation,
)
from domain.compose import SectionPlan, apply_plan, plan_recomposition
from domain.conflicts import ConflictCandidate, detect
from domain.direction import Comparison
from domain.extract import (
    KIND_ONLY_PREDICATES,
    ExtractionFailed,
    extract_from_document,
    predicate_allowed,
)
from domain.ingest import RawChunk, UnsupportedFormat, load_document
from domain.normalize import normalize, normalize_date
from domain.reconcile import (
    OBSERVATIONAL_KINDS,
    FactView,
    canonical_subject,
    reconcile,
)
from domain.rules import RuleError, evaluate, load_rules
from domain.verify import verify_run
from models import Chunk, Conflict, Decision, Document, Fact, Finding, Run, SectionVersion
from providers import build_provider
from services.metering import MeteredClient
from utils.logging_config import get_logger, log, run_context, stage_context
from utils.paths import basename


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
    # Deterministically detected discrepancies awaiting judgement. Empty means the
    # corpus is clean and adjudication is skipped rather than asked to confirm it.
    conflict_candidates: list[dict[str, Any]]
    plan_summary: dict[str, int]
    # False routes the run to `blocked`. Defaults to True only where absent, so a
    # missing value can never be read as "verification passed".
    verification_passed: bool
    verification: dict[str, Any]
    decisions: dict[str, str]
    actor: str
    status: str


logger = get_logger(__name__)

# Bumped when the extraction contract changes, because facts are reused across runs by
# (document, extractor_version) — so a stale version would keep serving facts extracted
# before rate-card qualifiers existed, and the rate-card fix would appear not to work on
# any corpus already ingested.
#
# v4: values are trimmed of boundary punctuation, citations tolerate typographic
# punctuation, and billing predicates are refused from documents that do not bill. All
# three change what a stored fact looks like, so reusing v3 facts would show none of it.
#
# v5: citations tolerate markdown layout. This changes which facts *survive* — a v4 run
# over the realistic corpus dropped five real terms whose quotes came out of table rows,
# including two invoice totals — and a dropped fact is not stored, so reusing v4 facts
# would leave every one of them missing with nothing to show why.
EXTRACTOR_VERSION = "extract-v5"


def _client(session, run_id: str) -> MeteredClient:
    return MeteredClient(build_provider(), session, UUID(run_id))


def mark_stage(run_id: str | None, stage: str | None, detail: str | None = None) -> None:
    """Record where a run currently is, in its own transaction.

    Its own transaction on purpose: this is progress reporting, and it has to be visible
    to a page polling from another connection *while* the node is still working. Written
    inside the node's session it would not appear until the node committed, which is
    exactly the moment it stops being useful.

    Failures are swallowed. A run must not die because it could not narrate itself.
    """
    if not run_id:
        return
    try:
        with session_scope() as session:
            run = session.get(Run, UUID(run_id))
            if run is not None:
                run.current_stage = stage
                run.stage_detail = detail
    except Exception as exc:  # pragma: no cover - reporting must never be fatal
        log(
            logger,
            logging.DEBUG,
            "could not record current stage",
            error=type(exc).__name__,
            detail=str(exc)[:120],
        )


def node(stage: str):
    """Bind logging context and announce entry/exit for a graph node.

    Applied as a decorator so no node can forget. Every line a node emits — including
    lines from deep inside the provider or the ORM — is then attributable to a
    specific run and stage, which is what makes concurrent runs debuggable at all.

    Entry is also recorded on the run row, so a page watching a two-minute run can say
    which stage it is in rather than showing "running" and nothing else.
    """

    def decorate(fn):
        def wrapper(state: RunState) -> dict[str, Any]:
            with run_context(state.get("run_id"), stage), stage_context(stage):
                log(logger, logging.INFO, f"{stage}: entering")
                mark_stage(state.get("run_id"), stage)
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

            # Insert-if-absent in one statement, then read back.
            #
            # A check-then-insert races: two concurrent runs over the same corpus both
            # see the document as absent, both insert, and one dies on
            # uq_document_corpus_sha256. That is not a rare interleaving — it is what
            # happens every time two runs start together, which behavior 9 requires to
            # work. Found by the concurrency test, not by review.
            inserted = session.execute(
                pg_insert(Document)
                .values(
                    corpus_id=UUID(state["corpus_id"]),
                    uri=loaded.uri,
                    sha256=loaded.sha256,
                    mime=loaded.mime,
                )
                .on_conflict_do_nothing(index_elements=["corpus_id", "sha256"])
                .returning(Document.id)
            ).scalar_one_or_none()

            if inserted is None:
                # The other run won, or this file was ingested earlier. Either way the
                # document exists and re-dropping a file is a normal no-op.
                existing_id = session.execute(
                    select(Document.id).where(
                        Document.corpus_id == UUID(state["corpus_id"]),
                        Document.sha256 == loaded.sha256,
                    )
                ).scalar_one()
                log(
                    logger,
                    logging.INFO,
                    "document already ingested, reusing",
                    document=path.name,
                    sha256=loaded.sha256[:12],
                )
                document_ids.append(str(existing_id))
                continue

            document = session.get(Document, inserted)
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

            name = basename(document.uri)
            text = "\n\n".join(
                c.text
                for c in session.query(Chunk)
                .filter(Chunk.document_id == document.id)
                .order_by(Chunk.ordinal)
                .all()
            )

            result = classify_document(text, name, client)
            document.kind_confidence = result.confidence
            document.vendor = result.vendor or None
            document.document_date = normalize_date(result.document_date)

            reason = escalation_reason(
                result, settings.classify_confidence_threshold
            )

            # A document that never names us may not be ours at all. Escalated rather
            # than rejected: deciding whether a subcontractor's agreement belongs in this
            # corpus is a judgement about the business, not about the text — and the
            # branch that asks a human already exists.
            if reason is None and names_our_organisation(
                text, settings.organisation_name
            ) is False:
                reason = (
                    f"the document never names {settings.organisation_name!r}, so it may "
                    f"be an agreement between other parties rather than one of ours"
                )

            if reason is not None:
                # The kind is deliberately NOT written when escalating.
                #
                # It used to be written first and escalated afterwards, which made the
                # gate fail open: classification is skipped for any document that already
                # has a kind, so a run abandoned at the escalation left the model's guess
                # in place and no later run ever asked again. A one-shot question whose
                # unanswered state is "accepted" is not a gate.
                #
                # Left unset, the document governs nothing — `_fact_views` reads an absent
                # kind as `unknown`, which is observational — and every subsequent run
                # re-asks until a human answers. Vendor and date are kept, because they
                # are what the reviewer needs in order to answer.
                log(
                    logger,
                    logging.WARNING,
                    "classification needs a human, escalating",
                    document=name,
                    kind=result.kind,
                    confidence=result.confidence,
                    fits_kind=result.fits_kind,
                    reason=reason,
                )
                escalations.append(
                    {
                        "document_id": str(document.id),
                        "document": name,
                        "proposed_kind": result.kind,
                        "confidence": result.confidence,
                        "reason": reason,
                        "reasoning": result.reasoning,
                    }
                )
                findings.append(
                    {
                        "severity": "medium",
                        "target_kind": "document",
                        "explanation": (
                            f"{name}: needs a human to say what it is — {reason}. "
                            f"Proposed {result.kind!r} at confidence "
                            f"{result.confidence:.2f}. "
                            f"Model's reasoning: {result.reasoning}"
                        ),
                    }
                )
            else:
                document.kind = result.kind
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


# What a human may answer when asked what a document is: the whole vocabulary, `unknown`
# included.
#
# It used to be the five governing kinds, which put the human in exactly the trap the
# classifier was in — obliged to name a nearest kind for a document that is none of
# them. Answering "amendment" for a data-protection addendum is how one came to outrank
# the agreement it accompanied. An `unknown` document is still ingested, extracted and
# reported; it simply does not govern.
ESCALATION_OPTIONS = DOCUMENT_KINDS

# What each kind *is*. The authority half is deliberately not written here — see
# `escalation_options`.
KIND_DESCRIPTIONS: dict[str, str] = {
    "msa": "a master agreement establishing base terms",
    "amendment": "modifies an existing agreement, and outranks it where they disagree",
    "sow": "a statement of work, binding its own engagement only",
    "invoice": "a bill for work done — it records what was charged",
    "renewal_notice": (
        "a notice about renewal or termination timing, which restates terms the "
        "agreement already set"
    ),
    "unknown": "none of these kinds fits the document",
}


def escalation_options() -> list[dict[str, object]]:
    """The answers a human may give, each with what choosing it *does*.

    Two of the six silently strip a document's authority, and the gate did not say so.
    That is not a hypothetical: on an independent re-test a rate revision letter
    escalated, and the model's own reasoning — shown at the gate — argued it was "a
    unilateral notice rather than a bilateral contract amendment". The reviewer followed
    that reasoning to `renewal_notice`, which is non-governing, so a 2026 rate card
    became observation-only and the register kept asserting 2024 rates. Invoices that
    billed the new rates correctly then produced two `high` "money may be recoverable"
    findings against a vendor that had done nothing wrong.

    The reviewer was not careless. They were shown a plausible option and not shown its
    one consequence that mattered.

    `governs` is **derived** from `OBSERVATIONAL_KINDS` rather than restated here. A
    second copy of that list is a second thing to update, and this project's recurring
    fault is exactly that shape — a rule applied in one place and not its peer, silently
    correct until the day it is not. Adding a kind to `OBSERVATIONAL_KINDS` changes what
    this gate tells a human, with no edit here.
    """
    options: list[dict[str, object]] = []
    for kind in ESCALATION_OPTIONS:
        governs = kind not in OBSERVATIONAL_KINDS
        options.append(
            {
                "kind": kind,
                "governs": governs,
                "description": KIND_DESCRIPTIONS[kind],
                "effect": (
                    "Sets terms in the register, and can supersede earlier documents."
                    if governs
                    else (
                        "Recorded and reported, but governs nothing — its values cannot "
                        "set or supersede a term, and will be treated as an observation "
                        "that may contradict the agreement."
                    )
                ),
            }
        )
    return options


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
            "options": ESCALATION_OPTIONS,
            # Added alongside `options` rather than replacing it: a machine client
            # reading the existing field keeps working, and one that wants to know what
            # an answer costs can now find out. Floor 4 is machine-drivable end to end,
            # and "which of these six answers disarms the document?" was answerable only
            # by reading the source.
            "option_effects": escalation_options(),
        }
    )

    # Same envelope tolerance as the gate: an empty answer must resume rather than
    # re-interrupt. "None of these needs reclassifying" is a legitimate reply.
    if isinstance(answers, dict) and "decisions" in answers:
        answers = answers.get("decisions") or {}

    applied = 0
    rejected = 0
    with session_scope() as session:
        for document_id, answer in (answers or {}).items():
            document = session.get(Document, UUID(document_id))
            if document is None or not answer:
                continue
            chosen_kind = accepted_kind(answer)
            if chosen_kind is None:
                # An answer outside the vocabulary is refused rather than stored. It
                # would otherwise reach `KIND_PRECEDENCE.get(kind, -1)` and be assigned
                # a rank by a dictionary miss — a typo becoming a silent decision about
                # what governs. The document keeps its escalation instead.
                log(
                    logger,
                    logging.WARNING,
                    "human classification refused: not a known document kind",
                    document_id=document_id,
                    chosen=answer,
                )
                rejected += 1
                continue
            document.kind = chosen_kind
            # A human answer is definitive; the threshold no longer applies to it.
            document.kind_confidence = 1.0
            applied += 1

    log(
        logger,
        logging.INFO,
        "human classifications applied",
        documents=applied,
        rejected=rejected,
    )
    return {"status": "classified_by_human"}


@node("extract")
def extract(state: RunState) -> dict[str, Any]:
    """Extract typed facts, keeping only those with resolvable citations."""
    settings = get_settings()
    fact_ids: list[str] = []
    findings: list[dict[str, Any]] = []

    with session_scope() as session:
        client = _client(session, state["run_id"])

        document_ids = state.get("document_ids", [])

        # Which documents will actually cost a model call, worked out once so progress
        # can be reported as "2 of 3" rather than "2 of 14" — on a corpus where eleven
        # documents are being reused, counting all of them would describe a run that is
        # not happening.
        reused = {
            str(row[0])
            for row in session.query(Fact.document_id)
            .filter(
                Fact.document_id.in_([UUID(d) for d in document_ids]),
                Fact.extractor_version == EXTRACTOR_VERSION,
            )
            .distinct()
            .all()
        } if document_ids else set()
        pending = [d for d in document_ids if d not in reused]

        for document_id in document_ids:
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
                    document=basename(document.uri),
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
            name = basename(document.uri)

            # Per document, because this is the stage that takes the time — a single
            # extract call on a 23 KB agreement ran to nearly forty seconds, and the
            # stage as a whole to ninety. "Extracting 2 of 3" is the difference between
            # waiting and wondering whether it has hung.
            mark_stage(
                state["run_id"],
                "extract",
                f"reading {name} ({pending.index(document_id) + 1} of {len(pending)})"
                if document_id in pending
                else f"reading {name}",
            )

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
            try:
                result = extract_from_document(
                    raw_chunks, name, client, max_retries=settings.extract_max_retries
                )
            except ExtractionFailed as exc:
                # Decision point 2's alternate branch: give up on this document,
                # report it, and keep going. One document the model cannot parse must
                # not take down a corpus — and must not disappear silently either.
                log(
                    logger,
                    logging.ERROR,
                    "extraction gave up on a document; run continues",
                    document=name,
                    detail=str(exc)[:200],
                )
                findings.append(
                    {
                        "severity": "high",
                        "target_kind": "document",
                        "explanation": (
                            f"{name} was skipped: extraction could not produce usable "
                            f"output after {settings.extract_max_retries + 1} attempts. "
                            f"No facts from this document are in the register."
                        ),
                    }
                )
                continue

            for fact in result.facts:
                if not predicate_allowed(fact.predicate, document.kind):
                    # A billing predicate from a document that bills nothing. Dropped
                    # here rather than filtered later, because storing it would let an
                    # amendment's cost table govern what was agreed — an observation
                    # arriving with the authority of an instrument.
                    findings.append(
                        {
                            "severity": "low",
                            "target_kind": "fact",
                            "explanation": (
                                f"{name}: dropped {fact.predicate} "
                                f"({fact.value_raw!r}) — only a document of kind "
                                f"{sorted(KIND_ONLY_PREDICATES[fact.predicate])} states "
                                f"what was billed, and this one is a "
                                f"{document.kind!r}."
                            ),
                        }
                    )
                    log(
                        logger,
                        logging.INFO,
                        "billing predicate dropped: wrong document kind",
                        document=name,
                        kind=document.kind,
                        predicate=fact.predicate,
                    )
                    continue

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
                    qualifier=fact.qualifier,
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


def _engagement_label(document) -> str:
    """A readable, stable name for the engagement a SOW covers.

    The scope used to be the document's raw UUID, and the scope is not an internal key —
    it is appended to the register row's heading and to the published document, so a
    reviewer was shown "Payment Terms (days) — 89Bcf37E 4753 4270 B534 3E97Cc29A320". A
    label nobody can read is a label that gets ignored, and the whole point of scoping was
    to make one rate-card row distinguishable from another.

    The filename stem, because it is what a person calls that SOW and it is unique within
    a corpus. The trade-off, stated plainly: renaming the file changes the scope and
    therefore re-keys that SOW's sections once. A UUID would not, and would go on being
    unreadable — the register is a document for people, so readability wins and the churn
    is bounded to the rename.
    """
    if document is None:
        return "engagement"
    stem = basename(document.uri).rsplit(".", 1)[0]
    return stem.replace("-", " ").replace("_", " ").strip() or "engagement"


def _scope_for(kind: str, fact, document=None) -> str | None:
    """What limits this fact's reach, if anything.

    A term can fail to govern the agreement for two unrelated reasons, and before this
    only the first was modelled:

    * it belongs to one engagement — a SOW's rate is not the agreement's rate;
    * it is one row of a rate card — a paralegal rate is not *the* hourly rate.

    Missing the second was the single worst defect found in realistic testing. Five
    timekeepers in one table became five competing values for `hourly_rate`, so every
    pair was a conflict candidate (C(5,2) = 10 of them, all false) and the governing
    value was decided by whichever fact happened to sort last — which picked the
    paralegal rate as the vendor's headline rate while the partner rate sat two rows
    above it. A wrong number carrying a real citation, which the verifier cannot catch
    because the cited passage genuinely does say $225.
    """
    parts = []
    if kind == "sow":
        parts.append(_engagement_label(document))
    qualifier = (getattr(fact, "qualifier", None) or "").strip()
    if qualifier:
        parts.append(qualifier.casefold())
    return "::".join(parts) or None


def _fact_views(session, corpus_id: str, fact_ids: list[str]) -> list[FactView]:
    """Build the reconciliation/conflict view over a run's facts.

    Shared by detect_conflicts and compose so the two can never disagree about which
    document a fact belongs to or which vendor it concerns — a divergence there would
    mean the register reports one thing and the conflict report another.
    """
    if not fact_ids:
        return []

    facts = (
        session.query(Fact).filter(Fact.id.in_([UUID(f) for f in fact_ids])).all()
    )
    documents = {
        d.id: d
        for d in session.query(Document).filter(Document.corpus_id == UUID(corpus_id)).all()
    }

    views = []
    for f in facts:
        document = documents.get(f.document_id)
        kind = (document.kind if document else None) or "unknown"
        views.append(
            FactView(
                fact_id=f.id,
                predicate=f.predicate,
                subject=(document.vendor if document and document.vendor else f.subject),
                value_raw=f.value_raw,
                value_norm=None,
                unit=f.unit,
                # A fact's own date wins; otherwise it inherits the document's.
                # Invoices state their date once in the header and never repeat it per
                # line, so without this fallback every invoice fact is undated and the
                # temporal comparator skips it silently.
                effective_date=(
                    f.effective_date
                    or (document.document_date if document else None)
                ),
                document_id=f.document_id,
                document_kind=kind,
                # Two independent reasons a term may not govern the agreement as a
                # whole: it belongs to one engagement (a SOW), or it is one row of a
                # rate card (a grade, a shift, a tier). Both make a value
                # non-competing, so both scope it — and a SOW's own rate card scopes
                # by both at once.
                scope=_scope_for(kind, f, document),
            )
        )
    return views


@node("detect_conflicts")
def detect_conflicts(state: RunState) -> dict[str, Any]:
    """Find discrepancies deterministically. No model runs in this node."""
    with session_scope() as session:
        views = _fact_views(session, state["corpus_id"], state.get("fact_ids", []))
        candidates = detect(views)

        log(
            logger,
            logging.INFO,
            "conflict candidates generated deterministically",
            candidates=len(candidates),
            by_kind={
                kind: sum(1 for c in candidates if c.kind == kind)
                for kind in {c.kind for c in candidates}
            },
        )

        client = _client(session, state["run_id"])
        client.record_stage("detect_conflicts")

        if not candidates:
            # Record the skip explicitly. Omitting the row would make "adjudication
            # correctly did not run" indistinguishable from "adjudication was never
            # wired up" — and the second is a bug that would ship unnoticed.
            client.record_stage("adjudicate", skipped=True)

        # Serialized into state rather than passed as objects: LangGraph checkpoints
        # after every node, and a FactView would not survive the round trip.
        payload = [
            {
                "kind": c.kind,
                "subject": c.subject,
                "predicate": c.predicate,
                "detail": c.detail,
                "a_fact_id": str(c.a.fact_id),
                "b_fact_id": str(c.b.fact_id),
                "a_value": c.a.value_raw,
                "b_value": c.b.value_raw,
                "a_kind": c.a.document_kind,
                "b_kind": c.b.document_kind,
                "a_date": c.a.effective_date.isoformat() if c.a.effective_date else None,
                "b_date": c.b.effective_date.isoformat() if c.b.effective_date else None,
                # The direction is measured here, by the code holding the magnitudes,
                # and must survive the checkpoint — recomputing it downstream would
                # mean guessing at an arithmetic candidate's product from its parts.
                "comparison": c.comparison.as_payload() if c.comparison else None,
            }
            for c in candidates
        ]

    return {"conflict_candidates": payload}


def route_after_detection(state: RunState) -> str:
    """Decision point 3: a clean corpus skips adjudication entirely.

    This is the honest-clean path. There is nothing to judge, so no model is called
    and no cost is incurred — and the skip is observable, because `stage_metric`
    records the stage as skipped rather than omitting it. A stage that legitimately
    did not run and a stage that was never wired up must not look alike.
    """
    return "adjudicate" if state.get("conflict_candidates") else "compose"


@node("adjudicate")
def adjudicate_conflicts(state: RunState) -> dict[str, Any]:
    """Ask the model which candidates are real contradictions, and how bad.

    The model can downgrade or explain. It cannot invent: the candidate list is fixed
    before this node runs, so recall is a property of the deterministic comparators
    rather than of a prompt.
    """
    payload = state.get("conflict_candidates", [])
    findings: list[dict[str, Any]] = []

    with session_scope() as session:
        views = {
            str(v.fact_id): v
            for v in _fact_views(session, state["corpus_id"], state.get("fact_ids", []))
        }

        candidates = [
            ConflictCandidate(
                kind=item["kind"],
                subject=item["subject"],
                predicate=item["predicate"],
                a=views[item["a_fact_id"]],
                b=views[item["b_fact_id"]],
                detail=item["detail"],
                comparison=Comparison.from_payload(item.get("comparison")),
            )
            for item in payload
            if item["a_fact_id"] in views and item["b_fact_id"] in views
        ]

        client = _client(session, state["run_id"])
        results = adjudicate(candidates, client)

        real = 0
        already_settled = 0

        for result in results:
            candidate = result.candidate

            # A contradiction between the same two facts is the same contradiction,
            # whichever run notices it. Facts are reused across runs, so re-detection
            # is normal — and inserting a second row for it violated the uniqueness
            # constraint and took the whole run down on the first incremental update.
            existing = (
                session.query(Conflict)
                .filter(
                    Conflict.a_fact_id == candidate.a.fact_id,
                    Conflict.b_fact_id == candidate.b.fact_id,
                    Conflict.kind == candidate.kind,
                )
                .one_or_none()
            )

            if existing is None:
                # Every candidate is persisted, including dismissed ones. I4 says never
                # silently resolve a contradiction — and a dismissal the reviewer cannot
                # see is exactly that, however well-reasoned it was.
                session.add(
                    Conflict(
                        kind=candidate.kind,
                        severity=result.severity,
                        status="open" if result.is_real_conflict else "dismissed",
                        a_fact_id=candidate.a.fact_id,
                        b_fact_id=candidate.b.fact_id,
                        explanation=result.explanation,
                        adjudicated=True,
                        detected_in_run=UUID(state["run_id"]),
                    )
                )
            elif existing.status != "open":
                # A human already ruled on this one. Re-raising it every run would
                # teach reviewers that their decisions do not stick, which is the
                # fastest way to make a review queue meaningless.
                already_settled += 1
                continue

            if result.is_real_conflict:
                real += 1
                findings.append(
                    {
                        "severity": result.severity,
                        "target_kind": "conflict",
                        "explanation": (
                            f"{candidate.subject} — {candidate.predicate}: "
                            f"{result.explanation}"
                        ),
                    }
                )

        client.record_stage("adjudicate")

        log(
            logger,
            logging.INFO,
            "adjudication complete",
            candidates=len(candidates),
            confirmed=real,
            dismissed=len(results) - real,
            already_settled=already_settled,
            explanations_withheld=sum(1 for r in results if r.explanation_withheld),
        )

    return {"findings": findings}


@node("compose")
def compose(state: RunState) -> dict[str, Any]:
    """Build the register, re-deriving only sections whose dependencies changed."""
    with session_scope() as session:
        # Same view the conflict detector saw. Built by one helper rather than two
        # copies, because a divergence here would mean the register reports one thing
        # and the conflict report another about the same fact.
        #
        # Reconcile before composing: without it the register would list every value
        # any document ever stated for a term, which is a pile of paper, not a
        # register. Reconciliation picks the one that governs and keeps the rest as
        # evidence.
        views = _fact_views(session, state["corpus_id"], state.get("fact_ids", []))

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
                        # Which rate-card row this is. Without it a five-grade rate card
                        # renders as five rows all labelled "Hourly Rate" with different
                        # values and nothing to tell them apart — technically correct
                        # and unreadable, which is its own kind of wrong for a register
                        # a person is meant to review.
                        "scope": resolution.scope,
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
                    governing_fact_id=governing.fact_id if governing else None,
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


@node("examine")
def examine(state: RunState) -> dict[str, Any]:
    """Stage A — apply the contract playbook to the reconciled values.

    Deterministic and model-free. "Payment terms must be net 30 or better" is
    arithmetic, and arithmetic done by a language model is arithmetic you cannot test.
    """
    settings = get_settings()
    findings: list[dict[str, Any]] = []

    try:
        ruleset_id, rules = load_rules(settings.rules_path)
    except (RuleError, OSError) as exc:
        # A broken playbook is reported, not skipped. Rules silently not running is
        # indistinguishable from rules passing, and that is the failure mode where a
        # compliance report is confidently empty.
        log(logger, logging.ERROR, "playbook could not be loaded", error=str(exc)[:300])
        return {
            "findings": [
                {
                    "severity": "high",
                    "target_kind": "ruleset",
                    "explanation": f"Rules were NOT applied: {exc}",
                }
            ]
        }

    with session_scope() as session:
        views = _fact_views(session, state["corpus_id"], state.get("fact_ids", []))
        violations = evaluate(rules, reconcile(views))

        # Compose runs before this stage, so the register already exists and a finding
        # can point at the row it is about. Attached only after confirming the key is
        # really there: a reviewer who clicks "show me the evidence" and gets nothing
        # learns to stop clicking, which costs more than never offering the link.
        #
        # Rules only ever evaluate unscoped resolutions, so the key is subject and
        # predicate. A `presence_required` violation has no section by definition —
        # the whole finding is that nothing was found — and correctly gets no link.
        section_keys = set(
            session.execute(
                select(SectionVersion.section_key).where(
                    SectionVersion.run_id == UUID(state["run_id"])
                )
            ).scalars()
        )

        for violation in violations:
            # Folded, because a section key is identity rather than display text. Built
            # from the written spelling this matched nothing once keys were folded, and
            # the failure would have been silent — every evidence link quietly absent,
            # with no error anywhere.
            candidate = f"{canonical_subject(violation.subject)}::{violation.rule.predicate}"
            findings.append(
                {
                    # The violation's severity, not the rule's: a rule that could not run
                    # is reported at "not checked" severity rather than at the severity
                    # of the breach it did not find.
                    "severity": violation.severity,
                    "target_kind": "rule",
                    "rule_code": violation.rule.code,
                    "subject": violation.subject,
                    "predicate": violation.rule.predicate,
                    "section_key": candidate if candidate in section_keys else None,
                    "explanation": f"[{violation.rule.code}] {violation.explanation}",
                }
            )

        client = _client(session, state["run_id"])
        client.record_stage("examine")

        log(
            logger,
            logging.INFO,
            "playbook applied",
            ruleset=ruleset_id,
            rules=len(rules),
            violations=len(violations),
            # Counted separately because it is a different fact about the run. A rule
            # that could not run is a gap in coverage, not a finding about the contract,
            # and lowering its severity to keep the queue readable would otherwise make
            # it invisible — which is how a vendor silently dropped out of three checks.
            not_checked=sum(1 for v in violations if not v.checked),
        )

    return {"findings": findings}


@node("verify")
def verify(state: RunState) -> dict[str, Any]:
    """Stage C — a fresh pair of eyes over the composed register.

    Re-derives nothing and trusts nothing: it checks that every claim still resolves to
    evidence that still exists and still says what it said. A failure here **blocks the
    commit** rather than annotating the output, because a run that cannot verify its own
    claims has not succeeded, and saying otherwise is exactly the lie I5 forbids.
    """
    with session_scope() as session:
        report = verify_run(session, UUID(state["run_id"]))

        log(
            logger,
            logging.INFO if report.passed else logging.ERROR,
            "verification passed" if report.passed else "VERIFICATION FAILED",
            **report.summary(),
        )

        findings = [
            {
                "severity": "high",
                "target_kind": "verification",
                "explanation": f"[{failure.reason}] {failure.section_key}: {failure.detail}",
            }
            for failure in report.failures
        ]

        client = _client(session, state["run_id"])
        client.record_stage("verify")

        if not report.passed:
            run = session.get(Run, UUID(state["run_id"]))
            if run is not None:
                run.status = "failed"

    return {
        "findings": findings,
        "verification_passed": report.passed,
        "verification": report.summary(),
    }


def route_after_verify(state: RunState) -> str:
    """Decision point: unverified work never reaches the commit.

    Routing to `blocked` rather than to `gate` is deliberate. Presenting a reviewer
    with findings drawn from a register we know is unsound invites them to approve it,
    and an approval obtained that way is worse than no approval at all.
    """
    return "gate" if state.get("verification_passed", True) else "blocked"


@node("blocked")
def blocked(state: RunState) -> dict[str, Any]:
    """Terminal state for a run whose register failed verification.

    Nothing is committed. The findings explaining *why* are already in state and are
    persisted here, so the failure is inspectable rather than merely reported.
    """
    with session_scope() as session:
        run = session.get(Run, UUID(state["run_id"]))
        if run is not None:
            run.status = "failed"
            run.ended_at = datetime.now(UTC)
            # Cleared, or the page would go on reporting the stage it died in as the
            # stage it is currently working.
            run.current_stage = None
            run.stage_detail = None

        for payload in state.get("findings", []):
            session.add(
                Finding(
                    run_id=UUID(state["run_id"]),
                    severity=payload.get("severity", "low"),
                    status="open",
                    target_kind=payload.get("target_kind", "document"),
                    explanation=payload.get("explanation", ""),
                )
            )

    log(
        logger,
        logging.ERROR,
        "run blocked: register failed verification, nothing committed",
        **(state.get("verification") or {}),
    )
    return {"status": "blocked_by_verification"}


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

    # Persisted before interrupt() pauses execution, so a GET on this run — which
    # reads this column, not the checkpoint — reports the true state instead of the
    # "running" it was created with. Without this a run parked at the gate looks
    # identical to one still mid-pipeline to anyone polling it from outside the
    # process that started it, which is every request the review UI makes.
    with session_scope() as session:
        run = session.get(Run, UUID(state["run_id"]))
        if run is not None:
            run.status = "awaiting_review"
            # Waiting for a person is not a stage in progress.
            run.current_stage = None
            run.stage_detail = None

    answer = interrupt(
        {
            "kind": "review_findings",
            "run_id": state["run_id"],
            "findings": [{"index": i, **f} for i, f in enumerate(findings)],
        }
    )

    # The resume value is an envelope (see service.resume_run). A bare mapping is still
    # accepted so a caller driving the graph directly does not have to know that.
    if isinstance(answer, dict) and "decisions" in answer:
        decisions = answer.get("decisions") or {}
        actor = answer.get("actor") or "human"
    else:
        decisions, actor = (answer or {}), "human"

    return {"decisions": decisions, "actor": actor, "status": "reviewed"}


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
                    actor=state.get("actor") or "human",
                )
            )

        if run is not None:
            run.status = "completed"
            run.current_stage = None
            run.stage_detail = None

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
    builder.add_node("detect_conflicts", detect_conflicts)
    builder.add_node("adjudicate", adjudicate_conflicts)
    builder.add_node("compose", compose)
    builder.add_node("examine", examine)
    builder.add_node("verify", verify)
    builder.add_node("blocked", blocked)
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
    builder.add_edge("extract", "detect_conflicts")

    # Decision point 3: a clean corpus skips adjudication entirely rather than paying
    # a model to confirm there is nothing to say.
    builder.add_conditional_edges(
        "detect_conflicts",
        route_after_detection,
        {"adjudicate": "adjudicate", "compose": "compose"},
    )
    builder.add_edge("adjudicate", "compose")
    builder.add_edge("compose", "examine")
    builder.add_edge("examine", "verify")

    # Decision point: an unverified register never reaches a human or a commit.
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {"gate": "gate", "blocked": "blocked"},
    )
    builder.add_edge("blocked", END)
    builder.add_edge("gate", "commit")
    builder.add_edge("commit", END)

    return builder.compile(checkpointer=checkpointer)
