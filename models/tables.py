"""The data model.

Three tables carry requirements the rest of the system cannot enforce on its own:

  section_dependency  makes "nothing else changed" provable rather than asserted (I2)
  claim_citation      makes "no claim without a citation" enforceable (I1)
  decision            makes the human gate auditable (I6)

Everything else is bookkeeping around those three.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBEDDING_DIM = 768  # text-embedding-004 / gemini-embedding output width


class Base(DeclarativeBase):
    pass


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


# ---------------------------------------------------------------------------
# Corpus and sources
# ---------------------------------------------------------------------------


class Corpus(Base):
    __tablename__ = "corpus"

    id: Mapped[uuid.UUID] = _pk()
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    created_at: Mapped[datetime] = _now()


class Document(Base):
    __tablename__ = "document"

    id: Mapped[uuid.UUID] = _pk()
    corpus_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("corpus.id", ondelete="CASCADE"), nullable=False
    )
    uri: Mapped[str] = mapped_column(Text, nullable=False)

    # Content hash, not path. Re-dropping the same file must be a no-op, and a file
    # edited in place must be a new document — both fall out of hashing content.
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    mime: Mapped[str] = mapped_column(String(120), nullable=False)

    # Classification result. Confidence is stored because it *routes* — below the
    # configured threshold the graph escalates to a human instead of guessing.
    kind: Mapped[str | None] = mapped_column(String(50))
    kind_confidence: Mapped[float | None] = mapped_column(Float)

    # The vendor this document concerns, decided once per document by the classifier.
    #
    # Not taken from the per-fact `subject`, which is asked of the extractor once per
    # fact and drifts accordingly: a live run produced "Northwind Analytics LLC",
    # "Meridian Retail Group" (the *client*) and "Meridian Retail Group and Northwind
    # Analytics LLC" for the same vendor across one corpus. Because the register is
    # grouped by subject, that fragmented one vendor into three and silently prevented
    # an amendment from superseding the term it amended — the register showed both the
    # old and new notice period, each looking authoritative.
    #
    # One decision, made by the pass that reads the whole document, beats N decisions
    # made from fragments.
    vendor: Mapped[str | None] = mapped_column(String(200))

    # The date this document takes effect, or is dated. Read once per document by the
    # classifier, and used as the effective date for any fact that does not state one
    # of its own.
    #
    # This is load-bearing rather than cosmetic. Invoices state their date in the
    # header and then never repeat it per line item, so every extracted invoice fact
    # arrived with a NULL effective date — which made the temporal comparator skip
    # them entirely and rendered the central conflict of the domain undetectable. The
    # arithmetic was right, the grouping was right, and the check silently never ran.
    document_date: Mapped[date | None] = mapped_column(Date)

    ingested_at: Mapped[datetime] = _now()

    __table_args__ = (
        UniqueConstraint("corpus_id", "sha256", name="uq_document_corpus_sha256"),
        Index("ix_document_corpus", "corpus_id"),
    )


class Chunk(Base):
    __tablename__ = "chunk"

    id: Mapped[uuid.UUID] = _pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Character spans are what make a citation resolvable back to the source. Without
    # them a "citation" is just a document name, which proves nothing.
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer)

    embedding: Mapped[Any | None] = mapped_column(Vector(EMBEDDING_DIM))

    __table_args__ = (
        UniqueConstraint("document_id", "ordinal", name="uq_chunk_document_ordinal"),
        Index("ix_chunk_document", "document_id"),
    )


# ---------------------------------------------------------------------------
# Facts — the normalized layer everything downstream reasons over
# ---------------------------------------------------------------------------


class Fact(Base):
    __tablename__ = "fact"

    id: Mapped[uuid.UUID] = _pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), nullable=False
    )
    chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("chunk.id", ondelete="SET NULL")
    )

    predicate: Mapped[str] = mapped_column(String(80), nullable=False)
    subject: Mapped[str] = mapped_column(String(200), nullable=False)

    # Raw keeps what the document literally said; norm is what we compare on.
    # Keeping both is what lets a conflict explanation quote the source verbatim
    # while the comparison still runs on canonical units.
    value_raw: Mapped[str] = mapped_column(Text, nullable=False)
    value_norm: Mapped[str | None] = mapped_column(Text)
    unit: Mapped[str | None] = mapped_column(String(30))

    effective_date: Mapped[date | None] = mapped_column(Date)
    confidence: Mapped[float | None] = mapped_column(Float)
    extractor_version: Mapped[str] = mapped_column(String(40), nullable=False)

    # An amendment supersedes an MSA term. Modelled as a link rather than a delete,
    # because the superseded value is exactly what a late invoice will conflict with.
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("fact.id", ondelete="SET NULL")
    )

    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        Index("ix_fact_predicate_subject", "predicate", "subject"),
        Index("ix_fact_document", "document_id"),
    )


# ---------------------------------------------------------------------------
# The deliverable, stored as sections — see DESIGN.md §3
# ---------------------------------------------------------------------------


class Section(Base):
    __tablename__ = "section"

    id: Mapped[uuid.UUID] = _pk()
    corpus_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("corpus.id", ondelete="CASCADE"), nullable=False
    )

    # Stable across runs — this is what a version chain hangs from, so it must be
    # derived from meaning (vendor + term type), never from row position.
    section_key: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(40), nullable=False)

    __table_args__ = (
        UniqueConstraint("corpus_id", "section_key", name="uq_section_corpus_key"),
        CheckConstraint("kind IN ('register_row','narrative')", name="ck_section_kind"),
    )


class SectionVersion(Base):
    __tablename__ = "section_version"

    id: Mapped[uuid.UUID] = _pk()
    section_key: Mapped[str] = mapped_column(String(200), nullable=False)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("run.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # The proof mechanism. "Untouched" is a hash comparison between two versions,
    # not a narrative claim in a README.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    prev_version_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("section_version.id", ondelete="SET NULL")
    )

    # True when this version was carried forward unchanged. Lets the change ledger
    # answer "what did this run actually re-derive?" with a COUNT rather than a diff.
    carried_forward: Mapped[bool] = mapped_column(nullable=False, default=False)

    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        UniqueConstraint("run_id", "section_key", name="uq_section_version_run_key"),
        Index("ix_section_version_key", "section_key"),
    )


class SectionDependency(Base):
    """THE dependency map. A new fact invalidates exactly the sections listed here
    against it — which is what makes an update cost like an update."""

    __tablename__ = "section_dependency"

    section_key: Mapped[str] = mapped_column(String(200), primary_key=True)
    fact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact.id", ondelete="CASCADE"), primary_key=True
    )

    __table_args__ = (Index("ix_section_dependency_fact", "fact_id"),)


# ---------------------------------------------------------------------------
# Claims and citations — I1
# ---------------------------------------------------------------------------


class Claim(Base):
    __tablename__ = "claim"

    id: Mapped[uuid.UUID] = _pk()
    section_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("section_version.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # 'unsupported' is a first-class outcome, not an error state. A claim whose
    # citations no longer resolve renders as unsupported rather than disappearing —
    # the system declining to bluff, visibly.
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="supported")

    __table_args__ = (
        CheckConstraint(
            "status IN ('supported','unsupported','conflicted')", name="ck_claim_status"
        ),
        Index("ix_claim_section_version", "section_version_id"),
    )


class ClaimCitation(Base):
    __tablename__ = "claim_citation"

    claim_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("claim.id", ondelete="CASCADE"), primary_key=True
    )
    fact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact.id", ondelete="CASCADE"), primary_key=True
    )


# ---------------------------------------------------------------------------
# Conflicts, rules, findings
# ---------------------------------------------------------------------------


class Conflict(Base):
    __tablename__ = "conflict"

    id: Mapped[uuid.UUID] = _pk()
    kind: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)

    # Never auto-resolved (I4). A conflict leaves this table only via a human
    # decision, which is why status and decision are separate concepts.
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open")

    a_fact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact.id", ondelete="CASCADE"), nullable=False
    )
    b_fact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact.id", ondelete="CASCADE"), nullable=False
    )
    explanation: Mapped[str | None] = mapped_column(Text)
    adjudicated: Mapped[bool] = mapped_column(nullable=False, default=False)

    detected_in_run: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("run.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(
            "status IN ('open','accepted','dismissed')", name="ck_conflict_status"
        ),
        CheckConstraint(
            "severity IN ('low','medium','high')", name="ck_conflict_severity"
        ),
        UniqueConstraint("a_fact_id", "b_fact_id", "kind", name="uq_conflict_pair_kind"),
    )


class Rule(Base):
    """Loaded from rules/*.yaml. A new rule is a data change; a new rule *kind* is a
    code change. Four kinds cover the playbook."""

    __tablename__ = "rule"

    id: Mapped[uuid.UUID] = _pk()
    ruleset_id: Mapped[str] = mapped_column(String(80), nullable=False)
    code: Mapped[str] = mapped_column(String(40), nullable=False)
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    predicate: Mapped[str] = mapped_column(String(80), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)

    __table_args__ = (
        UniqueConstraint("ruleset_id", "code", name="uq_rule_ruleset_code"),
        CheckConstraint(
            "kind IN ('numeric_bound','numeric_max','numeric_min','presence_required')",
            name="ck_rule_kind",
        ),
    )


class Finding(Base):
    __tablename__ = "finding"

    id: Mapped[uuid.UUID] = _pk()
    rule_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("rule.id", ondelete="SET NULL")
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("run.id", ondelete="CASCADE"), nullable=False
    )
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="open")

    target_kind: Mapped[str] = mapped_column(String(40), nullable=False)
    target_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True))
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(
            "status IN ('open','approved','rejected')", name="ck_finding_status"
        ),
        Index("ix_finding_run", "run_id"),
    )


class FindingEvidence(Base):
    __tablename__ = "finding_evidence"

    finding_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("finding.id", ondelete="CASCADE"), primary_key=True
    )
    fact_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("fact.id", ondelete="CASCADE"), primary_key=True
    )


# ---------------------------------------------------------------------------
# Runs, decisions, metering
# ---------------------------------------------------------------------------


class Run(Base):
    __tablename__ = "run"

    id: Mapped[uuid.UUID] = _pk()
    corpus_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("corpus.id", ondelete="CASCADE"), nullable=False
    )
    parent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("run.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="running")

    # There is no separate "incremental mode" (D6). Mode is a label for humans; the
    # code path is identical, and a full run is the case where every section happens
    # to be invalid.
    mode: Mapped[str] = mapped_column(String(20), nullable=False, default="incremental")

    started_at: Mapped[datetime] = _now()
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "status IN ('running','awaiting_review','completed','failed',"
            "'escalated','interrupted')",
            name="ck_run_status",
        ),
        Index("ix_run_corpus", "corpus_id"),
    )


class Decision(Base):
    """The audit trail. Every human verdict lands here before it changes anything,
    so "who approved what, when, and why" is a query rather than a log grep."""

    __tablename__ = "decision"

    id: Mapped[uuid.UUID] = _pk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("run.id", ondelete="CASCADE"), nullable=False
    )
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    verdict: Mapped[str] = mapped_column(String(20), nullable=False)
    actor: Mapped[str] = mapped_column(String(120), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(
            "verdict IN ('approved','rejected','deferred')", name="ck_decision_verdict"
        ),
        Index("ix_decision_run", "run_id"),
    )


class StageMetric(Base):
    """Behavior 10. Populated for every stage of every run, including stages that were
    skipped — a skipped stage with zero cost is itself the evidence that the graph
    took a different path."""

    __tablename__ = "stage_metric"

    id: Mapped[uuid.UUID] = _pk()
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("run.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    started: Mapped[datetime] = _now()
    ended: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    cache_hits: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cache_misses: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skipped: Mapped[bool] = mapped_column(nullable=False, default=False)

    __table_args__ = (Index("ix_stage_metric_run", "run_id"),)


class ModelCacheEntry(Base):
    """Content-addressed cache in front of every model call.

    Does more work than its size suggests: it makes resumption after a kill nearly
    free, and its hit counter is *how we prove* an update cost like an update. A
    resume that re-executed everything and a resume that skipped correctly both
    finish; only the hit count tells them apart.
    """

    __tablename__ = "model_cache"

    id: Mapped[uuid.UUID] = _pk()

    # (stage, input_hash, model, prompt_version) — prompt_version is in the key so
    # editing a prompt invalidates its cache instead of silently serving stale output.
    cache_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    stage: Mapped[str] = mapped_column(String(50), nullable=False)
    model: Mapped[str] = mapped_column(String(80), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(40), nullable=False)

    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (Index("ix_model_cache_stage", "stage"),)
