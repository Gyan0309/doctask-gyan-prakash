"""ORM models.

Re-exported from `models.tables` so callers write `from models import Fact` rather than
reaching into the module, and so the table definitions can later be split across files
without touching a single import site.
"""

from models.tables import (
    EMBEDDING_DIM,
    Base,
    Chunk,
    Claim,
    ClaimCitation,
    Conflict,
    Corpus,
    Decision,
    Document,
    Fact,
    Finding,
    FindingEvidence,
    ModelCacheEntry,
    Rule,
    Run,
    Section,
    SectionDependency,
    SectionVersion,
    StageMetric,
)

__all__ = [
    "EMBEDDING_DIM",
    "Base",
    "Chunk",
    "Claim",
    "ClaimCitation",
    "Conflict",
    "Corpus",
    "Decision",
    "Document",
    "Fact",
    "Finding",
    "FindingEvidence",
    "ModelCacheEntry",
    "Rule",
    "Run",
    "Section",
    "SectionDependency",
    "SectionVersion",
    "StageMetric",
]
