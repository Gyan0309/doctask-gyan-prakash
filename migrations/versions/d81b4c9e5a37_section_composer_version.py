"""section composer version, so a derivation change invalidates carried-forward content

Revision ID: d81b4c9e5a37
Revises: c5a1e70f3b42
Create Date: 2026-08-17 17:40:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "d81b4c9e5a37"
down_revision: str | None = "c5a1e70f3b42"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable and left NULL for existing rows on purpose. NULL compares unequal to any
    # current composer version, so every section written before this existed is re-derived
    # exactly once — the safe direction. Backfilling a value would assert that old content
    # was produced by the current logic, which is the thing this column exists to deny.
    op.add_column(
        "section_version",
        sa.Column("composer_version", sa.String(length=40), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("section_version", "composer_version")
