"""run current stage, so a long run can say where it is

Revision ID: c5a1e70f3b42
Revises: 78792bb0dacc
Create Date: 2026-08-17 15:10:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "c5a1e70f3b42"
down_revision: str | None = "78792bb0dacc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Nullable, because every existing run has already finished and has no current
    # stage. A default of "ingest" would claim otherwise about rows nobody is watching.
    op.add_column("run", sa.Column("current_stage", sa.String(length=40), nullable=True))
    op.add_column("run", sa.Column("stage_detail", sa.String(length=200), nullable=True))


def downgrade() -> None:
    op.drop_column("run", "stage_detail")
    op.drop_column("run", "current_stage")
