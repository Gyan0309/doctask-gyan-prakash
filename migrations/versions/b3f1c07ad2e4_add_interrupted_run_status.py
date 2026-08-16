"""add interrupted run status

A run whose process died has no way to say so. Its row stays `running` forever, so the
API reports work in progress that nothing is progressing — the same class of untruth as
a false success, and the one floor 5 forbids.

Revision ID: b3f1c07ad2e4
Revises: ce09d97cc5d2
Create Date: 2026-08-16 11:15:02.401118
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "b3f1c07ad2e4"
down_revision: str | None = "ce09d97cc5d2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("ck_run_status", "run", type_="check")
    op.create_check_constraint(
        "ck_run_status",
        "run",
        "status IN ('running','awaiting_review','completed','failed',"
        "'escalated','interrupted')",
    )


def downgrade() -> None:
    # Rows carrying the new value would violate the old constraint, so they are moved
    # to the closest honest pre-existing state before it is reinstated.
    op.execute("UPDATE run SET status = 'failed' WHERE status = 'interrupted'")
    op.drop_constraint("ck_run_status", "run", type_="check")
    op.create_check_constraint(
        "ck_run_status",
        "run",
        "status IN ('running','awaiting_review','completed','failed','escalated')",
    )
