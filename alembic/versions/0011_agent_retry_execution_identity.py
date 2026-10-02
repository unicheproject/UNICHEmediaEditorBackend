"""record deterministic physical-node identity for Agent retry reuse

Revision ID: 0011_agent_retry_execution_identity
Revises: 0010_agent_execution_attempts
Create Date: 2026-10-02
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0011_agent_retry_execution_identity"
down_revision: str | None = "0010_agent_execution_attempts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("agent_plan_id", sa.Uuid(), nullable=True))
    op.add_column("jobs", sa.Column("agent_attempt_id", sa.Uuid(), nullable=True))
    op.add_column("jobs", sa.Column("execution_identity", sa.String(length=64), nullable=True))
    op.create_foreign_key(
        "fk_jobs_agent_plan_id_agent_plans",
        "jobs",
        "agent_plans",
        ["agent_plan_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_jobs_agent_attempt_id_agent_plan_attempts",
        "jobs",
        "agent_plan_attempts",
        ["agent_attempt_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_jobs_agent_reuse_lookup",
        "jobs",
        ["agent_plan_id", "agent_attempt_id", "execution_identity"],
    )


def downgrade() -> None:
    op.drop_index("ix_jobs_agent_reuse_lookup", table_name="jobs")
    op.drop_constraint("fk_jobs_agent_attempt_id_agent_plan_attempts", "jobs", type_="foreignkey")
    op.drop_constraint("fk_jobs_agent_plan_id_agent_plans", "jobs", type_="foreignkey")
    op.drop_column("jobs", "execution_identity")
    op.drop_column("jobs", "agent_attempt_id")
    op.drop_column("jobs", "agent_plan_id")
