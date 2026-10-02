"""record AgentPlan execution attempts and claim tokens

Revision ID: 0010_agent_execution_attempts
Revises: 0009_temp_asset_retention
Create Date: 2026-10-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0010_agent_execution_attempts"
down_revision: str | None = "0009_temp_asset_retention"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_plans",
        sa.Column("active_attempt_id", sa.Uuid(), nullable=True),
    )
    op.create_table(
        "agent_plan_attempts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("plan_id", sa.Uuid(), nullable=False),
        sa.Column("attempt_number", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "step_runs",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "execution_summary",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
        sa.Column(
            "job_ids",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column(
            "result_asset_ids",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=False,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.ForeignKeyConstraint(["plan_id"], ["agent_plans.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("plan_id", "attempt_number", name="uq_agent_plan_attempt_number"),
    )
    op.create_index(
        "ix_agent_plan_attempts_plan_id", "agent_plan_attempts", ["plan_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_agent_plan_attempts_plan_id", table_name="agent_plan_attempts")
    op.drop_table("agent_plan_attempts")
    op.drop_column("agent_plans", "active_attempt_id")
