"""agent sessions + plans

Revision ID: 0004_agent_tables
Revises: 0003_subtitle_media_type
Create Date: 2026-06-12

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0004_agent_tables"
down_revision: str | None = "0003_subtitle_media_type"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

plan_status = sa.Enum(
    "proposed", "approved", "running", "succeeded", "failed", name="agent_plan_status"
)


def upgrade() -> None:
    op.create_table(
        "agent_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("asset_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("messages", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_sessions_project_id", "agent_sessions", ["project_id"])

    op.create_table(
        "agent_plans",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("session_id", sa.Uuid(), nullable=False),
        sa.Column("summary", sa.String(length=1024), nullable=True),
        sa.Column("status", plan_status, nullable=False),
        sa.Column("steps", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("step_runs", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("result_asset_ids", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["agent_sessions.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_agent_plans_session_id", "agent_plans", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_agent_plans_session_id", table_name="agent_plans")
    op.drop_table("agent_plans")
    op.drop_index("ix_agent_sessions_project_id", table_name="agent_sessions")
    op.drop_table("agent_sessions")
    plan_status.drop(op.get_bind(), checkfirst=True)
