"""persist compiled Agent execution summaries

Revision ID: 0008_agent_render_summary
Revises: 0007_asset_technical_metadata
Create Date: 2026-10-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0008_agent_render_summary"
down_revision: str | None = "0007_asset_technical_metadata"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "agent_plans",
        sa.Column(
            "execution_summary",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("agent_plans", "execution_summary")
