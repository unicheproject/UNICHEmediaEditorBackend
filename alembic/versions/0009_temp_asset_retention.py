"""add Temp Processed Asset retention state

Revision ID: 0009_temp_asset_retention
Revises: 0008_agent_render_summary
Create Date: 2026-10-01
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0009_temp_asset_retention"
down_revision: str | None = "0008_agent_render_summary"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "assets", sa.Column("temp_expires_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "assets", sa.Column("physical_expired_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_index("ix_assets_temp_expires_at", "assets", ["temp_expires_at"])


def downgrade() -> None:
    op.drop_index("ix_assets_temp_expires_at", table_name="assets")
    op.drop_column("assets", "physical_expired_at")
    op.drop_column("assets", "temp_expires_at")
