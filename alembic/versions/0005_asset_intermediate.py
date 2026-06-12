"""add assets.is_intermediate flag

Revision ID: 0005_asset_intermediate
Revises: 0004_agent_tables
Create Date: 2026-06-12

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0005_asset_intermediate"
down_revision: str | None = "0004_agent_tables"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "assets",
        sa.Column(
            "is_intermediate",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_index("ix_assets_is_intermediate", "assets", ["is_intermediate"])


def downgrade() -> None:
    op.drop_index("ix_assets_is_intermediate", table_name="assets")
    op.drop_column("assets", "is_intermediate")
