"""add assets.source_asset_id provenance link

Revision ID: 0002_asset_source
Revises: 0001_initial
Create Date: 2026-06-11

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002_asset_source"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "assets",
        sa.Column("source_asset_id", sa.Uuid(), nullable=True),
    )
    op.create_index(
        "ix_assets_source_asset_id", "assets", ["source_asset_id"]
    )
    op.create_foreign_key(
        "fk_assets_source_asset_id",
        "assets",
        "assets",
        ["source_asset_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_assets_source_asset_id", "assets", type_="foreignkey")
    op.drop_index("ix_assets_source_asset_id", table_name="assets")
    op.drop_column("assets", "source_asset_id")
