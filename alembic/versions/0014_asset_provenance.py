"""Persist authoritative editor provenance with conservative historical backfill.

Revision ID: 0014_asset_provenance
Revises: 0013_asset_waveforms
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0014_asset_provenance"
down_revision: str | None = "0013_asset_waveforms"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Unknown defaults also keep rolling-deployment legacy writes conservative.
    op.add_column("assets", sa.Column(
        "provenance_kind", sa.String(16), nullable=False, server_default="unknown",
    ))
    op.add_column("assets", sa.Column(
        "ai_provenance", sa.String(16), nullable=False, server_default="unknown",
    ))
    op.add_column("assets", sa.Column("source_job_id", sa.Uuid(), nullable=True))
    op.create_foreign_key(
        "fk_assets_source_job", "assets", "jobs", ["source_job_id"], ["id"],
        ondelete="SET NULL",
    )
    op.create_index("ix_assets_source_job_id", "assets", ["source_job_id"])
    op.create_check_constraint(
        "ck_assets_provenance_kind", "assets",
        "provenance_kind IN ('original', 'derived', 'unknown')",
    )
    op.create_check_constraint(
        "ck_assets_ai_provenance", "assets",
        "ai_provenance IN ('none', 'generated', 'modified', 'inherited', 'unknown')",
    )
    op.execute("UPDATE assets SET provenance_kind = 'derived' WHERE source_asset_id IS NOT NULL")


def downgrade() -> None:
    op.drop_constraint("ck_assets_ai_provenance", "assets", type_="check")
    op.drop_constraint("ck_assets_provenance_kind", "assets", type_="check")
    op.drop_index("ix_assets_source_job_id", table_name="assets")
    op.drop_constraint("fk_assets_source_job", "assets", type_="foreignkey")
    op.drop_column("assets", "source_job_id")
    op.drop_column("assets", "ai_provenance")
    op.drop_column("assets", "provenance_kind")
