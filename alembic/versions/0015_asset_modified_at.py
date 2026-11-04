"""Add a user-visible Asset modification time.

Existing rows are backfilled from created_at: updated_at is also advanced by
background analysis (metadata probe, waveform, Temp expiry), so it cannot tell
historical renames apart and is deliberately not used.

Revision ID: 0015_asset_modified_at
Revises: 0014_asset_provenance
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0015_asset_modified_at"
down_revision: str | None = "0014_asset_provenance"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("assets", sa.Column("modified_at", sa.DateTime(timezone=True), nullable=True))
    op.execute("UPDATE assets SET modified_at = created_at")
    op.alter_column(
        "assets", "modified_at", nullable=False, server_default=sa.func.now(),
    )


def downgrade() -> None:
    op.drop_column("assets", "modified_at")
