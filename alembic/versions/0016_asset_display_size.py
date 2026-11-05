"""Record Asset dimensions as displayed (video rotation / EXIF orientation).

Existing rows stay null: their stored width/height are coded dimensions, and
whether a file is rotated can only be learned by probing it again. Run
``python -m app.scripts.backfill_asset_display_size`` to fill them.

Revision ID: 0016_asset_display_size
Revises: 0015_asset_modified_at
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0016_asset_display_size"
down_revision: str | None = "0015_asset_modified_at"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("assets", sa.Column("display_width", sa.Integer(), nullable=True))
    op.add_column("assets", sa.Column("display_height", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("assets", "display_height")
    op.drop_column("assets", "display_width")
