"""add 'subtitle' to media_type enum

Revision ID: 0003_subtitle_media_type
Revises: 0002_asset_source
Create Date: 2026-06-12

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0003_subtitle_media_type"
down_revision: str | None = "0002_asset_source"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ALTER TYPE ... ADD VALUE cannot run inside a transaction block.
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("COMMIT")
        op.execute("ALTER TYPE media_type ADD VALUE IF NOT EXISTS 'subtitle'")


def downgrade() -> None:
    # PostgreSQL has no DROP VALUE for enums; leave the value in place.
    pass
