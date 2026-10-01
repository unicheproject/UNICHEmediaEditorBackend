"""persist normalized technical media metadata for assets

Revision ID: 0007_asset_technical_metadata
Revises: 0006_project_catalogue_link
Create Date: 2026-10-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0007_asset_technical_metadata"
down_revision: str | None = "0006_project_catalogue_link"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "assets",
        sa.Column(
            "technical_metadata_status",
            sa.String(length=20),
            server_default="pending",
            nullable=False,
        ),
    )
    op.create_index(
        "ix_assets_technical_metadata_status",
        "assets",
        ["technical_metadata_status"],
    )
    for name, type_ in (
        ("duration_seconds", sa.Float()),
        ("container_format", sa.String(length=128)),
        ("width", sa.Integer()),
        ("height", sa.Integer()),
        ("has_video_stream", sa.Boolean()),
        ("has_audio_stream", sa.Boolean()),
        ("video_codec", sa.String(length=128)),
        ("audio_codec", sa.String(length=128)),
        ("frame_rate", sa.Float()),
        ("audio_sample_rate", sa.Integer()),
        ("audio_channels", sa.Integer()),
        ("bit_rate", sa.BigInteger()),
        (
            "technical_streams",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
        ),
    ):
        op.add_column("assets", sa.Column(name, type_, nullable=True))


def downgrade() -> None:
    for name in (
        "technical_streams",
        "bit_rate",
        "audio_channels",
        "audio_sample_rate",
        "frame_rate",
        "audio_codec",
        "video_codec",
        "has_audio_stream",
        "has_video_stream",
        "height",
        "width",
        "container_format",
        "duration_seconds",
    ):
        op.drop_column("assets", name)
    op.drop_index("ix_assets_technical_metadata_status", table_name="assets")
    op.drop_column("assets", "technical_metadata_status")
