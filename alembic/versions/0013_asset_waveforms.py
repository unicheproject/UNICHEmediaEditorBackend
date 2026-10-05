"""Add independent Asset waveform enrichment status.

Revision ID: 0013_asset_waveforms
Revises: 0012_project_compositions
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0013_asset_waveforms"
down_revision: str | None = "0012_project_compositions"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("assets", sa.Column(
        "waveform_status", sa.String(20), nullable=False, server_default="pending",
    ))
    op.add_column("assets", sa.Column("waveform_reason", sa.String(32), nullable=True))


def downgrade() -> None:
    op.drop_column("assets", "waveform_reason")
    op.drop_column("assets", "waveform_status")
