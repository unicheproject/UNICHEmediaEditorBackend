"""add organisation brand kits and immutable revisions (MED/029)

One brand kit per Catalogue organisation (the tenant), with immutable,
numbered revisions as for project compositions. ``org_id`` is the Catalogue
organisation UUID; the Catalogue stays the authority, so there is no local
organisations table and no foreign key to one.

Revision ID: 0017_org_brand_kits
Revises: 0016_asset_display_size
Create Date: 2026-11-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0017_org_brand_kits"
down_revision: str | None = "0016_asset_display_size"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "org_brand_kits",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("org_id", sa.Uuid(), nullable=False),
        sa.Column("current_revision_number", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.CheckConstraint(
            "current_revision_number > 0", name="ck_org_brand_kit_current_positive"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("org_id", name="uq_org_brand_kits_org_id"),
    )
    op.create_table(
        "org_brand_kit_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("brand_kit_id", sa.Uuid(), nullable=False),
        sa.Column("revision_number", sa.Integer(), nullable=False),
        sa.Column(
            "document",
            sa.JSON().with_variant(
                postgresql.JSONB(astext_type=sa.Text()), "postgresql"
            ),
            nullable=False,
        ),
        sa.Column("document_hash", sa.String(length=64), nullable=False),
        sa.Column("author_subject", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.CheckConstraint("revision_number > 0", name="ck_org_brand_kit_revision_positive"),
        sa.ForeignKeyConstraint(["brand_kit_id"], ["org_brand_kits.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "brand_kit_id", "revision_number", name="uq_org_brand_kit_revision_number"
        ),
    )
    op.create_index(
        "ix_org_brand_kit_revisions_brand_kit_id",
        "org_brand_kit_revisions",
        ["brand_kit_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_org_brand_kit_revisions_brand_kit_id", table_name="org_brand_kit_revisions"
    )
    op.drop_table("org_brand_kit_revisions")
    op.drop_table("org_brand_kits")
