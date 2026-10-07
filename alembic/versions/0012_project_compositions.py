"""add project compositions and immutable revisions

Revision ID: 0012_project_compositions
Revises: 0011_agent_retry_identity
Create Date: 2026-10-02
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0012_project_compositions"
down_revision: str | None = "0011_agent_retry_identity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "project_compositions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("current_revision_number", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True),
            server_default=sa.func.now(), nullable=False,
        ),
        sa.ForeignKeyConstraint(["project_id"], ["projects.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("project_id", name="uq_project_compositions_project_id"),
    )
    op.create_table(
        "project_composition_revisions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("composition_id", sa.Uuid(), nullable=False),
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
        sa.CheckConstraint("revision_number > 0", name="ck_composition_revision_positive"),
        sa.ForeignKeyConstraint(["composition_id"], ["project_compositions.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "composition_id", "revision_number", name="uq_composition_revision_number"
        ),
    )
    op.create_index(
        "ix_project_composition_revisions_composition_id",
        "project_composition_revisions",
        ["composition_id"],
    )
    op.create_index(
        "ix_project_composition_revisions_document_hash",
        "project_composition_revisions",
        ["document_hash"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_project_composition_revisions_document_hash",
        table_name="project_composition_revisions",
    )
    op.drop_index(
        "ix_project_composition_revisions_composition_id",
        table_name="project_composition_revisions",
    )
    op.drop_table("project_composition_revisions")
    op.drop_table("project_compositions")
