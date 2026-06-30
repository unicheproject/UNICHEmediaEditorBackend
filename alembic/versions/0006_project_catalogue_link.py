"""link projects to the catalogue (add projects.org_id)

Revision ID: 0006_project_catalogue_link
Revises: 0005_asset_intermediate
Create Date: 2026-06-30

Projects are now companion rows keyed by the Catalogue's project UUID. The
editor no longer mints its own ids (the application supplies the catalogue id
on insert), and we record the owning organisation for context. ``description``
stays as an editor-local field.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0006_project_catalogue_link"
down_revision: str | None = "0005_asset_intermediate"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "projects",
        sa.Column("org_id", sa.Uuid(), nullable=True),
    )
    op.create_index("ix_projects_org_id", "projects", ["org_id"])


def downgrade() -> None:
    op.drop_index("ix_projects_org_id", table_name="projects")
    op.drop_column("projects", "org_id")
