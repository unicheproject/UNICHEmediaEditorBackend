"""Migration identifiers must fit Alembic's production version column."""

from pathlib import Path

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory


def test_revision_identifiers_fit_postgres_version_column() -> None:
    config = Config()
    config.set_main_option(
        "script_location", str(Path(__file__).resolve().parents[1] / "alembic"),
    )
    scripts = ScriptDirectory.from_config(config)
    context = MigrationContext.configure(dialect_name="postgresql")
    limit = context._version.c.version_num.type.length
    assert limit is not None
    assert scripts.get_heads() == ["0013_asset_waveforms"]
    for revision in scripts.walk_revisions():
        assert len(revision.revision) <= limit, (
            f"Revision {revision.revision!r} exceeds alembic_version limit {limit}"
        )
