"""Additive waveform migration from the existing head, with historical rows."""

import importlib.util
import io
from pathlib import Path

import sqlalchemy as sa
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory

from alembic import command
from app.core.config import settings

ROOT = Path(__file__).resolve().parents[1]


def test_waveform_migration_upgrades_existing_rows_and_downgrades(monkeypatch) -> None:
    spec = importlib.util.spec_from_file_location(
        "waveform_migration", ROOT / "alembic/versions/0013_asset_waveforms.py",
    )
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    assert migration.down_revision == "0012_project_compositions"
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE assets (id TEXT PRIMARY KEY)")
        connection.exec_driver_sql("INSERT INTO assets(id) VALUES ('existing-asset')")
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        migration.upgrade()
        row = connection.exec_driver_sql("SELECT * FROM assets").mappings().one()
        assert dict(row) == {
            "id": "existing-asset", "waveform_status": "pending", "waveform_reason": None,
        }
        migration.downgrade()
        assert connection.exec_driver_sql("SELECT * FROM assets").one() == ("existing-asset",)
    engine.dispose()


def test_single_head_and_postgres_offline_upgrade_sql(monkeypatch) -> None:
    output = io.StringIO()
    config = Config(output_buffer=output)
    config.set_main_option("script_location", str(ROOT / "alembic"))
    assert ScriptDirectory.from_config(config).get_heads() == ["0013_asset_waveforms"]
    monkeypatch.setattr(settings, "database_url", "postgresql+asyncpg://test:test@localhost/test")
    command.upgrade(config, "0012_project_compositions:head", sql=True)
    sql = output.getvalue()
    assert "ADD COLUMN waveform_status" in sql
    assert "DEFAULT 'pending' NOT NULL" in sql
    assert "ADD COLUMN waveform_reason" in sql
