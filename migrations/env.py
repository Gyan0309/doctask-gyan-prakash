"""Alembic environment.

The database URL is read from the environment, never from alembic.ini — a connection
string in a committed file is the same class of mistake as a committed key.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from ledger.config import get_settings
from ledger.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

config.set_main_option("sqlalchemy.url", get_settings().database_url)

target_metadata = Base.metadata


def _include_object(obj, name, type_, reflected, compare_to) -> bool:
    """Keep LangGraph's checkpoint tables out of our migrations.

    PostgresSaver creates and owns its own schema at runtime (checkpoints,
    checkpoint_blobs, checkpoint_writes, checkpoint_migrations). Autogenerate sees
    them as unmanaged tables and would helpfully write a migration that DROPs them —
    which destroys exactly the state that makes runs resumable.
    """
    if type_ == "table" and name.startswith("checkpoint"):
        return False
    return True


def _render_item(type_, obj, autogen_context) -> bool:
    """Make autogenerate import pgvector when it emits a vector column.

    Without this, every migration touching an embedding column is generated
    referencing `pgvector.sqlalchemy.vector.VECTOR` with no matching import, and
    fails with NameError the first time it runs anywhere clean. It is a silent trap:
    the file looks correct, and the developer who wrote it never sees the failure
    because their database is already migrated.

    Fixed here rather than by hand-editing each migration, so it cannot recur.
    """
    if type_ == "type" and obj.__class__.__module__.startswith("pgvector"):
        autogen_context.imports.add("import pgvector.sqlalchemy")
    return False  # fall through to Alembic's default rendering


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        include_object=_include_object,
        render_item=_render_item,
        compare_type=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_object=_include_object,
            render_item=_render_item,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
