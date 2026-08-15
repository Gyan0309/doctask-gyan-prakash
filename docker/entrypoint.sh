#!/usr/bin/env bash
# Migrations run here, not in a separate manual step, so `docker compose up` on a
# fresh clone yields a working system rather than a working system minus its schema.
set -euo pipefail

echo "[entrypoint] applying migrations..."
alembic upgrade head
echo "[entrypoint] migrations at head."

exec "$@"
