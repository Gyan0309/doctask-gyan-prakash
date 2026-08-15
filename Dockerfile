FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, in their own layer — editing source must not re-resolve pip.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

COPY pyproject.toml alembic.ini ./
COPY src/ ./src/
COPY migrations/ ./migrations/
COPY rules/ ./rules/
COPY tests/ ./tests/
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN pip install --no-deps -e . && chmod +x /usr/local/bin/entrypoint.sh

# Non-root. Nothing here needs privilege, and a container that runs as root by
# default is a habit that costs nothing to break now and a lot to break later.
RUN useradd --create-home --uid 10001 ledger && \
    mkdir -p /data/inbox && chown -R ledger:ledger /app /data
USER ledger

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "ledger.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
