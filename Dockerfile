# --------------------------------------------------------------------------
# Stage 1 — build the review page.
#
# Node exists only here. The runtime image gets the compiled bundle and nothing
# else, so the shipped container carries no node, no node_modules, and no npm.
# --------------------------------------------------------------------------
FROM node:22-alpine AS web

WORKDIR /web

# Manifest first so a source edit does not re-resolve npm.
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

COPY web/ ./
RUN npm run build


# --------------------------------------------------------------------------
# Stage 2 — the application.
# --------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependencies first, in their own layer — editing source must not re-resolve pip.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Packages live flat at the repository root, so they are copied as they are and
# imported directly. PYTHONPATH=/app is what makes that work; there is no build step.
COPY pytest.ini pyproject.toml alembic.ini conftest.py ./
COPY main.py ./
COPY api/ ./api/
COPY database/ ./database/
COPY domain/ ./domain/
COPY integrations/ ./integrations/
COPY mcp_server.py ./
COPY models/ ./models/
COPY providers/ ./providers/
COPY services/ ./services/
COPY utils/ ./utils/
COPY migrations/ ./migrations/
COPY rules/ ./rules/
COPY tests/ ./tests/

# The compiled review page. main.py mounts this at / when it exists.
COPY --from=web /web/dist ./web/dist

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

# Strip carriage returns before making it executable.
#
# `.gitattributes` already forces LF on checkout, and this makes the image immune even
# if the file arrives with CRLF some other way — a zip download, a copy off a Windows
# share, an editor that helpfully "fixed" it. Without one of these two layers the
# shebang becomes `#!/usr/bin/env bash\r`, the container looks for a program named
# "bash\r", and the API exits 127 with a message that names bash rather than the file.
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh \
    && chmod +x /usr/local/bin/entrypoint.sh

# Non-root. Nothing here needs privilege, and a container that runs as root by
# default is a habit that costs nothing to break now and a lot to break later.
RUN useradd --create-home --uid 10001 ledger && \
    mkdir -p /data/inbox && chown -R ledger:ledger /app /data
USER ledger

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
