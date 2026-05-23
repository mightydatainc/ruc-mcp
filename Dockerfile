# Use Python 3.12 as a stability baseline for the scientific stack.
# Newer interpreters (3.13/3.14) can have temporary wheel/support lag in
# transitive native dependencies, which may force source builds on slim images.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN mkdir -p /app/logs \
    && pip install .[datasci]

CMD ["python", "-m", "ruc_mcp"]