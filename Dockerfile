FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PORT=8080 \
    DEFAULT_FIXTURES_PATH=/app/data/evidence_fixtures \
    LOG_LEVEL=INFO

WORKDIR /app

RUN adduser --disabled-password --gecos "" --uid 10001 appuser

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/evidence_fixtures ./data/evidence_fixtures

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[web]" && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

CMD ["sh", "-c", "uvicorn restaurant_agent.web.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
