FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/evidence_fixtures ./data/evidence_fixtures

RUN pip install --no-cache-dir ".[web]"

ENV PORT=8080
ENV DEFAULT_FIXTURES_PATH=/app/data/evidence_fixtures

EXPOSE 8080

CMD ["sh", "-c", "uvicorn restaurant_agent.web.app:app --host 0.0.0.0 --port ${PORT:-8080}"]
