# Web Demo

A FastAPI web layer on top of the existing gather pipeline. Enter a restaurant name and city to see the outdoor seating prediction, confidence, reasoning, evidence snippets, source metadata, gather errors, and (when using the LangGraph backend) node-by-node execution details.

Results are returned as in-memory `GatherRunResult` objects — the web layer never reads CSV output files.

## Security model

- **API keys stay server-side.** `ANTHROPIC_API_KEY` and `GOOGLE_PLACES_API_KEY` are loaded from environment variables or Cloud Run Secret Manager. They are never rendered in HTML, JSON, or JavaScript.
- **No secrets in responses.** Errors are sanitized via `web/security.py`; stack traces and raw settings are never exposed to clients.
- **No filesystem exposure.** Only `/static` serves CSS from `src/restaurant_agent/web/static/`. The app does not mount `/data`, `/outputs`, `/docs`, or the project root. There are no CSV download links or file-path inputs.
- **Validated inputs.** Name and city are required with length limits (200 / 100). Backend must be `pipeline` or `graph`. `live_google` is a boolean flag only — no arbitrary URLs or crawl endpoints.
- **One request, one gather.** Each HTTP request runs a single gather operation using the configured server-side fixtures path (not user-supplied).
- **OpenAPI disabled.** `/docs` and `/openapi.json` are not exposed in the demo app.

## Local run

Requires Python 3.12+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[web,dev]"
cp .env.example .env   # optional; dry-run needs no keys

uvicorn restaurant_agent.web.app:app --reload --port 8080
```

Open [http://127.0.0.1:8080/demo](http://127.0.0.1:8080/demo).

Default form settings (dry-run + static fixtures) require **no API keys** and make **no external network calls**.

## Docker run

```bash
docker build -t restaurant-agent-web .
docker run --rm -p 8080:8080 restaurant-agent-web
```

For live Claude or Google Places, pass secrets at runtime — never bake them into the image:

```bash
docker run --rm -p 8080:8080 \
  -e ANTHROPIC_API_KEY="your-key" \
  -e GOOGLE_PLACES_API_KEY="your-google-key" \
  restaurant-agent-web
```

The image includes `data/evidence_fixtures` for server-side static adapters only; fixtures are not served over HTTP.

## Cloud Run deployment

### 1. Store secrets in Secret Manager

```bash
echo -n "your-anthropic-key" | gcloud secrets create anthropic-key --data-file=-
echo -n "your-google-places-key" | gcloud secrets create google-places-key --data-file=-
```

Do **not** put secrets in the Dockerfile, `.env` committed to git, or container environment files checked into the repo.

### 2. Deploy

```bash
gcloud run deploy restaurant-agent-web \
  --source . \
  --region europe-west1 \
  --allow-unauthenticated \
  --set-secrets=ANTHROPIC_API_KEY=anthropic-key:latest,GOOGLE_PLACES_API_KEY=google-places-key:latest \
  --set-env-vars=CONFIDENCE_THRESHOLD=0.6,DEFAULT_FIXTURES_PATH=/app/data/evidence_fixtures \
  --timeout=60 \
  --concurrency=4 \
  --min-instances=0
```

For a non-public deployment, omit `--allow-unauthenticated` and use [Identity-Aware Proxy (IAP)](https://cloud.google.com/iap) or Cloud Run authenticated invokers instead.

### 3. GCP security recommendations

| Practice | Why |
|----------|-----|
| Secret Manager for API keys | Keys are injected at runtime, not stored in images or source |
| No secrets in Docker image | Images are often shared; layers are inspectable |
| Restrict Google API key to Places API only | Limits blast radius if a key leaks |
| Minimal Cloud Run service account | Grant only `secretmanager.secretAccessor` (if using secrets) |
| IAP / authenticated access for production | Public demo is fine for dry-run; live modes cost money |
| Match Cloud Run `--timeout` to `ANTHROPIC_REQUEST_TIMEOUT` | Prevents hung requests |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Landing page |
| `GET` | `/health` | Liveness probe (`{"status":"ok"}`) |
| `GET` | `/demo` | Gather form |
| `POST` | `/demo` | Submit form; returns HTML results |
| `POST` | `/api/gather` | JSON API |

## API schema

### Request (`POST /api/gather`)

```json
{
  "name": "The River Cafe",
  "city": "London",
  "backend": "graph",
  "dry_run": true,
  "live_google": false
}
```

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `name` | string | required | max 200 chars |
| `city` | string | required | max 100 chars |
| `backend` | `"pipeline"` \| `"graph"` | `"pipeline"` | |
| `dry_run` | boolean | `true` | Uses deterministic local LLM |
| `live_google` | boolean | `false` | Replaces maps adapter with Google Places |

Single-result pages show confidence, validation status, route, evidence counts, and source reliability mix. Precision, recall, and F1 are dataset-level evaluation metrics and are reported for labelled batch runs (CLI `evaluate` / `gather` over `data/restaurants.csv`), not individual searches.

### Response (`200 OK`)

Returns a `GatherRunResult` JSON object:

```json
{
  "restaurant_id": "the-river-cafe-london",
  "name": "The River Cafe",
  "city": "London",
  "backend": "graph",
  "prediction": "yes",
  "confidence": 0.85,
  "reasoning": "...",
  "evidence": ["..."],
  "gathered_evidence": [
    {
      "source_type": "search",
      "source_name": "static_search",
      "url": "https://example.com",
      "snippet": "...",
      "reliability": "medium"
    }
  ],
  "source_results": [
    {"source_name": "static_search", "evidence_count": 2, "error": null}
  ],
  "gather_errors": [],
  "validation_status": "ok",
  "needs_review": false,
  "route": "success",
  "error": null,
  "graph_trace": [
    {"node": "gather_search", "update": {"source_results": ["..."]}}
  ]
}
```

### Error response (`400` / `500`)

```json
{"message": "Live Claude API is not configured on the server."}
```

## Environment variables

Same as the CLI — see [`.env.example`](../.env.example).

| Variable | Required when | Default |
|----------|---------------|---------|
| `ANTHROPIC_API_KEY` | Live Claude (`dry_run=false`) | — |
| `GOOGLE_PLACES_API_KEY` | `live_google=true` | — |
| `DEFAULT_FIXTURES_PATH` | No | `data/evidence_fixtures` |
| `CONFIDENCE_THRESHOLD` | No | `0.6` |
| `ANTHROPIC_REQUEST_TIMEOUT` | No | `60` |
| `LOG_LEVEL` | No | `INFO` |

The CLI is unchanged; use `restaurant-agent gather ...` for CSV-based workflows.
