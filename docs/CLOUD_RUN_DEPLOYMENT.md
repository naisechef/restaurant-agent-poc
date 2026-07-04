# Cloud Run Deployment

Deploy the Restaurant Agent web demo to Google Cloud Run. The recommended path uses **`gcloud run deploy --source .`**, which builds the container with Cloud Build and deploys it in one step. Use a **local Docker build** first to validate the image before deploying.

Region used in examples: **`europe-north1`**.

## Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (`gcloud`) authenticated to your project
- [Docker](https://docs.docker.com/get-docker/) for local image validation
- A GCP project with billing enabled
- IAM permission to enable APIs, deploy Cloud Run services, and manage Secret Manager

Set your project:

```bash
gcloud config set project YOUR_PROJECT_ID
```

## Enable required APIs

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com
```

Cloud Build (used by `--source .`) stores built images in Artifact Registry automatically. You do not need a separate image push step for source-based deploys.

## Local Docker smoke test

Validate the image locally before deploying:

```bash
docker build -t restaurant-agent-web .
docker run --rm -p 8080:8080 restaurant-agent-web
```

In another terminal:

```bash
# Liveness — always 200 when the process is up (no API keys required)
curl -i http://127.0.0.1:8080/health

# Readiness — 503 without secrets; 200 when configured (see below)
curl -i http://127.0.0.1:8080/ready

# Protected paths must not be exposed
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/data/evidence_fixtures/search.json
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/outputs/gather_results.csv
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/docs/WEB_DEMO.md
```

Expected: `/health` returns `200`, `/ready` returns `503` without API keys, protected paths return `404`.

Optional live-API run (never bake secrets into the image):

```bash
docker run --rm -p 8080:8080 \
  -e ANTHROPIC_API_KEY="your-key" \
  -e GOOGLE_PLACES_API_KEY="your-google-key" \
  restaurant-agent-web
```

Then `curl http://127.0.0.1:8080/ready` should return `200` with `"status":"ready"`.

## Secret Manager setup

Create secrets for the two API keys. Use the same names as the environment variables for clarity:

```bash
echo -n "your-anthropic-key" | gcloud secrets create ANTHROPIC_API_KEY --data-file=-
echo -n "your-google-places-key" | gcloud secrets create GOOGLE_PLACES_API_KEY --data-file=-
```

Do **not** put secrets in the Dockerfile, committed `.env` files, or container layers.

### Grant the Cloud Run service account access

After the first deploy, note the service account (default: `PROJECT_NUMBER-compute@developer.gserviceaccount.com`) or use a dedicated runtime service account.

```bash
PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

for SECRET in ANTHROPIC_API_KEY GOOGLE_PLACES_API_KEY; do
  gcloud secrets add-iam-policy-binding "$SECRET" \
    --member="serviceAccount:${RUNTIME_SA}" \
    --role="roles/secretmanager.secretAccessor"
done
```

For production, create a dedicated service account with only `secretmanager.secretAccessor` on these secrets.

## Recommended Cloud Run settings

| Setting | Recommended | Why |
|---------|-------------|-----|
| `--memory` | `1Gi` | LangGraph orchestration, evidence merging, and response serialization benefit from headroom; `512Mi` may OOM under concurrent gathers. |
| `--cpu` | `1` | One vCPU matches low concurrency; LLM calls are I/O-bound but local graph steps use CPU for parsing and validation. |
| `--timeout` | `60` | Aligns with `ANTHROPIC_REQUEST_TIMEOUT` (default 60s). A gather waits on Claude and optionally Google Places; shorter timeouts cause false 504s. |
| `--concurrency` | `4` | **Lower than Cloud Run’s default (80)** because each request runs a full LangGraph gather plus an Anthropic API call. High concurrency on one instance stacks memory usage and upstream rate limits. `4` keeps one instance predictable for demo/interview traffic. |
| `--min-instances` | `0` | Scale to zero when idle — appropriate for demos and cost control. |
| `--max-instances` | `5` | Caps burst cost and upstream API usage for a public demo. Increase for sustained load. |
| `--region` | `europe-north1` | Deploy close to your users and API policy constraints. |

## Deployment modes

### Public demo (interview / portfolio)

Use `--allow-unauthenticated` so anyone with the URL can open the demo. Appropriate when the service is a portfolio piece and you accept anonymous traffic on the landing page.

```bash
gcloud run deploy restaurant-agent-web \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --set-secrets=ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,GOOGLE_PLACES_API_KEY=GOOGLE_PLACES_API_KEY:latest \
  --set-env-vars=DEFAULT_FIXTURES_PATH=/app/data/evidence_fixtures,LOG_LEVEL=INFO,LIVE_GOOGLE_REQUIRED=true \
  --memory=1Gi \
  --cpu=1 \
  --timeout=60 \
  --concurrency=4 \
  --min-instances=0 \
  --max-instances=5
```

Set `LIVE_GOOGLE_REQUIRED=true` when the demo offers the live Google Places checkbox and you want `/ready` to require `GOOGLE_PLACES_API_KEY`.

### Private deployment

Omit `--allow-unauthenticated` and restrict access:

**Cloud Run IAM** — grant `roles/run.invoker` only to trusted principals:

```bash
gcloud run deploy restaurant-agent-web \
  --source . \
  --region europe-north1 \
  --no-allow-unauthenticated \
  --set-secrets=ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,GOOGLE_PLACES_API_KEY=GOOGLE_PLACES_API_KEY:latest \
  --set-env-vars=DEFAULT_FIXTURES_PATH=/app/data/evidence_fixtures,LOG_LEVEL=INFO \
  --memory=1Gi \
  --cpu=1 \
  --timeout=60 \
  --concurrency=4 \
  --min-instances=0 \
  --max-instances=5
```

Invoke with an identity token:

```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  https://SERVICE_URL/health
```

**Identity-Aware Proxy (IAP)** — for browser access behind Google sign-in, put the service behind a load balancer with IAP enabled. See [Cloud IAP documentation](https://cloud.google.com/iap).

## Update and rollback

Redeploy with the same command after code changes; Cloud Run creates a new revision.

Rollback traffic to a previous revision:

```bash
gcloud run revisions list --service restaurant-agent-web --region europe-north1

gcloud run services update-traffic restaurant-agent-web \
  --region europe-north1 \
  --to-revisions=restaurant-agent-web-00042-abc=100
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness — always fast, no external calls, `200` when process is alive |
| `GET` | `/ready` | Readiness — local config checks only (API keys, fixtures path); `503` if not ready |
| `GET` | `/` | Landing page |
| `POST` | `/` | Form search |
| `POST` | `/api/gather` | JSON API |

All responses include `X-Request-ID` for log correlation in Cloud Logging.

### Readiness checks (`GET /ready`)

No external network calls. Verifies locally:

- `ANTHROPIC_API_KEY` is set (required for live Claude gathers)
- `GOOGLE_PLACES_API_KEY` is set when `LIVE_GOOGLE_REQUIRED=true`
- Evidence fixtures directory exists at `DEFAULT_FIXTURES_PATH`

Example ready response (`200`):

```json
{
  "status": "ready",
  "checks": {
    "anthropic_api_key": {"ok": true, "required": true},
    "google_places_api_key": {"ok": true, "required": true},
    "fixtures_path": {"ok": true, "required": true, "path": "/app/data/evidence_fixtures"}
  }
}
```

## Deployed smoke-test checklist

After deploy, capture the service URL:

```bash
SERVICE_URL=$(gcloud run services describe restaurant-agent-web \
  --region europe-north1 \
  --format='value(status.url)')
echo "$SERVICE_URL"
```

### HTTP checks

```bash
curl -i "$SERVICE_URL/health"
curl -i "$SERVICE_URL/ready"

curl -s -o /dev/null -w "%{http_code}\n" "$SERVICE_URL/data/evidence_fixtures/search.json"
curl -s -o /dev/null -w "%{http_code}\n" "$SERVICE_URL/openapi.json"
```

Expected: `/health` → `200`, `/ready` → `200` (with secrets), protected paths → `404`.

For a private service, add `-H "Authorization: Bearer $(gcloud auth print-identity-token)"`.

### Browser test

1. Open `$SERVICE_URL/` in a browser.
2. Search **The River Cafe**, **London**, enable **live Google** if offered.
3. Verify UI sections: **Prediction**, **Decision Verification**, **Graph execution trace**, **location panel**.
4. View page source — confirm no `ANTHROPIC_API_KEY`, `GOOGLE_PLACES_API_KEY`, `api_key`, or stack traces.

### Logs

```bash
gcloud run services logs read restaurant-agent-web --region europe-north1 --limit 50
```

Confirm structured JSON logs include `request_id`, and gather logs include `restaurant`, `city`, `stage`, and `duration_ms`. No secret values in log lines.

Correlate a request using `X-Request-ID` from the response:

```bash
gcloud logging read \
  'resource.type="cloud_run_revision" AND jsonPayload.request_id="REQUEST_ID_HERE"' \
  --limit 20 \
  --format=json
```

## Security reminders

| Practice | Why |
|----------|-----|
| Secret Manager for API keys | Keys injected at runtime, not in images or git |
| No secrets in Docker image | Image layers are inspectable and often cached |
| Restrict Google API key to Places API | Limits blast radius if a key leaks |
| Minimal service account permissions | Grant only `secretmanager.secretAccessor` where needed |
| IAP / IAM for non-demo deployments | Live modes incur API cost |
| Low concurrency | Prevents one instance from running too many concurrent LLM calls |

## Future production improvements

Documentation only — not implemented in this PoC:

- **Cloud Monitoring dashboards** — request latency, 5xx rate, instance count, Anthropic latency
- **Cloud Trace** — distributed traces across gather stages
- **Error Reporting** — aggregate unhandled exceptions with `request_id` grouping
- **Budget alerts** — notify when Cloud Run or Places API spend exceeds thresholds
- **Cloud Armor** — WAF and DDoS protection in front of a load balancer
- **Rate limiting** — per-IP or per-user quotas on `/api/gather`
- **Custom domain** — branded URL with managed TLS
- **GitHub Actions CI/CD** — pytest on PR, `docker build` validation, automated `gcloud run deploy` on merge

## Related documentation

- [WEB_DEMO.md](WEB_DEMO.md) — local run, endpoints, API schema
- [README.md](../README.md) — project overview and environment variables
