# Observability

The unified API and Celery worker emit JSON logs with `request_id`, `session_id`,
`task_id`, `trace_id`, and `span_id`. LangSmith shows the agent's model and tool
runs; the same `request_id` is attached as trace metadata. OpenTelemetry exports
service traces and metrics to the optional local collector. No user messages or
model responses are added to OpenTelemetry spans or metric labels.

## Start the local viewers

From the repository root, add these values to the ignored `storage/.env`:

```dotenv
OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
WORKER_OTEL_ENDPOINT=http://otel-collector:4318
GRAFANA_PASSWORD=choose-a-unique-password
```

Choose your own Grafana password. The host endpoint is for the API process
running on your Mac; the worker uses the Docker service name.

On a Mac using Colima, start Colima and point the standalone `docker-compose`
command at its socket in each terminal where you run Compose:

```bash
colima start
export DOCKER_HOST="unix://${HOME}/.colima/default/docker.sock"
```

Start the viewers, then rebuild/restart the worker:

```bash
docker-compose --profile observability --env-file storage/.env up -d \
  jaeger otel-collector prometheus grafana
docker-compose --profile observability --env-file storage/.env up -d --build celery-worker
```

Stop any API process already listening on `API_PORT`, then restart the unified
API in another terminal:

```bash
uv run --project . --with-editable ./agent --env-file storage/.env python -m api.app
```

Then send a chat message and upload an image. The response header
`X-Request-ID` is the value to search for in API logs and LangSmith metadata.
The upload response also has a Celery `task_id`; its worker logs carry both IDs.

The API terminal prints one JSON log record per line. Search for
`intent_decision` to see whether the intent agent called a tool or replied
directly, and for `stylist_inference_start`, `stylist_inference_complete`, or
`stylist_inference_failed` to follow the stylist. Match records by
`request_id` and `session_id`. To follow image-worker logs:

```bash
docker-compose --env-file storage/.env logs -f --tail=100 celery-worker
```

| Viewer | Local address | What to inspect |
| --- | --- | --- |
| Jaeger | http://127.0.0.1:16686 | `fashion-api` and `fashion-worker` traces |
| Prometheus | http://127.0.0.1:9093 | Query `fashion_http_requests_total` |
| Grafana | http://127.0.0.1:3000 | Fashion Assistant dashboard, including intent and stylist metrics |

Grafana's default username is `admin`; use the password from `storage/.env`.
Metrics appear after the first export and scrape, usually within 20 seconds.
Jaeger uses temporary storage in this local setup, so traces disappear when
its container restarts. Prometheus data uses a Docker volume.

Keep `AGENT_LOG_CONTENT=false` for normal operation. Set it to `true` only
while examining local model payloads; those logs may contain personal data.
