# Streamlit fashion assistant

The Streamlit app has Profile, Wardrobe, and Chat views. All three use the
unified API (`api.app`), not the standalone `api.agent_chat` server. Ports and
service settings come from `storage/.env`.

From the repository root, start the infrastructure:

```bash
docker-compose --env-file storage/.env up -d postgres minio etcd milvus rabbitmq
```

For a new database, initialize storage once:

```bash
uv run --project storage --env-file storage/.env python -m storage.setup
```

Start the image-processing worker:

```bash
docker-compose --env-file storage/.env up -d --build celery-worker
```

In a separate terminal, run the unified API:

```bash
uv run --project . --with-editable ./agent \
  --env-file storage/.env python -m api.app
```

The API validates the `image_features` Milvus collection at startup. Chat
searches completed uploads owned by the current user, using CLIP text embeddings
from the running Celery worker. Chat also needs the configured local vLLM server.

In another terminal, run Streamlit:

```bash
uv run --project ui --env-file storage/.env streamlit run ui/streamlit_app.py
```

Open the address given by `STREAMLIT_SERVER_ADDRESS` and
`STREAMLIT_SERVER_PORT` in `storage/.env`. Enter a stable User ID, create a
profile, and upload clothing in Wardrobe. Queued and processing items update
automatically every five seconds; "Refresh now" is also available. The
wardrobe panel keeps IDs for this browser session; use "Find an existing
upload by item ID" after a reload. Chat sends that same User ID with a
conversation ID and handles clarification replies.

## Inspect chat traces

For API and Celery logs, OpenTelemetry traces, metrics, and a local dashboard,
see the [observability setup](../agent/observability/README.md).

To inspect model calls, selected tools, tool arguments, and the final answer,
create a LangSmith API key. In the ignored `storage/.env`, set
`LANGSMITH_API_KEY` to that key and change `LANGSMITH_TRACING` to `true`.
`LANGSMITH_PROJECT=fashion-chat` groups these runs in one project. Restart the
unified API, send a new chat message, then open the `fashion-chat` project under
**Tracing Projects** in LangSmith. Earlier chat turns cannot be recovered.
Traces include prompts and responses and are sent to LangSmith.

The API also logs `intent_model_start`, `intent_decision`, and
`stylist_inference_start`/`stylist_inference_complete` as JSON lines in its
terminal. To inspect the actual user message, intent tool arguments, and full
stylist request/result locally, set `AGENT_LOG_CONTENT=true` in `storage/.env`
and restart the API. These detailed logs can contain personal data. Set it
back to `false` after debugging. Use `request_id` and `session_id` to follow
one chat turn across the intent and stylist entries.
If the API writes to `api.log`, follow those entries with:

```bash
tail -f api.log | rg --line-buffered 'intent_|stylist_'
```
