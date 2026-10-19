# Fashion Multi-Agent System

A production-oriented hierarchical fashion assistant built with LangChain's
dynamic `create_agent` tool loop, LangGraph checkpointing and interrupts, an
independent Stylist Agent, and search over processed wardrobe uploads.

## Architecture

```text
User/API
  └─ Intent Agent (dynamic ReAct tool selection)
       ├─ ask_user                 → LangGraph interrupt
       ├─ consult_stylist          → independent Stylist Agent + wardrobe search
       ├─ search_wardrobe          → CLIP text query + image_features vectors
       ├─ count_wardrobe           → PostgreSQL count of completed uploads
       ├─ restore_previous_outfit  → deterministic snapshot rollback
       └─ select_wardrobe_item     → verify and store an uploaded item selection
```

There is no keyword router or fixed specialist sequence. The Intent Agent's LLM
chooses tools and may call multiple tools in a turn. Tool code owns validated,
deterministic state changes; the LLM cannot write product or outfit state
directly. Every successful outfit creation or revision searches completed
uploads for the current user. Milvus holds the CLIP image vectors; PostgreSQL
holds item metadata, and MinIO holds the processed images. The historical
`product_matches` field in the API represents wardrobe matches for compatibility.

## Main modules

- `agents/intent`: coordinator prompt, state, tools, and `create_agent` factory.
- `agents/stylist`: independent structured-output stylist and preservation checks.
- `skills/fashion_styling`: complete styling guidance in a single skill file.
- `search`: CLIP text queries, Milvus vector search, PostgreSQL joins, and ranking.
- `persistence`: in-memory development and PostgreSQL production checkpoints.
- `api`: FastAPI interface for profiles, wardrobe uploads, and chat.
- `ui`: chat, outfit summary, and wardrobe image display.
- `raw_data`: existing user registration and raw image upload functions.
- `setup`: one-time PostgreSQL, Milvus image collection, and MinIO initialization.

## Prerequisites

- Python 3.11+
- vLLM serving Qwen2.5-3B-Instruct as `fashion-llm`
- The `image_features` Milvus collection created by the storage initializer
- The Celery image worker for processing new wardrobe uploads
- PostgreSQL for production checkpoints
- MinIO/PostgreSQL for the existing upload pipeline

The chat API validates the `image_features` collection at startup. Only uploads
with `completed` status and a matching PostgreSQL `wardrobe_items` row are
eligible for search. Searches are limited to the current user before querying
Milvus.

## Installation

From the repository root:

```bash
cp storage/.env.example storage/.env
uv sync --locked
```

The bundled Compose file currently uses `postgres` for PostgreSQL and
`miniosecret` for MinIO. Set `POSTGRES_PASSWORD=postgres` and
`MINIO_ROOT_PASSWORD=miniosecret` in `storage/.env` before starting it.

Export the environment before running commands:

```bash
set -a
source storage/.env
set +a
```

The image worker and API use `openai/clip-vit-base-patch32` for uploaded image
vectors and chat text queries. `MILVUS_EMBEDDING_DIM` must match its 512-value
vectors. Chat search embeds text in the API process using that CLIP model
and searches stored image vectors in Milvus. RabbitMQ is used for new image
uploads; it is not needed to search already processed wardrobe items.

## Local LLM (vLLM)

The agents connect to a Qwen or Llama model served by your local vLLM process.
`ChatOpenAI` is used only as a client for vLLM's compatible HTTP protocol;
requests are sent to the host and port configured in `storage/.env`, not to
OpenAI's hosted service. Set `VLLM_HOST`, `VLLM_PORT`, `VLLM_MODEL`, and
`VLLM_API_KEY` there before starting the server.

Example server command (adjust dtype/device flags for your hardware):

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --served-model-name fashion-llm \
  --host 127.0.0.1 \
  --port "${VLLM_PORT}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Qwen2.5 uses Hermes-style tool calls in vLLM. Both auto tool choice and the
parser are required for the Intent Agent's dynamic tool loop. The configured
`VLLM_MODEL` must exactly match `--served-model-name`.

## Storage initialization

Start the local infrastructure from the repository root:

```bash
docker-compose --env-file storage/.env up -d postgres minio etcd milvus rabbitmq
docker-compose --env-file storage/.env ps
```

The PostgreSQL container is ready when its status is `healthy`. Its data is
kept in the `postgres-data` named volume when the container is stopped or
recreated.

For a new storage database only:

```bash
uv run --project storage --env-file storage/.env python -m storage.setup
```

This creates the upload tables, image feature collection, and MinIO bucket
idempotently. Re-run
the initializer after updating an older upload database; it adds any missing
wardrobe result columns without dropping existing data.

## Public API

From the repository root, start the unified profile, wardrobe, and chat API:

```bash
set -a
source storage/.env
set +a

uv run --project . --with-editable ./agent \
  --env-file storage/.env python -m api.app
```

Register a user's profile before uploading clothing:

```bash
curl -X POST "http://${API_HOST}:${API_PORT}/v1/user-profile" \
  -F "user_id=user-123" \
  -F "email=trang@example.com" \
  -F "age=25" \
  -F "gender=female"
```

Upload one clothing image for that user:

```bash
curl -X POST "http://${API_HOST}:${API_PORT}/v1/wardrobe" \
  -H "X-User-ID: user-123" \
  -F "image=@/absolute/path/to/shirt.png;type=image/png"
```

The upload returns `queued` and a `result_url` after the original image is
saved to MinIO and the Celery task is published. Start the inference worker with:

```bash
docker-compose --env-file storage/.env up -d --build celery-worker
```

The worker runs `inference/image_inference/ImageProcessing` with `ClothingModel`.
It removes the background, crops the garment, detects its color and type, and
generates a CLIP embedding. The transparent PNG is saved under `processed/`
in MinIO, the embedding goes to the `image_features` Milvus collection, and
`wardrobe_items` receives the color, type, embedding reference, and `completed`
status. The first processed image downloads model weights into the persistent
`image_model_cache` Docker volume. Set `REMBG_MODEL` in `storage/.env` to change
the background-removal model; the default is `u2net`.

Use the `item_id` returned by the upload to check the processing result:

```bash
curl "http://${API_HOST}:${API_PORT}/v1/wardrobe/<item_id>" \
  -H "X-User-ID: user-123"
```

List all of that user's clothing uploads, newest first:

```bash
curl "http://${API_HOST}:${API_PORT}/v1/wardrobe" \
  -H "X-User-ID: user-123"
```

Each item includes its processing `status`, `original_filename`, `dominant_color`,
`clothing_type`, `image_url`, `processed_image_url`, and any `error_message`.
`image_url` serves the processed image when available, or the original upload
while processing is pending or failed. Send the same `X-User-ID` header when
fetching either image URL. Processing results and `processed_image_url` are
`null` until available.

When `status` is `completed`, the response contains `dominant_color`,
`clothing_type`, and a `processed_image_url`. Fetch that PNG with the same
`X-User-ID` header:

```bash
curl "http://${API_HOST}:${API_PORT}/v1/wardrobe/<item_id>/image" \
  -H "X-User-ID: user-123" -o processed.png
```

Inspect recent results with:

```bash
docker-compose --env-file storage/.env \
  exec postgres psql -U postgres -d postgres \
  -c "SELECT item_id, status, dominant_color, clothing_type, embedding_ref, error_message FROM wardrobe_items ORDER BY created_at DESC LIMIT 10;"
```

Chat uses the same `X-User-ID` header at `POST /v1/chat`. The profile is
stored in PostgreSQL. Clothing images are stored in MinIO and their object
metadata is recorded in PostgreSQL. Configure the maximum accepted image size
with `UPLOAD_MAX_BYTES`; the default is 10 MiB.

## Run

Start the unified API with the uv command in the **Public API** section above.

Start the Streamlit UI in another shell:

```bash
uv run --project ui --env-file storage/.env streamlit run ui/streamlit_app.py
```

Example API request:

```bash
curl -X POST "http://${API_HOST}:${API_PORT}/v1/chat" \
  -H "X-User-ID: user-123" \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"thread-1","user_input":"Tạo outfit đi làm thanh lịch","resume":false}'
```

When the response contains `"interrupted": true`, submit the clarification
answer with the same user/thread and `"resume": true`. Ordinary feedback after
a completed response always uses `"resume": false`.

## Checkpoints and session isolation

Local development uses `InMemorySaver`. Production uses the async PostgreSQL
checkpointer:

```text
CHECKPOINT_BACKEND=postgres
CHECKPOINT_DATABASE_URL=postgresql://...
LANGGRAPH_STRICT_MSGPACK=true
```

Checkpoint setup runs automatically during API startup. Checkpoint thread IDs
are namespaced as `<X-User-ID>::<thread-id>`. Per-session async locks prevent
concurrent state mutations within one API process. For multi-instance
deployments, add a distributed lock keyed by the same session identifier.

Outfit history stores deep snapshots of the outfit and associated product state.
Revisions keep unchanged item matches and invalidate only modified items.

## ngrok and public UI safety

For a private Streamlit demo, tunnel only local port `8501` and enforce ngrok
authentication. Keep the API and storage bound to localhost. The current API
trusts the caller-supplied `X-User-ID`, and Streamlit exposes editable User ID
and API URL fields. Multi-user public access requires server-enforced
authentication and ownership checks.

## Tests and linting

```bash
ruff check .
pytest -q tests/unit
```

Unit tests mock vLLM and Milvus. Live integration tests are opt-in and were not
designed to silently contact services:

```bash
RUN_INTEGRATION_TESTS=true pytest -q -m integration
```

Do not enable them until the configured vLLM model and Milvus image collection
are available.
