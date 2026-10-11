# Fashion Multi-Agent System

A production-oriented hierarchical fashion assistant built with LangChain's
dynamic `create_agent` tool loop, LangGraph checkpointing and interrupts, an
independent Stylist Agent, and a validated Milvus catalog search service.

## Architecture

```text
User/API
  └─ Intent Agent (dynamic ReAct tool selection)
       ├─ ask_user                 → LangGraph interrupt
       ├─ consult_stylist          → independent Stylist Agent + skill
       ├─ search_products          → embedding + existing Milvus catalog
       ├─ restore_previous_outfit  → deterministic snapshot rollback
       └─ select_product           → verify and store a catalog selection
```

There is no keyword router or fixed specialist sequence. The Intent Agent's LLM
chooses tools and may call multiple tools in a turn. Tool code owns validated,
deterministic state changes; the LLM cannot write product or outfit state
directly. Catalog fields, prices, stock, IDs, and images only come from Milvus.

## Main modules

- `agents/intent`: coordinator prompt, state, tools, and `create_agent` factory.
- `agents/stylist`: independent structured-output stylist and preservation checks.
- `skills/fashion_styling`: complete styling guidance in a single skill file.
- `search`: embedding provider, Milvus schema validation, filtering, and ranking.
- `persistence`: in-memory development and PostgreSQL production checkpoints.
- `api`: authenticated FastAPI interface.
- `ui`: Gradio chat, outfit summary, catalog gallery, and missing-image marker.
- `raw_data`: existing user registration and raw image upload functions.
- `setup`: one-time PostgreSQL, Milvus image collection, and MinIO initialization.

## Prerequisites

- Python 3.11+
- vLLM serving Qwen2.5-3B-Instruct as `fashion-llm`
- An existing Milvus **product catalog** collection
- PostgreSQL for production checkpoints
- MinIO/PostgreSQL for the existing upload pipeline

The application never creates, drops, or alters the product catalog collection.
At startup it validates configured field names, embedding dimension, vector
index, and similarity metric. A mismatch fails with a configuration error.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Export the environment before running commands:

```bash
set -a
source .env
set +a
```

The default embedding values are examples only. `EMBEDDING_MODEL`,
`EMBEDDING_DIMENSION`, `EMBEDDING_NORMALIZE`, `EMBEDDING_QUERY_PREFIX`, and
`MILVUS_METRIC_TYPE` must exactly match product indexing.

## vLLM

Configure the server to expose an OpenAI-compatible endpoint and enable tool
calling for the Qwen tokenizer/model configuration. The application defaults to:

```text
VLLM_URL=http://localhost:8002/v1
VLLM_MODEL=fashion-llm
VLLM_API_KEY=EMPTY
```

Example server command (adjust dtype/device flags for your hardware):

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --served-model-name fashion-llm \
  --host 0.0.0.0 \
  --port 8002 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Qwen2.5 uses Hermes-style tool calls in vLLM. Both auto tool choice and the
parser are required for the Intent Agent's dynamic tool loop.

Model weights and vLLM launch flags are deployment-specific and are not stored
in this repository.

## Storage initialization

For a new storage database only:

```bash
python -m setup
```

This creates the upload tables, image feature collection, and MinIO bucket
idempotently. It is separate from the existing Milvus product catalog. If upload
tables already predate the inference result columns, apply a database migration;
`CREATE TABLE IF NOT EXISTS` does not alter an existing table.

## Run

Start the API:

```bash
python main.py
```

Start the Python UI in another shell:

```bash
python -m ui.gradio_app
```

Example API request:

```bash
curl -X POST http://127.0.0.1:8080/v1/chat \
  -H "Authorization: Bearer $APP_API_KEY" \
  -H "X-User-ID: user-123" \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"thread-1","message":"Tạo outfit đi làm thanh lịch","resume":false}'
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

Checkpoint setup runs automatically during API startup. For multiple users, set
`APP_USER_TOKENS_JSON` to a JSON object mapping each accepted `X-User-ID` to its
bearer token; otherwise `APP_API_KEY` is the single shared credential. Checkpoint
thread IDs are namespaced as `<authenticated-user-id>::<thread-id>`. Per-session async locks
prevent concurrent state mutations within one API process. For multi-instance
deployments, add a distributed lock keyed by the same session identifier.

Outfit history stores deep snapshots of the outfit and associated product state.
Revisions keep unchanged item matches and invalidate only modified items.

## ngrok and public UI safety

Set `ENABLE_NGROK=true` and provide `NGROK_AUTHTOKEN` through the environment.
Never commit the token. Configure `UI_AUTH_USER`, `UI_AUTH_PASSWORD`, and
`UI_USER_TOKEN` before exposing the UI publicly. Place an ingress/API gateway rate
limit in front of multi-instance production deployments.

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

Do not enable them until the configured vLLM model and compatible Milvus catalog
are available.
