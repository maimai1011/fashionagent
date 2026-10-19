# Fashion Agents

This is the `uv` package for the fashion assistant's agent, persistence, and
wardrobe-search modules. The main Python package is `agent.agents`.

## Setup

From this directory:

```bash
uv sync
```

To run wardrobe searches with Milvus and CLIP embeddings:

```bash
uv sync --extra catalog
```

To include PostgreSQL-backed LangGraph checkpoints:

```bash
uv sync --extra postgres
```

## Development

Run the unit tests from this directory:

```bash
uv run pytest -c pyproject.toml ../tests/unit
```

Run lint checks with:

```bash
uv run ruff check . ../tests
```

Wardrobe retrieval reads the current user's completed uploads from PostgreSQL,
filters garment type and explicit constraints, searches their vectors in Milvus,
then ranks matches. `WARDROBE_MIN_SIMILARITY` sets the minimum CLIP cosine
similarity (default `0.20`). Lower scores return no match.

From the repository root, evaluate saved, labeled search results with:

```bash
uv run --project . --with-editable ./agent \
  python -m agent.search.evaluate examples.jsonl --k 5
```

Each JSONL line needs `requested_category`, `requested_name`,
`relevant_item_ids`, and `results` in returned order. Each result has `item_id`,
`category`, and `name`. The script reports mean Precision@K over queries with
relevant items and the category mismatch rate over returned items.
