
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from search.schema import OutfitPlan, SearchResponse
from search.service import FashionSearchService


@asynccontextmanager
async def lifespan(app: FastAPI):

    service = FashionSearchService(
        milvus_uri=os.environ["MILVUS_URI"],
        milvus_token=os.getenv("MILVUS_TOKEN") or None,
        collection_name=os.getenv(
            "MILVUS_COLLECTION",
            "fashion_products",
        ),
        embedding_model=os.environ["EMBEDDING_MODEL"],
        vector_field=os.getenv(
            "MILVUS_VECTOR_FIELD",
            "embedding",
        ),
        top_k=5,
        candidate_limit=30,
    )

    try:
        service.initialize()
        app.state.search_service = service
        yield
    finally:
        service.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health/live")
def liveness():
    return {"status": "alive"}


@app.post(
    "/search/outfit",
    response_model=SearchResponse,
)
async def search_outfit(
    outfit: OutfitPlan,
    request: Request,
):
    service = request.app.state.search_service

    try:
        return await service.search_outfit(outfit)

    except Exception as exc:
        # Log the actual exception on the server,
        # but do not expose DB internals to clients.
        import logging
        logging.getLogger(__name__).exception(
            "Outfit search failed"
        )

        raise HTTPException(
            status_code=503,
            detail="Product search temporarily unavailable",
        ) from exc
