"""HTTP endpoint for conversations with the chat agent."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from pydantic import AliasChoices, BaseModel, Field

from agent.agents.chat.application import ChatResponse, FashionApplication
from agent.agents.chat.config import settings
from agent.agents.intent.agent import create_intent_agent
from agent.agents.stylist.agent import StylistAgent
from agent.persistence.checkpoint import checkpointer_context
from agent.search.wardrobe import WardrobeSearchService
from api.dependencies import request_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["chat"])


@asynccontextmanager
async def chat_lifespan(app: FastAPI):
    """Create the shared chat graph for standalone and unified API servers."""
    wardrobe_search = WardrobeSearchService()
    try:
        if os.getenv("VALIDATE_MILVUS_ON_STARTUP", "true").lower() == "true":
            await wardrobe_search.validate()
        async with checkpointer_context() as checkpointer:
            graph = create_intent_agent(
                checkpointer,
                stylist=StylistAgent(),
                wardrobe_search=wardrobe_search,
            )
            app.state.chat = FashionApplication(graph)
            yield
    finally:
        wardrobe_search.close()


class ChatRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=200)
    user_input: str = Field(
        min_length=1,
        max_length=10_000,
        validation_alias=AliasChoices("user_input", "message"),
    )
    resume: bool = False


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    request: Request,
    user_id: Annotated[str, Depends(request_user_id)],
) -> ChatResponse:
    logger.info("function_start function=chat")
    try:
        return await request.app.state.chat.chat(
            user_id=user_id,
            thread_id=body.thread_id,
            message=body.user_input,
            resume=body.resume,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "chat_request_failed user_id=%s thread_id=%s", user_id, body.thread_id
        )
        raise HTTPException(
            status_code=503,
            detail="Fashion service is temporarily unavailable",
        ) from exc


def create_app() -> FastAPI:
    """Build an API that serves only the chat endpoint."""
    chat_app = FastAPI(title="Fashion Chat API", version="0.1.0", lifespan=chat_lifespan)
    chat_app.include_router(router)
    return chat_app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
