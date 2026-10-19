"""FastAPI entrypoint for the fashion assistant."""

import logging
import time
from collections import defaultdict, deque
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent.agents.chat.config import settings
from agent.agents.chat.logging import configure_logging, request_id_var
from agent.observability.metrics import record_http
from agent.observability.telemetry import instrument_fastapi
from api.agent_chat import chat_lifespan
from api.agent_chat import router as chat_router
from api.user_profile import router as user_profile_router
from api.wardrobe import router as wardrobe_router

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


app = FastAPI(title="Fashion Multi-Agent API", version="0.1.0", lifespan=chat_lifespan)
app.include_router(chat_router)
app.include_router(user_profile_router)
app.include_router(wardrobe_router)
_request_times: defaultdict[str, deque[float]] = defaultdict(deque)


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    identity = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _request_times[identity]
    while window and window[0] <= now - 60:
        window.popleft()
    if len(window) >= settings.api_rate_limit_per_minute:
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded"})
    window.append(now)
    return await call_next(request)


@app.middleware("http")
async def request_context(request: Request, call_next):
    supplied_id = request.headers.get("X-Request-ID", "")
    request_id = (
        supplied_id if supplied_id and len(supplied_id) <= 128
        and all(char.isalnum() or char in "-_." for char in supplied_id)
        else str(uuid4())
    )
    request.state.request_id = request_id
    request_token = request_id_var.set(request_id)
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception:
        logger.exception("http_request_failed method=%s path=%s", request.method, request.url.path)
        raise
    finally:
        route = request.scope.get("route")
        route_name = getattr(route, "path", "unmatched")
        latency_seconds = time.perf_counter() - started
        record_http(request.method, route_name, status_code, latency_seconds)
        logger.info(
            "http_request method=%s route=%s status_code=%d latency_ms=%.2f",
            request.method, route_name, status_code, latency_seconds * 1000,
        )
        request_id_var.reset(request_token)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_error(request: Request, _exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers={"X-Request-ID": request_id} if request_id else None,
    )


instrument_fastapi(app)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
