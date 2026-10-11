"""Run the Fashion Multi-Agent API."""

import uvicorn

from fashion_agent.config import settings


if __name__ == "__main__":
    uvicorn.run("api.app:app", host=settings.api_host, port=settings.api_port, reload=False)
