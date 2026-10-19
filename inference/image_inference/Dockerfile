
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/models/huggingface \
    U2NET_HOME=/models/u2net

WORKDIR /app

ENV PYTHONPATH=/app

# System libraries for OpenCV and inference
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY inference/image_inference/requirements-worker.txt /tmp/requirements-worker.txt

RUN pip install --no-cache-dir \
        torch==2.6.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        -r /tmp/requirements-worker.txt

# Copy source code
COPY inference/ /app/inference/
COPY storage/ /app/storage/
COPY agent/__init__.py /app/agent/__init__.py
COPY agent/observability/ /app/agent/observability/

# Non-root user
RUN useradd --uid 10001 --create-home \
        --shell /usr/sbin/nologin imageworker \
    && mkdir -p /models/huggingface /models/u2net \
    && chown -R 10001:10001 /models /app

USER imageworker

# Start Celery Worker

CMD ["celery", "-A", "inference.image_inference.tasks:celery_app", "worker", "--loglevel=info", "--concurrency=1"]
