"""Port configuration stays environment-backed across components."""

from unittest.mock import Mock

from agent.agents.chat.config import milvus_port as catalog_milvus_port
from api.config import ApiSettings
from inference.image_inference import tasks
from storage.setup import milvus, minio, postgres
from ui.config import default_api_url


def test_storage_ports_follow_environment(monkeypatch):
    monkeypatch.setenv("POSTGRES_PORT", "15432")
    monkeypatch.setenv("MINIO_PORT", "19000")
    monkeypatch.setenv("MILVUS_PORT", "19531")
    monkeypatch.setenv("RABBITMQ_PORT", "15672")
    monkeypatch.setenv("MINIO_HOST", "localhost")
    monkeypatch.setenv("MINIO_SCHEME", "http")
    monkeypatch.setenv("RABBITMQ_HOST", "localhost")
    monkeypatch.delenv("MINIO_ENDPOINT", raising=False)
    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)

    connect_postgres = Mock()
    connect_milvus = Mock()
    monkeypatch.setattr(postgres.psycopg2, "connect", connect_postgres)
    monkeypatch.setattr(milvus.connections, "connect", connect_milvus)

    postgres.get_connection()
    milvus.connect_milvus()

    assert connect_postgres.call_args.kwargs["port"] == "15432"
    assert minio.get_minio_endpoint() == "http://localhost:19000"
    assert connect_milvus.call_args.kwargs["port"] == "19531"
    assert tasks._broker_url() == "amqp://guest:guest@localhost:15672//"
    assert catalog_milvus_port() == "19531"


def test_api_port_and_ui_url_follow_environment(monkeypatch):
    monkeypatch.setenv("API_HOST", "127.0.0.1")
    monkeypatch.setenv("API_PORT", "18002")
    monkeypatch.delenv("API_BASE_URL", raising=False)

    assert ApiSettings.from_env().port == 18002
    assert default_api_url() == "http://127.0.0.1:18002"
