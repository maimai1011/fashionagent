"""Verify HTTP error responses keep their correlation ID."""

import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.app import request_context, unhandled_error


class ObservabilityApiTests(unittest.TestCase):
    def test_unhandled_error_returns_request_id(self) -> None:
        app = FastAPI()
        app.middleware("http")(request_context)
        app.add_exception_handler(Exception, unhandled_error)

        @app.get("/boom")
        def boom():
            raise RuntimeError("test error")

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/boom", headers={"X-Request-ID": "probe-123"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.headers["X-Request-ID"], "probe-123")
