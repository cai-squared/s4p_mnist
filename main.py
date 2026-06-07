"""Cloud Functions Gen 2 entry point (gcloud --entry-point=handler)."""

from __future__ import annotations

import functions_framework
from flask import Response
from starlette.testclient import TestClient

from api.main import app

_client = TestClient(app)


@functions_framework.http
def handler(request):  # type: ignore[no-untyped-def]
    """Forward Cloud Functions HTTP requests to the FastAPI app."""
    path = request.path or "/"
    query = request.query_string.decode("utf-8") if request.query_string else ""
    url = f"{path}?{query}" if query else path
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }
    response = _client.request(
        method=request.method,
        url=url,
        headers=headers,
        content=request.get_data(),
    )
    excluded = {"content-length", "transfer-encoding", "connection"}
    response_headers = [
        (key, value)
        for key, value in response.headers.items()
        if key.lower() not in excluded
    ]
    return Response(response.content, status=response.status_code, headers=response_headers)


__all__ = ["handler"]
