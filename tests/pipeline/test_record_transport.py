# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Score queries retain explicit request and failure contracts through HTTPX."""

import json

import httpx
import pytest

from video2tenhou import record


def test_query_sends_json_and_honors_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queries use the fixed endpoint and return only the GraphQL data payload."""
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": {"game": {"id": 42}}})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(record.httpx, "post", client.post)
        assert record.gql("query Game", {"id": 42}, timeout=7) == {"game": {"id": 42}}
    request = requests[0]
    assert request.url == record.ENDPOINT
    assert request.method == "POST"
    assert request.headers["User-Agent"] == "video2tenhou"
    assert request.headers["Content-Type"] == "application/json"
    assert json.loads(request.content) == {
        "query": "query Game",
        "variables": {"id": 42},
    }
    assert set(request.extensions["timeout"].values()) == {7}


@pytest.mark.parametrize(
    ("status", "payload", "error"),
    [
        (503, b"unavailable", httpx.HTTPStatusError),
        (200, b'{"errors": ["query rejected"]}', RuntimeError),
        (200, b"not JSON", json.JSONDecodeError),
    ],
)
def test_query_errors_propagate(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    payload: bytes,
    error: type[Exception],
) -> None:
    """HTTP and GraphQL failures cannot become empty or successful site records."""
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, content=payload))
    ) as client:
        monkeypatch.setattr(record.httpx, "post", client.post)
        with pytest.raises(error):
            record.gql("query Game", {})
