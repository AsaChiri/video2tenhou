# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""The single built frontend is complete, local and served by the API router."""

from __future__ import annotations

from contextlib import closing
from html.parser import HTMLParser
from http import HTTPStatus
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest

from tests.web.server import studio_server
from video2tenhou import cli
from video2tenhou.tool.http import IMMUTABLE, STATIC


class AssetReferences(HTMLParser):
    """Collect script and asset references from built frontend HTML."""

    def __init__(self) -> None:
        """Initialize HTML parsing and empty asset collections."""
        super().__init__()
        self.urls = []
        self.scripts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record script attributes and linked asset URLs."""
        attributes = dict(attrs)
        if tag == "script":
            self.scripts.append(attributes)
            self.urls.append(attributes.get("src", ""))
        elif tag == "link" and attributes.get("rel") in ("stylesheet", "modulepreload"):
            self.urls.append(attributes["href"])


def test_built_application_and_assets_are_served_offline(tmp_path: Path) -> None:
    with studio_server(tmp_path) as server, httpx.Client(trust_env=False) as client:
        base = server.url
        response = client.get(base).raise_for_status()
        parser = AssetReferences()
        parser.feed(response.text)
        assert len(parser.scripts) == 1
        assert parser.scripts[0]["type"] == "module"
        assert any(url.endswith(".css") for url in parser.urls)
        for url in parser.urls:
            assert url.startswith("/assets/")
            assert (STATIC / url.lstrip("/")).is_file()
            response = client.get(base + url)
            assert response.status_code == HTTPStatus.OK
            assert response.headers["Cache-Control"] == IMMUTABLE
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            expected_type = "javascript" if url.endswith(".js") else "text/css"
            assert expected_type in response.headers["Content-Type"]
            assert response.content
        assert {path.name for path in (STATIC / "assets").iterdir()} == {
            Path(url).name for url in parser.urls
        }
        for path in (
            "/studio.html",
            "/api/hands",
            "/assets/../index.html",
            "/assets/source.map",
        ):
            # HTTPX normalizes dot segments. Send the raw traversal path so this
            # assertion still exercises the server's boundary checks.
            address = urlparse(base)
            assert address.hostname == "127.0.0.1"
            with closing(
                HTTPConnection(address.hostname, address.port, timeout=10)
            ) as raw:
                raw.request("GET", path)
                assert raw.getresponse().status == HTTPStatus.NOT_FOUND


def test_web_is_the_only_browser_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["review", "recording.mp4"])
    assert error.value.code == 2
    assert "invalid choice: 'review'" in capsys.readouterr().err
