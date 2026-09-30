"""The single built frontend is complete, local and served by the API router."""

import threading
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from video2tenhou import cli
from video2tenhou.tool.http import STATIC
from video2tenhou.tool.server import make_server


class AssetReferences(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
        self.scripts = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        if tag == "script":
            self.scripts.append(attributes)
            self.urls.append(attributes.get("src", ""))
        elif tag == "link" and attributes.get("rel") in ("stylesheet", "modulepreload"):
            self.urls.append(attributes["href"])


def test_built_application_and_assets_are_served_offline(tmp_path):
    server = make_server(tmp_path, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            parser = AssetReferences()
            parser.feed(response.read().decode())
        assert len(parser.scripts) == 1
        assert parser.scripts[0]["type"] == "module"
        assert any(url.endswith(".css") for url in parser.urls)
        for url in parser.urls:
            assert url.startswith("/assets/")
            assert (STATIC / url.lstrip("/")).is_file()
            with urlopen(base + url) as response:
                assert response.status == 200
                assert response.headers["Cache-Control"] == "no-store"
                assert response.headers["X-Content-Type-Options"] == "nosniff"
                expected_type = "javascript" if url.endswith(".js") else "text/css"
                assert expected_type in response.headers["Content-Type"]
                assert response.read()
        assert {path.name for path in (STATIC / "assets").iterdir()} == {
            Path(url).name for url in parser.urls
        }
        for path in (
            "/studio.html",
            "/api/hands",
            "/assets/../index.html",
            "/assets/source.map",
        ):
            with pytest.raises(HTTPError) as error:
                urlopen(base + path)
            assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_web_is_the_only_browser_command(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["review", "recording.mp4"])
    assert error.value.code == 2
    assert "invalid choice: 'review'" in capsys.readouterr().err
