"""Loopback-only browser workspace, including streamed video import and exports.

Project review routes reuse the mature review handler but resolve state from
the URL on every request. There is no mutable 'currently selected video' on
the server: two tabs cannot write facts into each other's recording.
"""
from __future__ import annotations

import json
import re
import threading
import uuid
import webbrowser
from pathlib import Path
from urllib.parse import unquote, urlparse

from .server import Handler, _Server
from .workflow import VIDEO_SUFFIXES, Workspace


class AppHandler(Handler):
    """Serve the studio, scoped review APIs and explicitly allowlisted exports."""

    @property
    def workspace(self) -> Workspace:
        """Access the server-owned workspace, never process-global project state."""
        return self.server.workspace

    @property
    def state(self):
        """Resolve review state from this request's project route."""
        return self.workspace.review_state(self.project_key)

    def _host_ok(self) -> bool:
        return self.headers.get("Host") in (f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}")

    def _review_route(self, path: str) -> bool:
        match = re.fullmatch(r"/review/([a-f0-9]{32})/(.*)", path)
        if not match:
            return False
        self.project_key = match[1]
        self.workspace.project(self.project_key)
        query = urlparse(self.path).query
        self.path = "/" + match[2] + ("?" + query if query else "")
        return True

    def do_GET(self):
        """Read project status, stream exports or dispatch project review requests."""
        if not self._host_ok():
            return self._json({"error": "Use the localhost address shown when starting the app."}, 403)
        path = urlparse(self.path).path
        try:
            if path == "/api/workspace":
                return self._json({"setup": self.workspace.setup(), "sources": self.workspace.sources(),
                                   "projects": [self.workspace.snapshot(k) for k in self.workspace.projects]})
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})", path)
            if match:
                return self._json(self.workspace.snapshot(match[1]))
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/results", path)
            if match:
                return self._json(self.workspace.results(match[1]))
            match = re.fullmatch(r"/exports/([a-f0-9]{32})/([^/]+)", path)
            if match:
                file = self.workspace.artifact(match[1], unquote(match[2]))
                data = file.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8" if file.suffix == ".html" else
                                 "application/json; charset=utf-8" if file.suffix == ".json" else "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                if file.suffix != ".html":
                    self.send_header("Content-Disposition", f'attachment; filename="{file.name}"')
                self.end_headers()
                self.wfile.write(data)
                return
            if self._review_route(path):
                if self.path.startswith("/api/read") and any(p.get("job", {}).get("running") for p in self.workspace.projects.values()):
                    return self._json({"error": "Analysis is running. Tile labeling becomes available when it finishes."}, 409)
                if self.path.startswith("/api/") or urlparse(self.path).path in ("/", "/index.html"):
                    return super().do_GET()
                return self._json({"error": "Page not found."}, 404)
            if path == "/":
                self.path = "/studio.html"
            elif not re.fullmatch(r"/tiles/[A-Za-z0-9_-]+\.(?:svg|md)", path):
                return self._json({"error": "Page not found."}, 404)
            return super().do_GET()
        except (KeyError, FileNotFoundError) as exc:
            return self._json({"error": str(exc)}, 404)
        except ValueError as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:
            return self._json({"error": str(exc)}, 500)

    def do_POST(self):
        """Accept same-origin project actions and bounded JSON or streamed uploads."""
        if not self._local_request():
            return self._json({"error": "Only requests from this local app are accepted."}, 403)
        path = urlparse(self.path).path
        try:
            if self._review_route(path):
                with self.workspace.lock:
                    if any(p.get("job", {}).get("running") for p in self.workspace.projects.values()):
                        return self._json({"error": "Analysis is running. Wait until it finishes before changing review data."}, 409)
                    if any(any(job.get("running") for job in state.jobs.values())
                           for state in self.workspace.states.values()):
                        return self._json({"error": "A review job is running. Wait for it to finish first."}, 409)
                    return super().do_POST()
            size = int(self.headers.get("Content-Length", "0"))
            if path == "/api/upload":
                return self._upload(size)
            if not 0 <= size <= 65536:
                return self._json({"error": "Request body is too large."}, 413)
            body = json.loads(self._read_request_bytes(size) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("Expected a JSON object.")
            if path == "/api/projects":
                return self._json(self.workspace.create(body), 201)
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/settings", path)
            if match:
                return self._json(self.workspace.update(match[1], body))
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/(prepare|analyze)", path)
            if match:
                return self._json(self.workspace.start(match[1], match[2]), 202)
            return self._json({"error": "Unknown action."}, 404)
        except (KeyError, FileNotFoundError) as exc:
            return self._json({"error": str(exc)}, 404)
        except (ValueError, OSError) as exc:
            return self._json({"error": str(exc)}, 400)
        except Exception as exc:
            return self._json({"error": str(exc)}, 500)

    def _upload(self, size: int):
        filename = unquote(self.headers.get("X-Filename", ""))
        error, status = None, 400
        if not 0 < size <= 100 * 1024 ** 3:
            error, status = "Choose a nonempty recording smaller than 100 GB.", 413
        if Path(filename).name != filename or "\\" in filename or "/" in filename:
            error = "Invalid upload filename."
        suffix = Path(filename).suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            error = "Choose an MP4, MKV, MOV, WebM, AVI or M4V recording."
        if error:
            return self._json({"error": error}, status)
        stem = re.sub(r"[^\w .-]", "_", Path(filename).stem).strip(" .")[:80] or "recording"
        folder = self.workspace.root / "samples"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{stem}_{uuid.uuid4().hex[:8]}{suffix}"
        partial = path.with_suffix(path.suffix + ".upload")
        try:
            with partial.open("xb") as output:
                remaining = size
                while remaining:
                    chunk = self._read_request_bytes(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("The upload was interrupted. Choose the file again to retry.")
                    output.write(chunk)
                    remaining -= len(chunk)
            partial.replace(path)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return self._json({"path": str(path), "name": path.name}, 201)


def make_server(root: Path, port: int = 8765, *, runner=None) -> _Server:
    """Construct a loopback server; port zero is useful for isolated HTTP tests."""
    server = _Server(("127.0.0.1", port), AppHandler)
    server.workspace = Workspace(root, runner=runner)
    return server


def serve_workspace(root: Path, port: int = 8765, *, open_browser: bool = True) -> None:
    """Open the local studio and serve until interrupted; no external listener."""
    try:
        server = make_server(root, port)
    except OSError as exc:
        raise SystemExit(f"Cannot start the app on port {port}: {exc}. Try --port with another number.") from exc
    url = f"http://localhost:{server.server_port}"
    print(f"video2tenhou: {url}\nKeep this window open while a recording is running.")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
