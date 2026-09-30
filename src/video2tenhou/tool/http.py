"""Shared local HTTP transport: origin checks, bounded bodies and responses."""

import json
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"


class LocalHandler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(STATIC), **kw)

    def end_headers(self):
        """Avoid stale job responses and evidence after review edits."""
        self.send_header(
            "Cache-Control", "no-store"
        )  # the page and images change between runs
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "frame-ancestors 'self'")
        super().end_headers()

    def log_message(self, fmt, *args):  # quieter: no line per API call
        """Keep periodic API polling out of the terminal access log."""
        if args and "/api/" in str(args[0]):
            return
        super().log_message(fmt, *args)

    def _json(self, obj, code=200):
        if code >= 400 and self.command == "POST":
            self._drain_rejected_body()
        from ..engine.decode import sanitize

        data = json.dumps(sanitize(obj), ensure_ascii=False).encode()
        self._bytes(data, "application/json; charset=utf-8", code)

    def _bytes(
        self,
        data: bytes,
        content_type: str,
        code: int = 200,
        *,
        headers: dict | None = None,
    ):
        """Send a complete response with the shared cache and security headers."""
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _read_request_bytes(self, size: int) -> bytes:
        """Account for consumed bytes so an error never reads the body twice."""
        data = self.rfile.read(size)
        self._request_bytes_read = getattr(self, "_request_bytes_read", 0) + len(data)
        return data

    def _drain_rejected_body(self) -> None:
        """Finish small rejected POST bodies before closing the TCP connection.

        Closing with unread payload can turn a useful 4xx response into a
        connection reset on Windows. Never wait indefinitely for a dishonest
        Content-Length or drain an entire rejected video upload: both bytes
        and waiting time are bounded, independently of endpoint body limits.
        """
        self.close_connection = True
        try:
            remaining = int(self.headers.get("Content-Length", "0")) - getattr(
                self, "_request_bytes_read", 0
            )
        except ValueError:
            return
        if not 0 < remaining <= 1024 * 1024:
            return
        previous_timeout = self.connection.gettimeout()
        try:
            deadline = time.monotonic() + 0.5
            while remaining > 0:
                wait = deadline - time.monotonic()
                if wait <= 0:
                    break
                self.connection.settimeout(min(previous_timeout or wait, wait))
                chunk = self.rfile.read1(min(65536, remaining))
                self._request_bytes_read = getattr(
                    self, "_request_bytes_read", 0
                ) + len(chunk)
                if not chunk:
                    break
                remaining -= len(chunk)
        except (OSError, TimeoutError):
            pass  # The response is still attempted; this connection is closed.
        finally:
            self.connection.settimeout(previous_timeout)

    def _local_request(self):
        """Reject browser cross-origin writes and DNS-rebinding hosts."""
        host = self.headers.get("Host", "")
        if not self._host_ok():
            return False
        origin = self.headers.get("Origin")
        return (origin is None or origin == f"http://{host}") and self.headers.get(
            "X-Video2Tenhou"
        ) == "1"

    def _host_ok(self) -> bool:
        return self.headers.get("Host") in (
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        )
