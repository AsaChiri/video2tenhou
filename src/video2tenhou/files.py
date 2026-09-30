"""Shared file operations for provenance and complete, atomic publications."""

import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    """Hash file contents with the standard library's streaming implementation."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def _atomic_text_file(path: Path, *, retry_windows: bool):
    """Sync a sibling temporary file before replacing the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            pending = Path(stream.name)
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        for attempt, delay in enumerate((0.05, 0.1, 0.2, 0.4, 0)):
            try:
                pending.replace(path)
                break
            except OSError as exc:
                if (
                    not retry_windows
                    or os.name != "nt"
                    or getattr(exc, "winerror", None) not in (5, 32, 33)
                    or attempt == 4
                ):
                    raise
                time.sleep(delay)
    finally:
        if pending is not None:
            try:
                pending.unlink(missing_ok=True)
            except OSError:
                pass  # Retain the original write/replacement failure.


def atomic_write_text(path: Path, content: str, *, retry_windows: bool = False) -> None:
    """Publish UTF-8 text completely or retain the previous destination.

    Optional Windows sharing-violation retries are bounded to 750 ms. This
    synchronizes file contents, but does not promise directory durability after
    power loss. Readers see either the old or the new complete file.
    """
    with _atomic_text_file(path, retry_windows=retry_windows) as stream:
        stream.write(content)


def atomic_write_json(
    path: Path, value, *, retry_windows: bool = False, **options
) -> None:
    """Publish JSON with the same failure guarantees as atomic_write_text."""
    with _atomic_text_file(path, retry_windows=retry_windows) as stream:
        json.dump(value, stream, ensure_ascii=False, **options)
