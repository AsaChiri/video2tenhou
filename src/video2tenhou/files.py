# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared file operations for provenance and complete, atomic publications."""

import hashlib
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import IO

WINDOWS_REPLACE_BUSY_ERRORS = (5, 32, 33)


@overload
def sanitize(value: dict) -> dict: ...


@overload
def sanitize(value: object) -> object: ...


def sanitize(value: object) -> object:
    """Replace nonfinite floats recursively before writing JSON for browsers."""
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return 1e9 if value > 0 else -1e9
        return value
    if isinstance(value, dict):
        return {key: sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    return value


def sha256_file(path: str | Path) -> str:
    """Hash file contents with the standard library's streaming implementation."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def _atomic_text_file(path: Path, *, retry_windows: bool) -> "Iterator[IO[str]]":
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
        for delay in (0.05, 0.1, 0.2, 0.4, 0):
            try:
                pending.replace(path)
                break
            except OSError as exc:
                if (
                    not retry_windows
                    or os.name != "nt"
                    or getattr(exc, "winerror", None) not in WINDOWS_REPLACE_BUSY_ERRORS
                    or not delay
                ):
                    raise
                time.sleep(delay)
    finally:
        if pending is not None:
            # Cleanup must retain the original write/replacement failure.
            with suppress(OSError):
                pending.unlink(missing_ok=True)


def atomic_write_text(path: Path, content: str, *, retry_windows: bool = False) -> None:
    """Publish UTF-8 text completely or retain the previous destination.

    Optional Windows sharing-violation retries are bounded to 750 ms. This
    synchronizes file contents, but does not promise directory durability after
    power loss. Readers see either the old or the new complete file.
    """
    with _atomic_text_file(path, retry_windows=retry_windows) as stream:
        stream.write(content)


def atomic_write_json(
    path: Path,
    value: object,
    *,
    retry_windows: bool = False,
    indent: int | None = None,
    sort_keys: bool = False,
) -> None:
    """Publish JSON with the same failure guarantees as atomic_write_text."""
    with _atomic_text_file(path, retry_windows=retry_windows) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=indent, sort_keys=sort_keys)
