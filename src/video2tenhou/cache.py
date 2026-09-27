"""Shared source provenance for pixel-derived caches."""
from __future__ import annotations

import hashlib
from pathlib import Path

_SOURCE_CACHE: dict[tuple, str] = {}


def source_identity(path: str | Path, *, refresh: bool = False) -> str:
    """Hash source contents, reusing the digest within one unchanged analysis.

    Stage entry points refresh the digest even when size and timestamps match.
    Dense evidence windows reuse it to avoid hashing an entire broadcast for
    every crop. Videos must remain unchanged while an analysis is running.
    """
    path = Path(path).resolve()
    st = path.stat()
    key = (str(path), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    if refresh or key not in _SOURCE_CACHE:
        with path.open("rb") as stream:
            _SOURCE_CACHE[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    return _SOURCE_CACHE[key]
