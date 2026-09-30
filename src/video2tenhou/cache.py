# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared source provenance for pixel-derived caches."""

from __future__ import annotations

from pathlib import Path

from .files import sha256_file

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
        _SOURCE_CACHE[key] = sha256_file(path)
    return _SOURCE_CACHE[key]
