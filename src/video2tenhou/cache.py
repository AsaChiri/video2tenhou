# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Persisted content digests of source recordings, keyed by file identity."""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path

from .files import atomic_write_json, sha256_file
from .paths import DATA_DIR

STORE = DATA_DIR / "work" / "source-digests.json"


def _file_key(path: Path) -> list[int]:
    st = path.stat()
    return [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev]


def _records() -> dict:
    try:
        records = json.loads(STORE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return {}
    return records if isinstance(records, dict) else {}


def source_identity(path: str | Path) -> str:
    """Return the SHA-256 of a file's contents, hashing only when its identity changed.

    The digest is recorded under the resolved path with size, modification and
    change times, file index and device. A matching record is returned without
    reading the file. Contents rewritten in place with all of those restored are
    outside this contract. A record lost to a concurrent writer, or one that cannot
    be saved, only means hashing again. The first call after a change can take
    minutes for a broadcast, so call this outside locks.
    """
    path = Path(path).resolve()
    key = _file_key(path)
    record = _records().get(str(path))
    if isinstance(record, dict) and record.get("file") == key:
        return record["sha256"]
    digest = sha256_file(path)
    if _file_key(path) == key:  # never record a digest of a file that was changing
        records = _records()
        records[str(path)] = {"file": key, "sha256": digest}
        with suppress(OSError):
            atomic_write_json(STORE, records, retry_windows=True)
    return digest
