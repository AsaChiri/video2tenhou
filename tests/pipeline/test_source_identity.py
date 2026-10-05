# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Source digests persist across processes and follow file identity."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from video2tenhou import cache


def _counted(monkeypatch: pytest.MonkeyPatch) -> list:
    calls = []
    digest = cache.sha256_file

    def counted(path: Path) -> str:
        calls.append(path)
        return digest(path)

    monkeypatch.setattr(cache, "sha256_file", counted)
    return calls


def test_digest_is_persisted_and_reused_without_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorded identity answers later calls, including from a fresh process."""
    monkeypatch.setattr(cache, "STORE", tmp_path / "store.json")
    calls = _counted(monkeypatch)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video-a")
    expected = hashlib.sha256(b"video-a").hexdigest()
    assert cache.source_identity(source) == expected
    assert cache.source_identity(tmp_path / "." / "video.mp4") == expected
    saved = json.loads((tmp_path / "store.json").read_text(encoding="utf-8"))
    assert saved[str(source.resolve())]["sha256"] == expected
    assert len(calls) == 1


def test_replaced_file_with_same_size_and_mtime_is_hashed_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new file at the same path has a new identity even with restored times."""
    monkeypatch.setattr(cache, "STORE", tmp_path / "store.json")
    calls = _counted(monkeypatch)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video-a")
    stat = source.stat()
    cache.source_identity(source)
    replacement = tmp_path / "video.new"
    replacement.write_bytes(b"video-b")
    os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    replacement.replace(source)
    assert cache.source_identity(source) == hashlib.sha256(b"video-b").hexdigest()
    assert len(calls) == 2


def test_unreadable_or_unwritable_store_only_costs_a_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A damaged store is ignored and a failed save does not fail the caller."""
    store = tmp_path / "store.json"
    store.write_text('{"interrupted', encoding="utf-8")
    monkeypatch.setattr(cache, "STORE", store)
    calls = _counted(monkeypatch)
    source = tmp_path / "video.mp4"
    source.write_bytes(b"video-a")

    def denied(*_unused_args: object, **_unused_kwargs: object) -> None:
        raise PermissionError

    monkeypatch.setattr(cache, "atomic_write_json", denied)
    assert cache.source_identity(source) == hashlib.sha256(b"video-a").hexdigest()
    assert cache.source_identity(source) == hashlib.sha256(b"video-a").hexdigest()
    assert len(calls) == 2
