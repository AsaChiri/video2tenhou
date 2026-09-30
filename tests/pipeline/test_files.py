# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""File publication preserves the old artifact on any incomplete write."""

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from video2tenhou import files

if TYPE_CHECKING:
    from typing import TextIO


def test_streaming_hash_and_nested_unicode_json(tmp_path: "Path") -> None:
    """Verify streaming hash and nested unicode json."""
    path = tmp_path / "nested" / "evidence.json"
    files.atomic_write_json(path, {"tile": "東"}, indent=1)
    assert json.loads(path.read_text(encoding="utf-8")) == {"tile": "東"}
    assert files.sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("failure", ["serialization", "sync", "replacement"])
def test_failed_publication_preserves_existing_contents_and_cleans_up(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", failure: str
) -> None:
    """Verify failed publication preserves existing contents and cleans up."""
    path = tmp_path / "evidence.json"
    path.write_text('{"complete":true}', encoding="utf-8")
    before = path.read_bytes()

    def fail(*_unused_args: object, **_unused_kwargs: object) -> None:
        msg = "interrupted write"
        raise OSError(msg)

    if failure == "serialization":

        def partial(
            value: "object", stream: "TextIO", **_unused_kwargs: object
        ) -> None:
            stream.write('{"partial":')
            fail()

        monkeypatch.setattr(files.json, "dump", partial)
    elif failure == "sync":
        monkeypatch.setattr(files.os, "fsync", fail)
    else:
        monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="interrupted write"):
        files.atomic_write_json(path, {"complete": False})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_text_publication_replaces_complete_file(tmp_path: "Path") -> None:
    """Verify text publication replaces complete file."""
    path = tmp_path / "rows.jsonl"
    files.atomic_write_text(path, '"old"\n')
    files.atomic_write_text(path, '"new"\n"rows"\n')
    assert path.read_text() == '"new"\n"rows"\n'
