# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Source archives ship tracked files and the built UI, never local data."""

from __future__ import annotations

import pytest

from tools import check_source_release as release

TRACKED = {"pyproject.toml", "tests/data/fixture.json.gz"}
BUILT = {
    release.BUILT_INDEX,
    release.BUILT_ASSETS + "index-1.js",
    release.BUILT_ASSETS + "index-1.css",
}


def test_archive_inventory_matches_tracked_files_and_built_ui() -> None:
    """The sdist metadata and generated assets are the only untracked members."""
    release.check_contents(TRACKED | BUILT | {"PKG-INFO"}, TRACKED)


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (TRACKED, "built frontend"),
        (BUILT | {"pyproject.toml"}, "differs from tracked"),
        (TRACKED | BUILT | {"labels/recording/answers.jsonl"}, "differs from tracked"),
    ],
)
def test_archive_inventory_rejects_unbuilt_missing_or_private_files(
    names: set[str], message: str
) -> None:
    """A missing fixture or an included local answer file fails the release check."""
    with pytest.raises(ValueError, match=message):
        release.check_contents(names, TRACKED)
