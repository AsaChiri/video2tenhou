# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Keep the persisted source-digest store out of the user's workspace."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from video2tenhou import cache


@pytest.fixture(autouse=True, scope="session")
def _source_digest_store(tmp_path_factory: pytest.TempPathFactory) -> Iterator:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            cache, "STORE", tmp_path_factory.mktemp("cache") / "source-digests.json"
        )
        yield
