# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Unreadable storage is an operational failure, not a reason to redo recognition."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.recognition import RecognitionStub
from video2tenhou import calibfit, calm, observe, read, timeline
from video2tenhou.layout import Calibration


@pytest.mark.parametrize(
    "stage", ["calm", "plate", "timing", "observations", "intervals"]
)
@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_storage_errors_propagate_without_regenerating_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    error_type: type[OSError],
) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    work = tmp_path / "work"
    work.mkdir()
    cal = Calibration.load("pml")
    hand = {"hand": 0, "t_start": 0.0, "t_end": 1.0}
    paths = {
        "calm": "calm_scores.npz",
        "plate": "plate.meta.json",
        "timing": "table-timing.json",
        "observations": "obs/provenance/00.json",
        "intervals": "calm.jsonl",
    }
    inaccessible = work / paths[stage]
    inaccessible.parent.mkdir(parents=True, exist_ok=True)
    inaccessible.write_bytes(b"untouched cache")
    failure = error_type("storage unavailable")
    read_text, load = Path.read_text, calm.np.load

    def denied_read(
        path: Path, encoding: str | None = None, errors: str | None = None
    ) -> str:
        if path == inaccessible:
            raise failure
        return read_text(path, encoding=encoding, errors=errors)

    def denied_load(path: Path, *, allow_pickle: bool = False) -> object:
        if path == inaccessible:
            raise failure
        return load(path, allow_pickle=allow_pickle)

    def unexpected(*_unused_args: object, **_unused_kwargs: object) -> None:
        pytest.fail("Storage failure started recognition again")

    monkeypatch.setattr(Path, "read_text", denied_read)
    monkeypatch.setattr(calm.np, "load", denied_load)
    monkeypatch.setattr(calm, "scores", unexpected)
    monkeypatch.setattr(calibfit.videomod, "probe", unexpected)
    monkeypatch.setattr(timeline, "read_pond_counts", unexpected)
    if stage == "timing":
        monkeypatch.setattr(calm, "run_calm", lambda *_unused_a, **_unused_kw: [])
    model = RecognitionStub("test")
    runs = {
        "calm": lambda: calm.run_calm(source, cal, work),
        "plate": lambda: calibfit.table_plate(source, work),
        "timing": lambda: timeline.run_header(
            read.ReadContext(source, cal, work, model, model), []
        ),
        "observations": lambda: observe.validate_observation_cache(work, [hand], []),
        "intervals": lambda: observe.validate_observation_cache(work, [hand]),
    }
    with pytest.raises(error_type) as result:
        runs[stage]()
    assert result.value is failure
    assert inaccessible.read_bytes() == b"untouched cache"
