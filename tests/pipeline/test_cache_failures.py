"""Unreadable storage is an operational failure, not a reason to redo recognition."""

from pathlib import Path

import pytest

from video2tenhou import calibfit, calm, observe, timeline
from video2tenhou.layout import Calibration


@pytest.mark.parametrize(
    "stage", ["calm", "plate", "timing", "observations", "intervals"]
)
@pytest.mark.parametrize("error_type", [PermissionError, OSError])
def test_storage_errors_propagate_without_regenerating_caches(
    tmp_path, monkeypatch, stage, error_type
):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    work = tmp_path / "work"
    work.mkdir()
    cal = Calibration.load("pml")
    hand = dict(hand=0, t_start=0.0, t_end=1.0)
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

    def denied_read(path, *args, **kwargs):
        if path == inaccessible:
            raise failure
        return read_text(path, *args, **kwargs)

    def denied_load(path, *args, **kwargs):
        if path == inaccessible:
            raise failure
        return load(path, *args, **kwargs)

    def unexpected(*args, **kwargs):
        pytest.fail("Storage failure started recognition again")

    monkeypatch.setattr(Path, "read_text", denied_read)
    monkeypatch.setattr(calm.np, "load", denied_load)
    monkeypatch.setattr(calm, "scores", unexpected)
    monkeypatch.setattr(calibfit.videomod, "probe", unexpected)
    monkeypatch.setattr(timeline, "read_pond_counts", unexpected)
    if stage == "timing":
        from types import SimpleNamespace
        from video2tenhou.perception import detector

        monkeypatch.setattr(calm, "run_calm", lambda *a, **kw: [])
        monkeypatch.setattr(detector, "Detector", lambda: SimpleNamespace(id="test"))
    runs = {
        "calm": lambda: calm.run_calm(source, cal, work),
        "plate": lambda: calibfit.table_plate(source, work),
        "timing": lambda: timeline.run_header(source, cal, [], work),
        "observations": lambda: observe.validate_observation_cache(work, [hand], []),
        "intervals": lambda: observe.validate_observation_cache(work, [hand]),
    }
    with pytest.raises(error_type) as result:
        runs[stage]()
    assert result.value is failure
    assert inaccessible.read_bytes() == b"untouched cache"
