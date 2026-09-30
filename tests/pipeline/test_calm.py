# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0


"""Calm intervals, cache identity and minimum reading coverage."""

import hashlib
import json
import os
from typing import TYPE_CHECKING

import numpy as np
import pytest

from video2tenhou import calm
from video2tenhou.layout import Calibration

if TYPE_CHECKING:
    from typing import BinaryIO


if TYPE_CHECKING:
    from collections.abc import Callable


if TYPE_CHECKING:
    from pathlib import Path


def test_scoring_uses_the_same_offscreen_hand_crop_as_the_preview(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Verify scoring uses the same offscreen hand crop as the preview."""
    data = Calibration.load("pml").data
    data["hand"]["TL"].update(rect=[-2, 182, 609, 304], scale=1)
    data["meld"]["BR"].update(rect=[1919, 1079, 1, 1], scale=1)
    cal = Calibration(data)
    frame = np.full((1080, 1920, 3), 100, np.uint8)
    crop = calm.cheap_regions(frame, cal)["hand:TL"]
    assert crop.shape == (304, 609, 3)
    assert not crop[:, :2].any()
    assert np.all(crop[:, 2:] == 100)
    assert np.array_equal(crop, cal.region(frame, "hand:TL")[0])
    monkeypatch.setattr(
        calm.video,
        "sample",
        lambda *_unused_a, **_unused_kw: [(0, frame), (0.5, frame)],
    )
    ts, motion, skin = calm.scores("recording.mp4", cal)
    assert ts.tolist() == [0, 0.5]
    assert motion.shape == skin.shape == (2, len(calm.REGIONS))
    assert np.isfinite(skin).all()
    assert not motion.any()


def test_intervals_merge_short_calm_runs() -> None:
    """Verify intervals merge short calm runs."""
    n = 20
    ts = np.arange(n) * 0.5
    region_count = len(calm.REGIONS)
    mot = np.zeros((n, region_count), np.float32)
    skn = np.zeros((n, region_count), np.float32)
    r = calm.REGIONS.index("hand:TL")
    mot[5:8, r] = 20.0  # a 1.5 s disturbance
    mot[10, r] = 20.0  # a single disturbed sample
    mot[12, r] = 20.0  # calm run of one sample between two disturbances -> disturbed
    skn[15:, r] = 0.5  # an arm for the rest
    ivs = [iv for iv in calm.intervals(ts, mot, skn) if iv.region == "hand:TL"]
    flags = [(iv.calm, iv.n) for iv in ivs]
    # calm runs of 2, 1 and 2 samples between the disturbances are too short (< 3), so
    # from sample 5 on
    # the region is one disturbed interval
    assert flags == [(True, 5), (False, 15)]
    mot[5:8, r] = (
        0.0  # without the first disturbance the calm run reaches sample 9 (10 samples)
    )
    ivs = [iv for iv in calm.intervals(ts, mot, skn) if iv.region == "hand:TL"]
    assert [(iv.calm, iv.n) for iv in ivs] == [(True, 10), (False, 10)]
    # every other region is one calm interval covering everything
    other = [iv for iv in calm.intervals(ts, mot, skn) if iv.region == "pond:BR"]
    assert len(other) == 1
    assert other[0].calm
    assert other[0].n == n


def test_read_floor_fills_blind_spans() -> None:
    """Verify read floor fills blind spans."""
    n = 200  # 100 s at 2 fps
    ts = np.arange(n) * 0.5
    region_count = len(calm.REGIONS)
    mot = np.zeros((n, region_count), np.float32)
    skn = np.zeros((n, region_count), np.float32)
    r = calm.REGIONS.index("meld:BR")
    skn[20:120, r] = (
        0.3  # an arm over the inset from 10 s to 60 s, region still: partial run
    )
    mot[120:190, r] = (
        20.0  # then 35 s of continuous motion: only a still enough sample is read
    )
    mot[150, r] = 5.0
    ivs = [
        iv
        for iv in calm.fill_gaps(calm.intervals(ts, mot, skn), ts, mot, skn)
        if iv.region == "meld:BR"
    ]
    partial = [iv for iv in ivs if iv.partial]
    assert all(iv.calm for iv in partial)
    runs = [iv for iv in partial if iv.n >= 3]
    assert len(runs) == 1
    assert runs[0].t0 == 10.0
    assert runs[0].t1 == 59.5
    # a frame in motion shows tiles being moved: of the 35 s of motion only the dip at
    # sample 150 is read
    singles = [iv for iv in partial if iv.n == 1]
    assert [iv.t0 for iv in singles] == [75.0]
    # other regions are calm throughout: nothing added
    assert not any(
        iv.partial
        for iv in calm.fill_gaps(calm.intervals(ts, mot, skn), ts, mot, skn)
        if iv.region == "pond:TL"
    )


def test_run_calm_recomputes_scores_when_the_geometry_changes(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify run calm recomputes scores when the geometry changes."""
    calls = []

    def fake_scores(
        path: "Path",
        cal: "Calibration",
        *,
        fps: float = 2.0,
        log: "Callable[[str], None]" = print,
    ) -> tuple:
        calls.append(1)
        n, region_count = 6, len(calm.REGIONS)
        return (
            np.arange(n) * 0.5,
            np.zeros((n, region_count), np.float32),
            np.zeros((n, region_count), np.float32),
        )

    monkeypatch.setattr(calm, "scores", fake_scores)
    source = tmp_path / "v.mp4"
    source.write_bytes(b"video fixture; scores are supplied by the test")
    cal = Calibration.load("pml")
    calm.run_calm(source, cal, tmp_path)
    calm.run_calm(source, cal, tmp_path)  # cached
    assert len(calls) == 1
    d2 = json.loads(json.dumps(cal.data))
    d2["notes"] = ["other notes"]
    d2["overhead"]["scale"] = 1.0  # written where nothing was: the same pixels
    cal2 = Calibration(d2)
    calm.run_calm(source, cal2, tmp_path)  # neither moves a rect: cached
    assert len(calls) == 1
    d3 = json.loads(json.dumps(cal.data))
    d3["pond"]["TL"]["rect"][2] += 10
    cal3 = Calibration(d3)
    calm.run_calm(
        source, cal3, tmp_path
    )  # a pond rect moved: the scores are of other pixels
    assert len(calls) == 2
    assert calm.geometry_key(cal3) != calm.geometry_key(cal)
    assert calm.geometry_key(cal2) == calm.geometry_key(cal)


def test_calm_cache_requires_source_contents_and_provenance_even_with_same_stat(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify calm cache requires source contents and provenance even with same stat."""
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"video-a")
    original_stat = source.stat()
    work = tmp_path / "work"
    calls, refreshes = [], []
    identity = calm.source_identity

    def source_digest(path: "Path", *, refresh: bool = False) -> "str":
        refreshes.append(refresh)
        return identity(path, refresh=refresh)

    def score_pixels(
        path: "Path", cal: "Calibration", **_unused_kwargs: object
    ) -> tuple:
        data = source.read_bytes()
        calls.append(data)
        shape = (6, len(calm.REGIONS))
        motion = np.full(shape, 0 if data == b"video-a" else 20, np.float32)
        return np.arange(6) * 0.5, motion, np.zeros(shape, np.float32)

    monkeypatch.setattr(calm, "source_identity", source_digest)
    monkeypatch.setattr(calm, "scores", score_pixels)
    cal = Calibration.load("pml")
    first = calm.run_calm(source, cal, work)
    assert all(iv.calm for iv in first)
    assert calm.run_calm(source, cal, work) == first
    assert calls == [b"video-a"]
    cache = work / "calm_scores.npz"
    with np.load(cache) as z:
        assert str(z["source_sha256"]) == hashlib.sha256(b"video-a").hexdigest()
        legacy = {key: z[key] for key in z.files if key != "source_sha256"}
    np.savez_compressed(cache, **legacy)
    calm.run_calm(source, cal, work)
    assert calls == [b"video-a", b"video-a"]  # unproven legacy cache is not adopted

    source.write_bytes(b"video-b")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert source.stat().st_size == original_stat.st_size
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    changed = calm.run_calm(source, cal, work)
    assert not any(iv.calm for iv in changed)
    assert calls == [b"video-a", b"video-a", b"video-b"]
    calm.run_calm(source, cal, work, force=True)
    assert calls[-2:] == [b"video-b", b"video-b"]
    assert len(calls) == 4
    assert len(refreshes) == 5
    assert all(refreshes)


def test_calm_interrupted_publication_preserves_cache_and_corrupt_cache_resumes(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify calm interrupted publication preserves cache and corrupt cache resumes."""
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fixed source")
    work = tmp_path / "work"
    calls = []

    def fake_scores(*_unused_args: object, **_unused_kwargs: object) -> tuple:
        calls.append(1)
        shape = (6, len(calm.REGIONS))
        return np.arange(6) * 0.5, np.zeros(shape), np.zeros(shape)

    monkeypatch.setattr(calm, "scores", fake_scores)
    cal = Calibration.load("pml")
    expected = calm.run_calm(source, cal, work)
    cache = work / "calm_scores.npz"
    complete = cache.read_bytes()
    save = np.savez_compressed

    def interrupted_save(stream: "BinaryIO", **_unused_arrays: object) -> None:
        stream.write(b"incomplete ZIP")
        msg = "simulated interrupted write"
        raise OSError(msg)

    with monkeypatch.context() as patch:
        patch.setattr(np, "savez_compressed", interrupted_save)
        with pytest.raises(OSError, match="interrupted write"):
            calm.run_calm(source, cal, work, force=True)
    assert cache.read_bytes() == complete
    assert not list(work.glob(".calm-scores-*"))
    assert calm.run_calm(source, cal, work) == expected
    assert len(calls) == 2  # resume uses the intact pre-interruption file

    cache.write_bytes(b"PK\x03\x04truncated ZIP")
    assert calm.run_calm(source, cal, work) == expected
    assert len(calls) == 3
    with np.load(cache) as z:
        incomplete = {key: z[key] for key in z.files if key != "motion"}
    save(cache, **incomplete)
    assert calm.run_calm(source, cal, work) == expected
    assert len(calls) == 4
