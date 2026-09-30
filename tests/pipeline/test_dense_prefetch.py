"""Dense scheduling must preserve pixels, inference batches, and failure semantics."""

import hashlib
import shutil
import subprocess
import threading
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest

from video2tenhou import read
from video2tenhou.layout import Calibration
from video2tenhou.perception.evidence_policy import DEFAULT_POLICY
from video2tenhou.perception.reader import Box, Reading
from video2tenhou.train.data import CLASSES


def test_dense_prefetch_preserves_batches_pixels_posteriors_and_caller_thread(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(read, "source_identity", lambda _: "source")
    monkeypatch.setattr(read, "READ_BATCH", 4)
    producer_ready = threading.Event()
    inference_started = threading.Event()
    caller = threading.get_ident()
    calls = []
    closes = []
    concurrent = [False]

    def sample(path, **kwargs):
        try:
            for i in range(7):
                if concurrent[0] and i == 2:
                    assert inference_started.wait(2)
                    producer_ready.set()
                yield 3.125 + i / 5, np.full((8, 8, 3), i, dtype=np.uint8)
        finally:
            closes.append(True)

    def crop(frame, cal, kind, corner):
        return frame[:6, :6].copy() + (1 if corner == "TR" else 0), None

    def infer(items, det, clf):
        assert threading.get_ident() == caller
        if concurrent[0] and not inference_started.is_set():
            inference_started.set()
            assert producer_ready.wait(2)  # cropping advances during this inference
        calls.append([(t, region, img.tobytes()) for t, region, img in items])
        result = []
        for t, region, img in items:
            posterior = np.zeros(len(CLASSES), dtype=float)
            posterior[0] = 0.8 + int(img[0, 0, 0]) / 100
            posterior[1] = 1 - posterior[0]
            result.append(
                Reading(t, region, (6, 6), [Box((1, 1, 5, 5), 0.95, False, posterior)])
            )
        return result

    monkeypatch.setattr(read.video, "sample", sample)
    monkeypatch.setattr(read, "region_upright", crop)
    monkeypatch.setattr(read, "read_regions", infer)
    cal = Calibration.load("pml")
    stub = SimpleNamespace(
        id="model", classes=CLASSES, T=1.0, evidence_policy=DEFAULT_POLICY
    )
    prefetch = read._prefetch_batches
    monkeypatch.setattr(read, "_prefetch_batches", lambda batches: batches)
    serial = read.dense_reads(
        "video",
        cal,
        tmp_path / "serial",
        stub,
        stub,
        3.125,
        4.6,
        ["hand:TL", "hand:TR"],
    )
    expected_calls = list(calls)
    calls.clear()
    concurrent[0] = True
    monkeypatch.setattr(read, "_prefetch_batches", prefetch)
    parallel = read.dense_reads(
        "video",
        cal,
        tmp_path / "parallel",
        stub,
        stub,
        3.125,
        4.6,
        ["hand:TL", "hand:TR"],
    )
    assert calls == expected_calls
    assert [len(batch) for batch in calls] == [4, 4, 4, 2]
    assert parallel == serial
    assert closes == [True, True]


@pytest.mark.parametrize("failure", ["producer", "inference"])
def test_dense_failure_closes_sampler_and_never_publishes_cache(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(read, "source_identity", lambda _: "source")
    monkeypatch.setattr(read, "READ_BATCH", 2)
    closed = threading.Event()
    produced = []

    def sample(*args, **kwargs):
        try:
            for i in range(100):
                if failure == "producer" and i == 3:
                    raise RuntimeError("sampling failed")
                produced.append(i)
                yield i / 5, np.zeros((4, 4, 3), np.uint8)
        finally:
            closed.set()

    def infer(items, *args):
        if failure == "inference":
            raise RuntimeError("inference failed")
        return [Reading(t, region, (4, 4)) for t, region, _ in items]

    monkeypatch.setattr(read.video, "sample", sample)
    monkeypatch.setattr(read, "region_upright", lambda frame, *args: (frame, None))
    monkeypatch.setattr(read, "read_regions", infer)
    stub = SimpleNamespace(
        id="model", classes=CLASSES, T=1.0, evidence_policy=DEFAULT_POLICY
    )
    with pytest.raises(RuntimeError, match="failed"):
        read.dense_reads(
            "video", Calibration.load("pml"), tmp_path, stub, stub, 0, 20, ["hand:TL"]
        )
    assert closed.is_set()
    assert not list((tmp_path / "dense").glob("*.json"))
    assert len(produced) <= 4  # one current batch and at most one prefetched batch
    assert not any(t.name.startswith("dense-crops") for t in threading.enumerate())


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_real_ffmpeg_prefetch_keeps_fractional_window_pixels_and_closes_child(
    tmp_path, monkeypatch
):
    clip = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=96x64:rate=12",
            "-t",
            "2",
            "-threads",
            "1",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    original_sample = read.video.sample
    original_popen = read.video.subprocess.Popen
    children = []

    def popen(*args, **kwargs):
        child = original_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(read.video.subprocess, "Popen", popen)
    monkeypatch.setattr(
        read.video, "sample", lambda *a, **k: original_sample(*a, **k, size=(96, 64))
    )
    monkeypatch.setattr(
        read, "region_upright", lambda frame, *args: (frame[4:44, 8:64].copy(), None)
    )
    monkeypatch.setattr(read, "READ_BATCH", 2)

    def digest(batches):
        return [
            [
                (t, region, hashlib.sha256(img.tobytes()).hexdigest())
                for t, region, img in batch
            ]
            for batch in batches
        ]

    for lo, hi, fps in ((0.031, 1.731, 5.0), (0.143, 1.913, 2.5)):
        serial = digest(read._dense_batches(clip, None, lo, hi, ["hand:TL"], fps))
        with closing(
            read._prefetch_batches(
                read._dense_batches(clip, None, lo, hi, ["hand:TL"], fps)
            )
        ) as batches:
            assert digest(batches) == serial
    # Abandoning the consumer must terminate its real decoder, not leave a
    # prefetched worker waiting on a full pipe after the UI operation fails.
    with closing(
        read._prefetch_batches(read._dense_batches(clip, None, 0, 2, ["hand:TL"], 30))
    ) as batches:
        assert next(batches)
    assert children and all(child.poll() is not None for child in children)
    assert not any(t.name.startswith("dense-crops") for t in threading.enumerate())
