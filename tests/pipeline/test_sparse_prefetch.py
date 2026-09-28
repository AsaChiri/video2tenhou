"""Sparse overlap preserves planned evidence and joins decoding on failure."""
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from video2tenhou import read
from video2tenhou.calm import Interval, REGIONS
from video2tenhou.layout import Calibration
from video2tenhou.perception.reader import Reading
from video2tenhou.train.data import CLASSES


def test_sparse_plan_holes_keep_window_order_batches_and_caller_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(read, "source_identity", lambda *a, **kw: "source")
    monkeypatch.setattr(read, "READ_BATCH", 3)
    caller = threading.get_ident()
    windows, batches, closed = [], [], []
    first_inference = threading.Event()
    ahead = threading.Event()
    row = dict(hand=2, t_start=1.125, t_end=3.125)
    ivs = [Interval("hand:TL", 1.5, 2.5, 3, True, 0., 0.),
           Interval("pond:TR", 2., 3., 3, True, 0., 0.)]

    def sample(path, **kwargs):
        windows.append(kwargs)
        try:
            for index in range(7):
                if index == 3:
                    assert first_inference.wait(2)
                    ahead.set()
                yield 1. + index / 2, np.full((8, 8, 3), index, dtype=np.uint8)
        finally:
            closed.append(True)

    def infer(items, *_):
        assert threading.get_ident() == caller
        if not first_inference.is_set():
            first_inference.set()
            assert ahead.wait(2)
        batches.append([(t, region, int(image[0, 0, 0])) for t, region, image in items])
        return [Reading(t, region, (8, 8)) for t, region, _ in items]

    monkeypatch.setattr(read.video, "sample", sample)
    monkeypatch.setattr(read, "region_upright", lambda image, *_: (image.copy(), None))
    monkeypatch.setattr(read, "read_regions", infer)
    model = SimpleNamespace(id="same", classes=CLASSES, T=1.0)
    stats, touched = read.run_read("video", Calibration.load("pml"), tmp_path,
                                   [row], ivs, model, model, log=lambda *_: None)
    assert windows == [dict(fps=2., start=1., end=3.625)]
    assert batches == [[(1.5, "hand:TL", 1), (2., "pond:TR", 2), (2., "hand:TL", 2)],
                       [(2.5, "pond:TR", 3), (2.5, "hand:TL", 3), (3., "pond:TR", 4)]]
    assert stats["readings"] == 6 and touched == {2} and closed == [True]
    saved = json.loads((tmp_path / "reads/02/done.json").read_text())
    assert saved["readings"] == 6
    assert not any(thread.name.startswith("dense-crops") for thread in threading.enumerate())
    # A complete raw cache must bypass both the producer and inference.
    read.run_read("video", Calibration.load("pml"), tmp_path, [row], ivs, model, model, log=lambda *_: None)
    assert len(windows) == 1 and len(batches) == 2


@pytest.mark.parametrize("failure", ["crop", "inference", "sampler"])
def test_sparse_failure_joins_producer_and_preserves_previous_publication(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(read, "source_identity", lambda *a, **kw: "source")
    monkeypatch.setattr(read, "READ_BATCH", 2)
    closed = threading.Event()
    produced = []
    output = tmp_path / "reads/02"
    output.mkdir(parents=True)
    (output / "done.json").write_text('{"previous":"manifest"}')
    (output / "hand_TL.jsonl").write_text('previous bytes\n')

    def sample(*args, **kwargs):
        try:
            for i in range(20):
                if failure == "sampler" and i == 2:
                    raise RuntimeError("sampler failed")
                produced.append(i)
                yield i / 2, np.zeros((8, 8, 3), dtype=np.uint8)
        finally:
            closed.set()

    def crop(image, *_):
        if failure == "crop":
            raise RuntimeError("crop failed")
        return image, None

    def infer(items, *_):
        if failure == "inference":
            raise RuntimeError("inference failed")
        return [Reading(t, region, (8, 8)) for t, region, _ in items]

    monkeypatch.setattr(read.video, "sample", sample)
    monkeypatch.setattr(read, "region_upright", crop)
    monkeypatch.setattr(read, "read_regions", infer)
    model = SimpleNamespace(id="same", classes=CLASSES, T=1.0)
    with pytest.raises(RuntimeError, match="failed"):
        read.run_read("video", Calibration.load("pml"), tmp_path,
                      [dict(hand=2, t_start=0., t_end=9.)],
                      [Interval("hand:TL", 0., 9., 19, True, 0., 0.)],
                      model, model, cap=20, force=True, log=lambda *_: None)
    assert closed.is_set() and len(produced) <= 4
    assert (output / "done.json").read_text() == '{"previous":"manifest"}'
    assert (output / "hand_TL.jsonl").read_text() == 'previous bytes\n'
    assert not any(thread.name.startswith("dense-crops") for thread in threading.enumerate())
