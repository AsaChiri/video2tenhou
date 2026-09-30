# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Sparse overlap preserves planned evidence and joins decoding on failure."""

import json
import threading
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.recognition import RecognitionStub
from video2tenhou import read
from video2tenhou.calm import Interval
from video2tenhou.layout import Calibration
from video2tenhou.perception.reader import Reading
from video2tenhou.train.data import CLASSES

if TYPE_CHECKING:
    from collections.abc import Iterator


if TYPE_CHECKING:
    from pathlib import Path


def test_sparse_plan_holes_keep_window_order_batches_and_caller_inference(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify sparse plan holes keep window order batches and caller inference."""
    monkeypatch.setattr(
        read, "source_identity", lambda *_unused_a, **_unused_kw: "source"
    )
    monkeypatch.setattr(read, "READ_BATCH", 3)
    caller = threading.get_ident()
    windows, batches, closed = [], [], []
    first_inference = threading.Event()
    ahead = threading.Event()
    row = {"hand": 2, "t_start": 1.125, "t_end": 3.125}
    ivs = [
        Interval("hand:TL", 1.5, 2.5, 3, calm=True, motion=0.0, skin=0.0),
        Interval("pond:TR", 2.0, 3.0, 3, calm=True, motion=0.0, skin=0.0),
    ]

    def sample(
        path: "str | Path", **kwargs: "object"
    ) -> "Iterator[tuple[float, np.ndarray]]":
        windows.append(kwargs)
        try:
            for index in range(7):
                if index == 3:
                    assert first_inference.wait(2)
                    ahead.set()
                yield 1.0 + index / 2, np.full((8, 8, 3), index, dtype=np.uint8)
        finally:
            closed.append(True)

    def infer(items: "read.CropBatch", *_: object) -> list:
        assert threading.get_ident() == caller
        if not first_inference.is_set():
            first_inference.set()
            assert ahead.wait(2)
        batches.append([(t, region, int(image[0, 0, 0])) for t, region, image in items])
        return [Reading(t, region, (8, 8)) for t, region, _ in items]

    monkeypatch.setattr(read.video, "sample", sample)
    monkeypatch.setattr(read, "region_upright", lambda image, *_: (image.copy(), None))
    monkeypatch.setattr(read, "read_regions", infer)
    model = RecognitionStub(id="same", classes=CLASSES, T=1.0)
    stats, touched = read.run_read(
        read.ReadContext("video", Calibration.load("pml"), tmp_path, model, model),
        [row],
        ivs,
        log=lambda *_: None,
    )
    assert windows == [{"fps": 2.0, "start": 1.0, "end": 3.625}]
    assert batches == [
        [(1.5, "hand:TL", 1), (2.0, "pond:TR", 2), (2.0, "hand:TL", 2)],
        [(2.5, "pond:TR", 3), (2.5, "hand:TL", 3), (3.0, "pond:TR", 4)],
    ]
    assert stats["readings"] == 6
    assert touched == {2}
    assert closed == [True]
    saved = json.loads((tmp_path / "reads/02/done.json").read_text())
    assert saved["readings"] == 6
    assert not any(
        thread.name.startswith("dense-crops") for thread in threading.enumerate()
    )
    # A complete raw cache must bypass both the producer and inference.
    read.run_read(
        read.ReadContext("video", Calibration.load("pml"), tmp_path, model, model),
        [row],
        ivs,
        log=lambda *_: None,
    )
    assert len(windows) == 1
    assert len(batches) == 2


@pytest.mark.parametrize("failure", ["crop", "inference", "sampler"])
def test_sparse_failure_joins_producer_and_preserves_previous_publication(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", failure: str
) -> None:
    """Verify sparse failure joins producer and preserves previous publication."""
    monkeypatch.setattr(
        read, "source_identity", lambda *_unused_a, **_unused_kw: "source"
    )
    monkeypatch.setattr(read, "READ_BATCH", 2)
    closed = threading.Event()
    produced = []
    output = tmp_path / "reads/02"
    output.mkdir(parents=True)
    (output / "done.json").write_text('{"previous":"manifest"}')
    (output / "hand_TL.jsonl").write_text("previous bytes\n")

    def sample(
        *_unused_args: object, **_unused_kwargs: object
    ) -> "Iterator[tuple[float, np.ndarray]]":
        try:
            for i in range(20):
                if failure == "sampler" and i == 2:
                    msg = "sampler failed"
                    raise RuntimeError(msg)
                produced.append(i)
                yield i / 2, np.zeros((8, 8, 3), dtype=np.uint8)
        finally:
            closed.set()

    def crop(image: "np.ndarray", *_: object) -> tuple:
        if failure == "crop":
            msg = "crop failed"
            raise RuntimeError(msg)
        return image, None

    def infer(items: "read.CropBatch", *_: object) -> list:
        if failure == "inference":
            msg = "inference failed"
            raise RuntimeError(msg)
        return [Reading(t, region, (8, 8)) for t, region, _ in items]

    monkeypatch.setattr(read.video, "sample", sample)
    monkeypatch.setattr(read, "region_upright", crop)
    monkeypatch.setattr(read, "read_regions", infer)
    model = RecognitionStub(id="same", classes=CLASSES, T=1.0)
    with pytest.raises(RuntimeError, match="failed"):
        read.run_read(
            read.ReadContext("video", Calibration.load("pml"), tmp_path, model, model),
            [{"hand": 2, "t_start": 0.0, "t_end": 9.0}],
            [Interval("hand:TL", 0.0, 9.0, 19, calm=True, motion=0.0, skin=0.0)],
            options=read.ReadOptions(cap=20, force=True),
            log=lambda *_: None,
        )
    assert closed.is_set()
    assert len(produced) <= 4
    assert (output / "done.json").read_text() == '{"previous":"manifest"}'
    assert (output / "hand_TL.jsonl").read_text() == "previous bytes\n"
    assert not any(
        thread.name.startswith("dense-crops") for thread in threading.enumerate()
    )
