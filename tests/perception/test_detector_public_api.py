# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Public prediction calls preserve geometry, graph lifetime and model ownership."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest

from video2tenhou.perception import detector

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def public_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> detector.Detector:
    """Construct against a backend exposing only the public model interface."""
    model = SimpleNamespace(
        names={0: "face"},
        family="yolo9",
        predict=Mock(return_value=SimpleNamespace(boxes=None)),
        release_graphs=Mock(),
    )
    monkeypatch.setitem(
        sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=lambda *_a, **_kw: model)
    )
    monkeypatch.setattr(detector, "select_device", lambda device: device)
    monkeypatch.setattr(detector.metadata, "version", lambda _name: "1.5.0")
    monkeypatch.setattr(detector, "runtime_signature", lambda _: {"device": "cuda:0"})
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face checkpoint")
    return detector.Detector(
        weights,
        device="cuda:0",
        settings=detector.InferenceOptions(cuda_graph=True, imgsz=128),
    )


def test_shape_changes_release_graphs_before_public_prediction(
    public_detector: detector.Detector,
) -> None:
    """Same padded shapes replay; A/B/A drops each old capture before forwarding."""
    events = []
    model = public_detector.model
    model.release_graphs.side_effect = lambda: events.append("release")
    model.predict.side_effect = lambda _image, **kw: (
        events.append(kw["imgsz"]) or SimpleNamespace(boxes=None)
    )
    # Different source sizes can share the same padded tensor shape.
    images = [
        np.zeros((*shape, 3), np.uint8)
        for shape in [(16, 32), (32, 64), (32, 16), (64, 32), (16, 32)]
    ]
    assert public_detector.predict_batch(images) == [[], [], [], [], []]
    assert events == [
        (64, 128),
        (64, 128),
        "release",
        (128, 64),
        (128, 64),
        "release",
        (64, 128),
    ]
    for image, call in zip(images, model.predict.call_args_list, strict=True):
        assert call.args[0] is image
        assert call.kwargs["cuda_graph"] is True
        assert call.kwargs["color_format"] == "bgr"


def test_eager_predictions_do_not_manage_graphs(
    public_detector: detector.Detector,
) -> None:
    """Eager inference uses the same public prediction and coordinate contract."""
    public_detector.cuda_graph = False
    images = [np.zeros((*shape, 3), np.uint8) for shape in [(16, 32), (32, 16)]]
    assert public_detector.predict_batch(images) == [[], []]
    public_detector.model.release_graphs.assert_not_called()
    assert all(
        call.kwargs["cuda_graph"] is False
        for call in public_detector.model.predict.call_args_list
    )


def test_failed_prediction_still_releases_graphs_before_next_shape(
    public_detector: detector.Detector,
) -> None:
    """A failure after capture must not leave old buffers live across shape changes."""
    events = []
    model = public_detector.model
    model.release_graphs.side_effect = lambda: events.append("release")
    model.predict.side_effect = RuntimeError("postprocessing failed")
    with pytest.raises(RuntimeError, match="postprocessing failed"):
        public_detector.predict(np.zeros((16, 32, 3), np.uint8))
    model.predict.side_effect = lambda _image, **_kw: (
        events.append("predict") or SimpleNamespace(boxes=None)
    )
    assert public_detector.predict(np.zeros((32, 16, 3), np.uint8)) == []
    assert events == ["release", "predict"]


def test_failed_release_prevents_forward_and_is_retried(
    public_detector: detector.Detector,
) -> None:
    """Never forward a changed shape when the preceding captures cannot be freed."""
    first = np.zeros((16, 32, 3), np.uint8)
    second = np.zeros((32, 16, 3), np.uint8)
    public_detector.predict(first)
    model = public_detector.model
    model.release_graphs.side_effect = RuntimeError("release failed")
    with pytest.raises(RuntimeError, match="release failed"):
        public_detector.predict(second)
    assert model.predict.call_count == 1
    model.release_graphs.side_effect = None
    public_detector.predict(second)
    assert model.release_graphs.call_count == 2
    assert model.predict.call_count == 2


def test_batch_lock_blocks_preview_until_all_predictions_finish(
    public_detector: detector.Detector,
) -> None:
    """A preview cannot replace graph/head storage midway through a batch."""
    entered, release, attempted = Event(), Event(), Event()
    values = []

    def predict(image: np.ndarray, **_kwargs: object) -> SimpleNamespace:
        value = int(image[0, 0, 0])
        values.append(value)
        if value == 0:
            entered.set()
            assert release.wait(2)
        boxes = SimpleNamespace(
            xyxy=np.array([[value, 0, 4, 4]]),
            conf=np.array([0.8]),
            cls=np.array([0]),
        )
        return SimpleNamespace(boxes=boxes)

    public_detector.model.predict.side_effect = predict
    images = [np.full((8, 8, 3), n, np.uint8) for n in range(3)]

    def preview() -> list[detector.Det]:
        attempted.set()
        return public_detector.predict(images[2])

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(public_detector.predict_batch, images[:2])
        try:
            assert entered.wait(1)
            second = pool.submit(preview)
            assert attempted.wait(1)
            assert values == [0]
        finally:
            release.set()
        assert [row[0].xyxy[0] for row in first.result(timeout=2)] == [0, 1]
        assert second.result(timeout=2)[0].xyxy[0] == 2
    assert values == [0, 1, 2]
