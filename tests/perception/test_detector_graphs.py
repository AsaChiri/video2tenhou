# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-shape graph models, weight stability and serialized prediction."""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from video2tenhou.perception import detector


class FakeModel:
    """A LibreYOLO model double that records its graph scopes."""

    names: ClassVar[dict[int, str]] = {0: "face"}
    family = "yolo9"

    def __init__(self) -> None:
        """Start without any graph scope."""
        self.scopes: list[object] = []

    def cuda_graph_scope(self, mode: object) -> nullcontext[None]:
        """Record the requested graph mode."""
        self.scopes.append(mode)
        return nullcontext()

    def release_graphs(self) -> None:
        """Fail: shape changes must not release another shape's capture."""
        pytest.fail("Captures are never released while predicting")


class Detections:
    """Replace the direct forward, recording which model saw which shape."""

    def __init__(self) -> None:
        """Start without calls."""
        self.calls: list[tuple[FakeModel, tuple[int, int]]] = []

    def __call__(
        self,
        model: FakeModel,
        padded: np.ndarray,
        original_size: tuple[int, int],
        conf: float,
        iou: float,
    ) -> tuple[list[list[float]], list[float]]:
        """Return one box whose x0 is the crop's first pixel value."""
        self.calls.append((model, padded.shape[:2]))
        return [[float(padded[0, 0, 0]), 0.0, 4.0, 4.0]], [0.8]


def build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, cuda_graph: bool
) -> tuple[detector.Detector, list[FakeModel], Detections]:
    """Construct a CUDA detector whose backend loads a new fake model per call."""
    loaded: list[FakeModel] = []

    def load(*_args: object, **_kwargs: object) -> FakeModel:
        loaded.append(FakeModel())
        return loaded[-1]

    detections = Detections()
    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=load))
    monkeypatch.setattr(detector.yolo9_direct, "detect", detections)
    monkeypatch.setattr(detector, "select_device", lambda device: device)
    monkeypatch.setattr(detector.metadata, "version", lambda _name: "1.5.0")
    monkeypatch.setattr(detector, "runtime_signature", lambda _: {"device": "cuda:0"})
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face checkpoint")
    current = detector.Detector(
        weights,
        device="cuda:0",
        settings=detector.InferenceOptions(cuda_graph=cuda_graph, imgsz=128),
    )
    return current, loaded, detections


def images(*shapes: tuple[int, int]) -> list[np.ndarray]:
    """Return crops of the given sizes whose first pixel is their index."""
    return [np.full((*shape, 3), i, np.uint8) for i, shape in enumerate(shapes)]


def test_each_padded_shape_replays_only_its_own_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A/B/A/C shapes load one model per padded shape and reuse it on return."""
    current, loaded, detections = build(tmp_path, monkeypatch, cuda_graph=True)
    # (16, 32) and (32, 64) share the padded shape (64, 128).
    crops = images((16, 32), (32, 16), (32, 64), (64, 64))
    result = current.predict_batch(crops)
    assert [row[0].xyxy[0] for row in result] == [0, 1, 2, 3]
    assert len(loaded) == 3
    assert loaded[0] is current.model
    assert detections.calls == [
        (loaded[0], (64, 128)),
        (loaded[1], (128, 64)),
        (loaded[0], (64, 128)),
        (loaded[2], (128, 128)),
    ]
    assert [model.scopes for model in loaded] == [[True, True], [True], [True]]


def test_eager_inference_shares_the_validated_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without graphs no capture can go stale, so every shape uses one model."""
    current, loaded, detections = build(tmp_path, monkeypatch, cuda_graph=False)
    current.predict_batch(images((16, 32), (32, 16)))
    assert loaded == [current.model]
    assert [shape for _, shape in detections.calls] == [(64, 128), (128, 64)]
    assert current.model.scopes == [False, False]


def test_replaced_weights_stop_loading_further_shapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A later model must come from the checkpoint the identity describes."""
    current, loaded, _ = build(tmp_path, monkeypatch, cuda_graph=True)
    first, second = images((16, 32), (32, 16))
    current.predict(first)
    (tmp_path / "weights.pt").write_bytes(b"retrained face checkpoint")
    with pytest.raises(RuntimeError, match="weights changed"):
        current.predict(second)
    assert len(loaded) == 1
    assert current.predict(first)[0].conf == 0.8


def test_batch_lock_blocks_preview_until_all_predictions_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preview cannot use graph or head storage midway through a batch."""
    current, _, detections = build(tmp_path, monkeypatch, cuda_graph=True)
    entered, release, attempted = Event(), Event(), Event()
    values = []

    def detect(
        model: FakeModel,
        padded: np.ndarray,
        original_size: tuple[int, int],
        conf: float,
        iou: float,
    ) -> tuple[list[list[float]], list[float]]:
        values.append(int(padded[0, 0, 0]))
        if values == [0]:
            entered.set()
            assert release.wait(2)
        return detections(model, padded, original_size, conf, iou)

    monkeypatch.setattr(detector.yolo9_direct, "detect", detect)
    crops = [np.full((8, 8, 3), n, np.uint8) for n in range(3)]

    def preview() -> list[detector.Det]:
        attempted.set()
        return current.predict(crops[2])

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(current.predict_batch, crops[:2])
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
