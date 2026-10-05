# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Equivalence checks at the detector/classifier/region-structure boundary."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torchvision import models

from video2tenhou.perception import detector
from video2tenhou.perception.classifier import Classifier, make_model, to_tensor
from video2tenhou.perception.crops import to_crop
from video2tenhou.perception.detector import Det, Detector
from video2tenhou.perception.reader import read_region, read_regions
from video2tenhou.perception.tiles import CLASS_INDEX, CLASSES


class DetectorFixture:
    """Deterministic tile boxes for comparing serial and buffered reading."""

    def predict(self, img: np.ndarray) -> list[Det]:
        """Return the fixture's deterministic boxes for one crop."""
        return [
            Det((15, 15, 35, 45), 0.95),
            Det((40, 15, 75, 35), 0.9),
            Det((105, 15, 125, 45), 0.7),
            Det((0, 0, 1, 1), 0.6),
        ]

    def predict_batch(self, imgs: list[np.ndarray]) -> list[list[Det]]:
        """Return one deterministic detection result per crop."""
        return [self.predict(img) for img in imgs]


class ClassifierFixture:
    """Deterministic posteriors with recorded crop batching calls."""

    def __init__(self) -> None:
        """Start recording classifier batch sizes and orientations."""
        self.calls = []

    def classify(
        self, crops: list[np.ndarray], sideways: list[bool] | None = None
    ) -> np.ndarray:
        """Record crop batching and return deterministic tile posteriors."""
        self.calls.append((len(crops), tuple(sideways or [])))
        p = np.zeros((len(crops), len(CLASS_INDEX)), np.float32)
        for i, crop in enumerate(crops):
            tile = "none" if crop.mean() > 150 else "2p"
            p[i, CLASS_INDEX[tile]] = 1
        return p


def test_batch_keeps_region_structure_and_sideways_semantics() -> None:
    """Cross-region classifier batching matches reading each region alone."""
    img = np.zeros((100, 140, 3), np.uint8)
    img[:, 104:] = 255
    items = [(1.5, "pond:TL", img), (2.0, "hand:TR", img), (2.5, "meld:BL", img)]
    det = DetectorFixture()
    serial_clf, batch_clf = ClassifierFixture(), ClassifierFixture()
    serial = [
        read_region(image, name, det, serial_clf, t=t) for t, name, image in items
    ]
    batched = read_regions(items, det, batch_clf)
    assert [rd.to_dict() for rd in batched] == [rd.to_dict() for rd in serial]
    assert len(batch_clf.calls) == 1
    assert len(serial_clf.calls) == 3
    assert all(len(rd.boxes) == 2 for rd in batched)  # none and invalid crop removed
    assert read_regions([], det, batch_clf) == []


def _face_detector(
    detect: Callable, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Detector:
    """Initialize the production adapter, replacing the model and its forward."""
    model = SimpleNamespace(
        names={0: "face"}, family="yolo9", cuda_graph_scope=lambda _mode: nullcontext()
    )
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(LibreYOLO=lambda *_unused_a, **_unused_kw: model),
    )
    monkeypatch.setattr(detector.yolo9_direct, "detect", detect)
    package_version = detector.metadata.version
    monkeypatch.setattr(
        detector.metadata,
        "version",
        lambda name: "fixture" if name == "libreyolo" else package_version(name),
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture")
    return Detector(weights, device="cpu")


def test_detector_preserves_input_order_and_single_image_padding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every crop is forwarded alone, padded only to its own stride-aligned shape."""
    calls = []

    def detect(
        _model: object,
        padded: np.ndarray,
        original_size: tuple[int, int],
        *_: float,
    ) -> tuple[list[list[float]], list[float]]:
        calls.append((original_size, padded.shape[:2]))
        return [[float(padded[0, 0, 0]), 1, 2, 3]], [0.8]

    det = _face_detector(detect, tmp_path, monkeypatch)
    imgs = [np.full((10 + i % 2, 20, 3), i, np.uint8) for i in range(7)]
    result = det.predict_batch(imgs)
    assert [ds[0].xyxy[0] for ds in result] == list(range(7))
    assert calls == [
        ((20, img.shape[0]), (512 if img.shape[0] == 10 else 576, 1024)) for img in imgs
    ]


def test_detector_production_default_keeps_single_image_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def detect(
        _model: object,
        padded: np.ndarray,
        original_size: tuple[int, int],
        *_: float,
    ) -> tuple[list[list[float]], list[float]]:
        calls.append((original_size, padded.shape))
        return [], []

    det = _face_detector(detect, tmp_path, monkeypatch)
    assert det.predict_batch([np.zeros((20, 20, 3), np.uint8)] * 3) == [[], [], []]
    assert calls == [((20, 20), (1024, 1024, 3))] * 3


def test_inference_model_construction_never_requests_downloaded_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local checkpoint replaces every parameter, so ImageNet is unnecessary."""

    def resnet18(*, weights: models.ResNet18_Weights | None) -> SimpleNamespace:
        assert weights is None
        return SimpleNamespace(fc=torch.nn.Linear(8, 1000))

    monkeypatch.setattr(models, "resnet18", resnet18)
    head = make_model(len(CLASS_INDEX)).fc
    assert isinstance(head, torch.nn.Linear)
    assert head.out_features == len(CLASS_INDEX)


def test_classifier_inputs_and_posteriors_match_per_crop_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Device-side byte normalization reproduces ``to_tensor`` exactly."""
    torch.manual_seed(0)
    model = make_model(len(CLASSES)).eval()
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    meta = {"classes": CLASSES, "temperature": 0.7}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    clf = Classifier(tmp_path, device="cpu")
    rng = np.random.default_rng(0)
    crops = [
        rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        for h, w in [(40, 30), (130, 90), (96, 64), (20, 70)]
    ]
    inputs = []
    forward = clf.model.forward
    monkeypatch.setattr(clf.model, "forward", lambda x: inputs.append(x) or forward(x))
    posteriors = clf.posteriors(crops)
    reference = torch.stack([to_tensor(to_crop(c)) for c in crops])
    assert torch.equal(inputs[0], reference)
    assert inputs[0].stride() == reference.stride()
    with torch.no_grad():
        expected = torch.softmax(model(reference).float() / 0.7, dim=1).numpy()
    assert np.array_equal(posteriors, expected)
