"""Equivalence checks at the detector/classifier/region-structure boundary."""

import sys
from types import SimpleNamespace

import numpy as np

from video2tenhou.perception import detector
from video2tenhou.perception.detector import Det, Detector
from video2tenhou.perception.reader import read_region, read_regions
from video2tenhou.train.data import CLASS_INDEX


class DetectorFixture:
    def predict(self, img):
        return [
            Det((15, 15, 35, 45), 0.95, False),
            Det((40, 15, 75, 35), 0.9, False),
            Det((80, 15, 100, 45), 0.8, True),
            Det((105, 15, 125, 45), 0.7, False),
            Det((0, 0, 1, 1), 0.6, False),
        ]

    def predict_batch(self, imgs):
        return [self.predict(img) for img in imgs]


class ClassifierFixture:
    def __init__(self):
        self.calls = []

    def classify(self, crops, sideways):
        self.calls.append((len(crops), tuple(sideways)))
        p = np.zeros((len(crops), len(CLASS_INDEX)), np.float32)
        for i, crop in enumerate(crops):
            tile = "none" if crop.mean() > 150 else "2p"
            p[i, CLASS_INDEX[tile]] = 1
        return p


def test_batch_keeps_region_structure_back_tiles_and_sideways_semantics():
    """Cross-region aggregation must not assign probabilities to tile backs."""
    img = np.zeros((100, 140, 3), np.uint8)
    img[:, 104:] = 255
    items = [(1.5, "pond:TL", img), (2.0, "hand:TR", img), (2.5, "meld:BL", img)]
    det = DetectorFixture()
    serial_clf, batch_clf = ClassifierFixture(), ClassifierFixture()
    serial = [
        read_region(None, None, name, det, serial_clf, t=t, img=image)
        for t, name, image in items
    ]
    batched = read_regions(items, det, batch_clf)
    assert [rd.to_dict() for rd in batched] == [rd.to_dict() for rd in serial]
    assert len(batch_clf.calls) == 1 and len(serial_clf.calls) == 3
    assert all(len(rd.boxes) == 3 for rd in batched)  # none and invalid crop removed
    assert sum(b.p[CLASS_INDEX["X"]] == 1 for rd in batched for b in rd.boxes) == 3
    assert read_regions([], det, batch_clf) == []


def _face_detector(model, tmp_path, monkeypatch):
    """Initialize the production adapter while replacing only the external model."""
    model.names = {0: "face"}
    model.family = "yolo9"
    monkeypatch.setitem(
        sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=lambda *a, **kw: model)
    )
    package_version = detector.metadata.version
    monkeypatch.setattr(
        detector.metadata,
        "version",
        lambda name: "fixture" if name == "libreyolo" else package_version(name),
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"fixture")
    return Detector(weights, device="cpu")


def test_detector_preserves_input_order_and_single_image_padding(tmp_path, monkeypatch):
    calls = []

    class Model:
        def predict(self, img, **kwargs):
            imgs = [img]
            calls.append((len(imgs), {i.shape for i in imgs}, kwargs))
            return SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=np.array([[float(img[0, 0, 0]), 1, 2, 3]]),
                    conf=np.array([0.8]),
                    cls=np.array([0]),
                )
            )

    det = _face_detector(Model(), tmp_path, monkeypatch)
    imgs = [np.full((10 + i % 2, 20, 3), i, np.uint8) for i in range(7)]
    result = det.predict_batch(imgs)
    assert [ds[0].xyxy[0] for ds in result] == list(range(7))
    assert all(
        n == 1
        and len(shapes) == 1
        and args["imgsz"] == (512 if next(iter(shapes))[0] == 10 else 576, 1024)
        for n, shapes, args in calls
    )


def test_detector_production_default_keeps_single_image_geometry(tmp_path, monkeypatch):
    calls = []
    model = SimpleNamespace(
        predict=lambda img, **kw: calls.append(img.shape) or SimpleNamespace(boxes=None)
    )
    det = _face_detector(model, tmp_path, monkeypatch)
    assert det.predict_batch([np.zeros((20, 20, 3), np.uint8)] * 3) == [[], [], []]
    assert calls == [(20, 20, 3)] * 3


def test_inference_model_construction_never_requests_downloaded_weights(monkeypatch):
    """The local checkpoint replaces every parameter, so ImageNet is unnecessary."""
    import torch
    from torchvision import models

    from video2tenhou.train.train_classifier import make_model

    def resnet18(*, weights):
        assert weights is None
        return SimpleNamespace(fc=torch.nn.Linear(8, 1000))

    monkeypatch.setattr(models, "resnet18", resnet18)
    assert make_model(len(CLASS_INDEX)).fc.out_features == len(CLASS_INDEX)
