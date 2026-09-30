"""Backend adapters preserve BGR/coordinate/class and semantic cache contracts."""

import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from video2tenhou.perception import detector


def test_face_backend_keeps_one_class_bgr_and_rectangular_inputs(tmp_path, monkeypatch):
    calls = []
    boxes = SimpleNamespace(
        xyxy=np.array([[1.0, 2.0, 30.0, 40.0]]), conf=np.array([0.8]), cls=np.array([0])
    )
    model = SimpleNamespace(
        names={0: "face"},
        family="yolo9",
        predict=lambda image, **kwargs: (
            calls.append((image, kwargs)) or SimpleNamespace(boxes=boxes)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(LibreYOLO=lambda *args, **kwargs: model),
    )
    package_version = detector.metadata.version
    monkeypatch.setattr(
        detector.metadata,
        "version",
        lambda name: "1.5.0" if name == "libreyolo" else package_version(name),
    )
    weights = tmp_path / "face.pt"
    weights.write_bytes(b"one")
    current = detector.Detector(weights, backend="libreyolo", device="cpu")
    image = np.zeros((100, 400, 3), np.uint8)
    result = current.predict(image)
    assert calls[0][0] is image
    assert calls[0][1]["color_format"] == "bgr" and calls[0][1]["imgsz"] == (256, 1024)
    assert result == [detector.Det((1.0, 2.0, 30.0, 40.0), 0.8, False)]
    first_id = current.id
    stat = weights.stat()
    weights.write_bytes(b"two")
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert detector.Detector(weights, backend="libreyolo", device="cpu").id != first_id
    assert (
        detector.Detector(weights, backend="libreyolo", device="cpu", conf=0.2).id
        != current.id
    )
    current_id = detector.Detector(weights, backend="libreyolo", device="cpu").id
    monkeypatch.setattr(
        detector,
        "runtime_signature",
        lambda device: {"device": str(device), "math": "different"},
    )
    assert (
        detector.Detector(weights, backend="libreyolo", device="cpu").id != current_id
    )
    boxes.cls = np.array([1])
    with pytest.raises(ValueError, match="unexpected class"):
        current.predict(image)


def test_face_backend_rejects_tile_identity_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(
            LibreYOLO=lambda *args, **kwargs: SimpleNamespace(names={0: "1m"})
        ),
    )
    weights = tmp_path / "face.pt"
    weights.write_bytes(b"one")
    with pytest.raises(ValueError, match="exactly one class"):
        detector.Detector(weights, backend="libreyolo", device="cpu")


def test_graph_metadata_override_and_effective_policy_invalidate_cache(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(detector, "select_device", lambda device: device)
    calls = []
    model = SimpleNamespace(
        names={0: "face"},
        family="yolo9",
        size="s",
        predict=lambda image, **kw: calls.append(kw) or SimpleNamespace(boxes=None),
    )
    monkeypatch.setitem(
        sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=lambda *a, **k: model)
    )
    monkeypatch.setattr(detector.metadata, "version", lambda _: "1.5.0")
    monkeypatch.setattr(
        detector, "runtime_signature", lambda device: {"device": str(device)}
    )
    monkeypatch.setattr(
        detector, "enable_owned_graphs", lambda model, **kw: kw["device"] == "cuda:0"
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face")
    (tmp_path / "meta.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                backend="libreyolo",
                architecture="yolo9-s",
                classes={"0": "face"},
                weights_sha256=hashlib.sha256(b"face").hexdigest(),
                inference=dict(cuda_graph=True),
            )
        )
    )
    automatic = detector.Detector(weights, device="cuda:0")
    eager = detector.Detector(weights, device="cuda:0", cuda_graph=False)
    assert automatic.cuda_graph and not eager.cuda_graph and automatic.id != eager.id
    monkeypatch.setattr(
        automatic,
        "_consume_prepared",
        lambda prepared: calls.append({"cuda_graph": True}) or [],
    )
    automatic.predict(np.zeros((10, 10, 3), np.uint8))
    eager.predict(np.zeros((10, 10, 3), np.uint8))
    assert [row["cuda_graph"] for row in calls] == [True, False]
    assert not detector.Detector(weights, device="cpu").cuda_graph
    monkeypatch.setattr(detector, "GRAPH_POLICY", "different-bounded-capture-policy")
    assert detector.Detector(weights, device="cuda:0").id != automatic.id
    assert detector.Detector(weights, device="cuda:0", cuda_graph=False).id == eager.id


def test_shared_predictions_stay_locked_until_boxes_are_materialized(
    tmp_path, monkeypatch
):
    materializing, release, attempted, second_framework = (Event() for _ in range(4))
    calls = []

    class Coordinates:
        def tolist(self):
            materializing.set()
            assert release.wait(2)
            return [[1.0, 2.0, 3.0, 4.0]]

    def predict(image, **kwargs):
        calls.append(image)
        if len(calls) > 1:
            second_framework.set()
        boxes = SimpleNamespace(
            xyxy=Coordinates(), conf=np.array([0.9]), cls=np.array([0])
        )
        return SimpleNamespace(boxes=boxes)

    model = SimpleNamespace(names={0: "face"}, family="yolo9", predict=predict)
    monkeypatch.setitem(
        sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=lambda *a, **kw: model)
    )
    monkeypatch.setattr(detector.metadata, "version", lambda _: "1.5.0")
    monkeypatch.setattr(detector, "runtime_signature", lambda _: {})
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face")
    current = detector.Detector(weights, backend="libreyolo", device="cpu")
    image = np.zeros((10, 10, 3), np.uint8)

    def another_request():
        attempted.set()
        return current.predict(image)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(current.predict, image)
        try:
            assert materializing.wait(1)
            second = pool.submit(another_request)
            assert attempted.wait(1)
            assert not second_framework.wait(0.05)
        finally:
            release.set()
        assert first.result(timeout=2) == second.result(timeout=2)
        assert second_framework.is_set() and len(calls) == 2


def test_face_backend_rejects_other_families_with_different_preprocessing(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(
            LibreYOLO=lambda *args, **kwargs: SimpleNamespace(
                names={0: "face"}, family="yolox"
            )
        ),
    )
    weights = tmp_path / "face.pt"
    weights.write_bytes(b"one")
    with pytest.raises(ValueError, match="standard yolo9"):
        detector.Detector(weights, backend="libreyolo", device="cpu")


def test_metadata_selects_calibrated_defaults_and_explicit_overrides(
    tmp_path, monkeypatch
):
    model = SimpleNamespace(names={0: "face"}, family="yolo9", size="s")
    monkeypatch.setitem(
        sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=lambda *a, **k: model)
    )
    monkeypatch.setattr(detector.metadata, "version", lambda name: "1.5.0")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face checkpoint")
    meta = dict(
        schema_version=1,
        backend="libreyolo",
        architecture="yolo9-s",
        classes={"0": "face"},
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        inference=dict(imgsz=1024, confidence=0.02, iou=0.5),
    )
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta))
    automatic = detector.Detector(weights, device="cpu")
    assert (automatic.backend, automatic.conf, automatic.imgsz, automatic.iou) == (
        "libreyolo",
        0.02,
        1024,
        0.5,
    )
    explicit = detector.Detector(weights, device="cpu", conf=0.15, imgsz=640)
    assert (explicit.conf, explicit.imgsz, explicit.iou) == (0.15, 640, 0.5)
    assert explicit.id != automatic.id
    meta["inference"]["confidence"] = "invalid metadata default"
    path.write_text(json.dumps(meta))
    # Explicit settings replace defaults before validation, as for valid metadata.
    assert detector.Detector(weights, device="cpu", conf=0.15).conf == 0.15
    with pytest.raises(ValueError, match="confidence"):
        detector.Detector(weights, device="cpu")
    meta["inference"]["confidence"] = 0.02
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="conflicts"):
        detector.Detector(weights, device="cpu", backend="unsupported")
    meta["classes"] = {"0": "back"}
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="classes differ"):
        detector.Detector(weights, device="cpu")
    weights.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        detector.Detector(weights, device="cpu")


@pytest.mark.parametrize(
    "settings",
    [
        {"conf": float("nan")},
        {"conf": 0},
        {"iou": 2},
        {"imgsz": 0},
        {"cuda_graph": "yes"},
    ],
)
def test_invalid_inference_settings_fail_before_loading_backend(tmp_path, settings):
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face checkpoint")
    with pytest.raises(ValueError, match="Detector"):
        detector.Detector(weights, backend="libreyolo", device="cpu", **settings)


def test_default_face_backend_preserves_recognition_fingerprint(tmp_path, monkeypatch):
    """Recognition cache identity includes the exact supported inference contract."""
    model = SimpleNamespace(names={0: "face"}, family="yolo9")
    monkeypatch.setitem(
        sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=lambda *a, **kw: model)
    )
    monkeypatch.setattr(detector.metadata, "version", lambda _: "1.5.0")
    monkeypatch.setattr(detector, "runtime_signature", lambda _: {"device": "cpu"})
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face")
    current = detector.Detector(weights, device="cpu")
    identity = dict(
        weights_sha256=hashlib.sha256(b"face").hexdigest(),
        backend="libreyolo",
        version="1.5.0",
        imgsz=1024,
        conf=0.15,
        iou=0.5,
        color="BGR",
        classes={0: "face"},
        cuda_graph=False,
        graph_policy=None,
        runtime={"device": "cpu"},
        preprocessing="libreyolo9-top-left-rect-stride32-v1",
    )
    assert (
        current.id
        == "detector:"
        + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    )
    assert current.backend == "libreyolo"


@pytest.mark.parametrize(
    "error", [RuntimeError("weights load failed"), OSError("unreadable checkpoint")]
)
def test_backend_loading_error_preserves_original_exception(
    tmp_path, monkeypatch, error
):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=fail))
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"checkpoint")
    with pytest.raises(type(error)) as caught:
        detector.Detector(weights, device="cpu")
    assert caught.value is error
