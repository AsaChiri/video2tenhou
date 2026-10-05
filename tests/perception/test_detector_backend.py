# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Backend adapters preserve BGR/coordinate/class and semantic cache contracts."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from video2tenhou.perception import detector


def test_face_backend_keeps_rectangular_inputs_and_semantic_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crops keep stride-aligned rectangular padding; inputs change the identity."""
    calls = []
    model = SimpleNamespace(
        names={0: "face"}, family="yolo9", cuda_graph_scope=lambda _mode: nullcontext()
    )

    def detect(
        model: object,
        padded: np.ndarray,
        original_size: tuple[int, int],
        conf: float,
        iou: float,
    ) -> tuple[list[list[float]], list[float]]:
        calls.append((model, padded.shape, original_size, conf, iou))
        return [[1.0, 2.0, 30.0, 40.0]], [0.8]

    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(LibreYOLO=lambda *_unused_args, **_unused_kwargs: model),
    )
    monkeypatch.setattr(detector.yolo9_direct, "detect", detect)
    package_version = detector.metadata.version
    monkeypatch.setattr(
        detector.metadata,
        "version",
        lambda name: "1.5.0" if name == "libreyolo" else package_version(name),
    )
    weights = tmp_path / "face.pt"
    weights.write_bytes(b"one")
    current = detector.Detector(weights, device="cpu")
    image = np.zeros((100, 400, 3), np.uint8)
    result = current.predict(image)
    assert calls == [(model, (256, 1024, 3), (400, 100), 0.15, 0.5)]
    assert result == [detector.Det((1.0, 2.0, 30.0, 40.0), 0.8)]
    first_id = current.id
    stat = weights.stat()
    weights.write_bytes(b"two")
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert detector.Detector(weights, device="cpu").id != first_id
    assert (
        detector.Detector(
            weights,
            device="cpu",
            settings=detector.InferenceOptions(confidence=0.2),
        ).id
        != current.id
    )
    current_id = detector.Detector(weights, device="cpu").id
    monkeypatch.setattr(
        detector,
        "runtime_signature",
        lambda device: {"device": str(device), "math": "different"},
    )
    assert detector.Detector(weights, device="cpu").id != current_id


def test_face_backend_rejects_tile_identity_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(
            LibreYOLO=lambda *_unused_args, **_unused_kwargs: SimpleNamespace(
                names={0: "1m"}
            )
        ),
    )
    weights = tmp_path / "face.pt"
    weights.write_bytes(b"one")
    with pytest.raises(ValueError, match="exactly one class"):
        detector.Detector(weights, device="cpu")


def test_graph_metadata_override_and_effective_policy_invalidate_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(detector, "select_device", lambda device: device)
    scopes = []
    model = SimpleNamespace(
        names={0: "face"},
        family="yolo9",
        size="s",
        cuda_graph_scope=lambda mode: scopes.append(mode) or nullcontext(),
    )
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(LibreYOLO=lambda *_unused_a, **_unused_k: model),
    )
    monkeypatch.setattr(detector.yolo9_direct, "detect", lambda *_unused_args: ([], []))
    monkeypatch.setattr(detector.metadata, "version", lambda _: "1.5.0")
    monkeypatch.setattr(
        detector, "runtime_signature", lambda device: {"device": str(device)}
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face")
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "libreyolo",
                "architecture": "yolo9-s",
                "classes": {"0": "face"},
                "weights_sha256": hashlib.sha256(b"face").hexdigest(),
                "inference": {"cuda_graph": True},
            }
        )
    )
    automatic = detector.Detector(weights, device="cuda:0")
    eager = detector.Detector(
        weights, device="cuda:0", settings=detector.InferenceOptions(cuda_graph=False)
    )
    assert automatic.cuda_graph
    assert not eager.cuda_graph
    assert automatic.id != eager.id
    automatic.predict(np.zeros((10, 10, 3), np.uint8))
    eager.predict(np.zeros((10, 10, 3), np.uint8))
    assert scopes == [True, False]
    assert not detector.Detector(weights, device="cpu").cuda_graph
    monkeypatch.setattr(detector, "GRAPH_POLICY", "different-bounded-capture-policy")
    assert detector.Detector(weights, device="cuda:0").id != automatic.id
    assert (
        detector.Detector(
            weights,
            device="cuda:0",
            settings=detector.InferenceOptions(cuda_graph=False),
        ).id
        == eager.id
    )


def test_shared_predictions_stay_locked_until_boxes_are_materialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    materializing, release, attempted, second_framework = (Event() for _ in range(4))
    calls = []

    def forward(tensor: torch.Tensor) -> torch.Tensor:
        calls.append(tensor)
        if len(calls) > 1:
            second_framework.set()
        return tensor

    def postprocess(*_unused_args: object, **_unused_kwargs: object) -> dict:
        materializing.set()
        assert release.wait(2)
        return {"num_detections": 1, "boxes": [[1.0, 2.0, 3.0, 4.0]], "scores": [0.9]}

    model = SimpleNamespace(
        names={0: "face"},
        family="yolo9",
        device=torch.device("cpu"),
        cuda_graph_scope=lambda _mode: nullcontext(),
        _forward_graphed=forward,
        _postprocess=postprocess,
    )
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(LibreYOLO=lambda *_unused_a, **_unused_kw: model),
    )
    monkeypatch.setattr(detector.metadata, "version", lambda _: "1.5.0")
    monkeypatch.setattr(detector, "runtime_signature", lambda _: {})
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face")
    current = detector.Detector(weights, device="cpu")
    image = np.zeros((10, 10, 3), np.uint8)

    def another_request() -> list[detector.Det]:
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
        assert second_framework.is_set()
        assert len(calls) == 2


def test_face_backend_rejects_other_families_with_different_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(
            LibreYOLO=lambda *_unused_args, **_unused_kwargs: SimpleNamespace(
                names={0: "face"}, family="yolox"
            )
        ),
    )
    weights = tmp_path / "face.pt"
    weights.write_bytes(b"one")
    with pytest.raises(ValueError, match="standard yolo9"):
        detector.Detector(weights, device="cpu")


def test_metadata_selects_calibrated_defaults_and_explicit_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = SimpleNamespace(names={0: "face"}, family="yolo9", size="s")
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(LibreYOLO=lambda *_unused_a, **_unused_k: model),
    )
    monkeypatch.setattr(detector.metadata, "version", lambda _name: "1.5.0")
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face checkpoint")
    meta: dict = {
        "schema_version": 1,
        "backend": "libreyolo",
        "architecture": "yolo9-s",
        "classes": {"0": "face"},
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "inference": {"imgsz": 1024, "confidence": 0.02, "iou": 0.5},
    }
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta))
    automatic = detector.Detector(weights, device="cpu")
    assert (automatic.conf, automatic.imgsz, automatic.iou) == (0.02, 1024, 0.5)
    explicit = detector.Detector(
        weights,
        device="cpu",
        settings=detector.InferenceOptions(confidence=0.15, imgsz=640),
    )
    assert (explicit.conf, explicit.imgsz, explicit.iou) == (0.15, 640, 0.5)
    assert explicit.id != automatic.id
    meta = json.loads(path.read_text())
    meta["inference"]["confidence"] = "invalid metadata default"
    path.write_text(json.dumps(meta))
    # Explicit settings replace defaults before validation, as for valid metadata.
    assert (
        detector.Detector(
            weights, device="cpu", settings=detector.InferenceOptions(confidence=0.15)
        ).conf
        == 0.15
    )
    with pytest.raises(ValueError, match="confidence"):
        detector.Detector(weights, device="cpu")
    meta["inference"]["confidence"] = 0.02
    path.write_text(json.dumps(meta))
    meta["backend"] = "unsupported"
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="metadata backend"):
        detector.Detector(weights, device="cpu")
    meta["backend"] = "libreyolo"
    meta["classes"] = {"0": "back"}
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="classes differ"):
        detector.Detector(weights, device="cpu")
    weights.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        detector.Detector(weights, device="cpu")


@pytest.mark.parametrize(
    ("settings", "error"),
    [
        ({"confidence": float("nan")}, ValueError),
        ({"confidence": 0}, ValueError),
        ({"iou": 2}, ValueError),
        ({"imgsz": 0}, ValueError),
        ({"cuda_graph": "yes"}, TypeError),
    ],
)
def test_invalid_inference_settings_fail_before_loading_backend(
    tmp_path: Path, settings: dict, error: type[Exception]
) -> None:
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face checkpoint")
    with pytest.raises(error, match="Detector"):
        detector.Detector(
            weights,
            device="cpu",
            settings=detector.InferenceOptions(**settings),
        )


def test_default_face_backend_preserves_recognition_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recognition cache identity includes the exact supported inference contract."""
    model = SimpleNamespace(names={0: "face"}, family="yolo9")
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(LibreYOLO=lambda *_unused_a, **_unused_kw: model),
    )
    monkeypatch.setattr(detector.metadata, "version", lambda _: "1.5.0")
    monkeypatch.setattr(detector, "runtime_signature", lambda _: {"device": "cpu"})
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face")
    current = detector.Detector(weights, device="cpu")
    identity = {
        "weights_sha256": hashlib.sha256(b"face").hexdigest(),
        "backend": "libreyolo",
        "version": "1.5.0",
        "imgsz": 1024,
        "conf": 0.15,
        "iou": 0.5,
        "color": "BGR",
        "classes": {0: "face"},
        "cuda_graph": False,
        "graph_policy": None,
        "runtime": {"device": "cpu"},
        "preprocessing": "libreyolo9-top-left-rect-stride32-v1",
    }
    assert (
        current.id
        == "detector:"
        + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    )


@pytest.mark.parametrize(
    "error", [RuntimeError("weights load failed"), OSError("unreadable checkpoint")]
)
def test_backend_loading_error_preserves_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*_unused_args: object, **_unused_kwargs: object) -> None:
        raise error

    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=fail))
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"checkpoint")
    with pytest.raises(type(error)) as caught:
        detector.Detector(weights, device="cpu")
    assert caught.value is error


def test_lazy_detector_resolves_identity_before_loading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache checks need no weights; a changed checkpoint cannot be loaded later."""
    loaded = []
    model = SimpleNamespace(
        names={0: "face"}, family="yolo9", cuda_graph_scope=lambda _mode: nullcontext()
    )
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(
            LibreYOLO=lambda *args, **_unused_k: loaded.append(args) or model
        ),
    )
    monkeypatch.setattr(detector.yolo9_direct, "detect", lambda *_unused: ([], []))
    monkeypatch.setattr(detector.metadata, "version", lambda _: "1.5.0")
    monkeypatch.setattr(detector, "select_device", lambda _device: "cpu")
    monkeypatch.setattr(detector, "runtime_signature", lambda _: {"device": "cpu"})
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face")
    lazy = detector.LazyDetector(weights)
    assert lazy.id == detector.detector_config(weights).id
    assert loaded == []
    image = np.zeros((32, 32, 3), np.uint8)
    assert lazy.predict_batch([image, image]) == [[], []]
    assert lazy.predict(image) == []
    assert len(loaded) == 1
    assert lazy.id == detector.Detector(weights).id
    stale = detector.LazyDetector(weights)
    weights.write_bytes(b"replaced")
    with pytest.raises(RuntimeError, match="changed after evidence was validated"):
        stale.predict(image)
