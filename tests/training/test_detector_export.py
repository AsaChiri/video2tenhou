# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Deployment export must preserve the selected inference state exactly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from video2tenhou.train import export_detector
from video2tenhou.train.export_detector import export_inference_checkpoint


def test_export_keeps_ema_inference_state_not_raw_training_weights(
    tmp_path: Path,
) -> None:
    source, target = tmp_path / "training.pt", tmp_path / "runtime.pt"
    selected = {
        "weight": torch.tensor([0.125, -2.5], dtype=torch.float32),
        "count": torch.tensor(7),
    }
    checkpoint = {
        "model_family": "yolo9",
        "task": "detect",
        "size": "s",
        "names": {0: "face"},
        "nc": 1,
        "model": selected,
        "train_model": {"weight": torch.ones(2)},
        "ema": selected,
        "optimizer": {"state": torch.ones(1000)},
        "ema_updates": 42,
        "is_ema_weights": True,
        "imgsz": 1024,
    }
    torch.save(checkpoint, source)
    before = source.read_bytes()
    report = export_inference_checkpoint(source, target)
    actual = torch.load(target, weights_only=True)
    assert source.read_bytes() == before
    assert report["exact_tensor_equality"]
    assert report["model_tensors"] == 2
    assert report["target_bytes"] < report["source_bytes"]
    assert torch.equal(actual["model"]["weight"], selected["weight"])
    assert actual["is_ema_weights"]
    assert actual["names"] == {0: "face"}
    assert "optimizer" not in actual
    assert "train_model" not in actual
    assert "ema" not in actual
    with pytest.raises(FileExistsError):
        export_inference_checkpoint(source, target)
    assert torch.equal(
        torch.load(target, weights_only=True)["model"]["weight"], selected["weight"]
    )


def test_export_rejects_another_family_or_class_contract(tmp_path: Path) -> None:
    source, target = tmp_path / "training.pt", tmp_path / "runtime.pt"
    for family, names in (("yolox", {0: "face"}), ("yolo9", {0: "1m"})):
        torch.save(
            {
                "model_family": family,
                "names": names,
                "model": {"weight": torch.ones(2)},
            },
            source,
        )
        with pytest.raises(
            ValueError, match=r"Expected a standard|Expected the single face"
        ):
            export_inference_checkpoint(source, target)
        assert not target.exists()


def test_changed_source_does_not_leave_a_publishable_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, target = tmp_path / "training.pt", tmp_path / "runtime.pt"
    torch.save(
        {
            "model_family": "yolo9",
            "task": "detect",
            "names": {0: "face"},
            "nc": 1,
            "model": {"weight": torch.ones(2)},
        },
        source,
    )
    original_hash = export_detector.file_hash
    reads = 0

    def changed_hash(path: Path) -> str:
        nonlocal reads
        if path == source:
            reads += 1
            if reads == 2:
                return "changed"
        return original_hash(path)

    monkeypatch.setattr(export_detector, "file_hash", changed_hash)
    with pytest.raises(RuntimeError, match="changed during export"):
        export_inference_checkpoint(source, target)
    assert source.exists()
    assert not target.exists()


def test_runtime_metadata_binds_checkpoint_and_explicit_operating_point(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights.pt"
    torch.save(
        {
            "model_family": "yolo9",
            "task": "detect",
            "size": "s",
            "names": {0: "face"},
            "nc": 1,
        },
        weights,
    )
    path = export_detector.write_runtime_metadata(
        weights,
        settings=export_detector.InferenceOptions(confidence=0.08, cuda_graph=True),
    )
    metadata = json.loads(path.read_text())
    assert (
        metadata["weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    )
    assert (metadata["backend"], metadata["architecture"], metadata["classes"]) == (
        "libreyolo",
        "yolo9-s",
        {"0": "face"},
    )
    assert metadata["inference"] == {
        "imgsz": 1024,
        "confidence": 0.08,
        "iou": 0.5,
        "cuda_graph": True,
    }
    with pytest.raises(FileExistsError):
        export_detector.write_runtime_metadata(
            weights, settings=export_detector.InferenceOptions(confidence=0.15)
        )


def test_metadata_does_not_accept_an_unknown_checkpoint_contract(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "weights.pt"
    torch.save(
        {"model_family": "yolo9", "size": "s", "names": {0: "1m"}, "nc": 1}, weights
    )
    with pytest.raises(ValueError, match="standard YOLO9 face"):
        export_detector.write_runtime_metadata(
            weights, settings=export_detector.InferenceOptions(confidence=0.08)
        )
    assert not weights.with_name("meta.json").exists()


def test_export_and_metadata_require_declared_detection_task(tmp_path: Path) -> None:
    source, target = tmp_path / "training.pt", tmp_path / "runtime.pt"
    torch.save(
        {
            "model_family": "yolo9",
            "size": "s",
            "names": {0: "face"},
            "nc": 1,
            "model": {"weight": torch.ones(2)},
        },
        source,
    )
    with pytest.raises(ValueError, match="detection checkpoint"):
        export_inference_checkpoint(source, target)
    with pytest.raises(ValueError, match="standard YOLO9 face"):
        export_detector.write_runtime_metadata(
            source, settings=export_detector.InferenceOptions(confidence=0.08)
        )
    assert not target.exists()
    assert not source.with_name("meta.json").exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"confidence": "0.2"},
        {"confidence": None},
        {"confidence": True},
        {"confidence": float("inf")},
        {"iou": 0},
        {"imgsz": True},
        {"cuda_graph": 1},
    ],
)
def test_metadata_rejects_invalid_runtime_settings_before_reading_checkpoint(
    tmp_path: Path, settings: dict
) -> None:
    weights = tmp_path / "weights.pt"
    # Invalid inference settings must fail before a checkpoint is loaded or metadata is
    # written.
    runtime_settings: dict = {"confidence": 0.08, **settings}
    error = TypeError if "cuda_graph" in settings else ValueError
    with pytest.raises(error, match="Detector"):
        export_detector.write_runtime_metadata(
            weights, settings=export_detector.InferenceOptions(**runtime_settings)
        )
    assert not weights.with_name("meta.json").exists()
