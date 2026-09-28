"""Deployment export must preserve the selected inference state exactly."""
import json
import hashlib
import torch
import pytest

from video2tenhou.train.export_detector import export_inference_checkpoint
from video2tenhou.train import export_detector


def test_export_keeps_ema_inference_state_not_raw_training_weights(tmp_path):
    source, target = tmp_path / "training.pt", tmp_path / "runtime.pt"
    selected = {"weight": torch.tensor([.125, -2.5], dtype=torch.float32), "count": torch.tensor(7)}
    checkpoint = dict(model_family="yolo9", size="s", names={0: "face"}, nc=1, model=selected,
                      train_model={"weight": torch.ones(2)}, ema=selected,
                      optimizer={"state": torch.ones(1000)}, ema_updates=42,
                      is_ema_weights=True, imgsz=1024)
    torch.save(checkpoint, source)
    before = source.read_bytes()
    report = export_inference_checkpoint(source, target)
    actual = torch.load(target, weights_only=True)
    assert source.read_bytes() == before
    assert report["exact_tensor_equality"] and report["model_tensors"] == 2
    assert report["target_bytes"] < report["source_bytes"]
    assert torch.equal(actual["model"]["weight"], selected["weight"])
    assert actual["is_ema_weights"] and actual["names"] == {0: "face"}
    assert "optimizer" not in actual and "train_model" not in actual and "ema" not in actual
    with pytest.raises(FileExistsError):
        export_inference_checkpoint(source, target)
    assert torch.equal(torch.load(target, weights_only=True)["model"]["weight"], selected["weight"])


def test_export_rejects_another_family_or_class_contract(tmp_path):
    source, target = tmp_path / "training.pt", tmp_path / "runtime.pt"
    for family, names in (("yolox", {0: "face"}), ("yolo9", {0: "1m"})):
        torch.save(dict(model_family=family, names=names, model={"weight": torch.ones(2)}), source)
        with pytest.raises(ValueError):
            export_inference_checkpoint(source, target)
        assert not target.exists()


def test_changed_source_does_not_leave_a_publishable_export(tmp_path, monkeypatch):
    source, target = tmp_path / "training.pt", tmp_path / "runtime.pt"
    torch.save(dict(model_family="yolo9", names={0: "face"}, nc=1, model={"weight": torch.ones(2)}), source)
    original_hash = export_detector.file_hash
    reads = 0

    def changed_hash(path):
        nonlocal reads
        if path == source:
            reads += 1
            if reads == 2:
                return "changed"
        return original_hash(path)

    monkeypatch.setattr(export_detector, "file_hash", changed_hash)
    with pytest.raises(RuntimeError, match="changed during export"):
        export_inference_checkpoint(source, target)
    assert source.exists() and not target.exists()


def test_runtime_metadata_binds_checkpoint_and_explicit_operating_point(tmp_path):
    weights = tmp_path / "weights.pt"
    torch.save(dict(model_family="yolo9", size="s", names={0: "face"}, nc=1), weights)
    path = export_detector.write_runtime_metadata(weights, confidence=.08, cuda_graph=True)
    metadata = json.loads(path.read_text())
    assert metadata["weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert (metadata["backend"], metadata["architecture"], metadata["classes"]) == ("libreyolo", "yolo9-s", {"0": "face"})
    assert metadata["inference"] == dict(imgsz=1024, confidence=.08, iou=.5, cuda_graph=True)
    with pytest.raises(FileExistsError):
        export_detector.write_runtime_metadata(weights, confidence=.15)


def test_metadata_does_not_accept_an_unknown_checkpoint_contract(tmp_path):
    weights = tmp_path / "weights.pt"
    torch.save(dict(model_family="yolo9", size="s", names={0: "1m"}, nc=1), weights)
    with pytest.raises(ValueError, match="standard YOLO9 face"):
        export_detector.write_runtime_metadata(weights, confidence=.08)
    assert not weights.with_name("meta.json").exists()


@pytest.mark.parametrize("settings", [
    {"confidence": "0.2"}, {"confidence": None}, {"confidence": True},
    {"confidence": float("inf")}, {"iou": 0}, {"imgsz": True}, {"cuda_graph": 1},
])
def test_metadata_rejects_invalid_runtime_settings_before_reading_checkpoint(tmp_path, settings):
    weights = tmp_path / "weights.pt"
    # Invalid inference settings must fail before a checkpoint is loaded or metadata is written.
    with pytest.raises(ValueError, match="Detector"):
        export_detector.write_runtime_metadata(weights, **{**dict(confidence=.08), **settings})
    assert not weights.with_name("meta.json").exists()
