"""Changing calibrated model content must invalidate evidence, irrespective of paths."""

import json
import os

import pytest

from video2tenhou.perception import classifier


def test_identity_tracks_checkpoint_metadata_and_preprocessing(tmp_path, monkeypatch):
    class Model:
        def load_state_dict(self, state):
            pass

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(classifier, "make_model", lambda n: Model())
    monkeypatch.setattr(classifier.torch, "load", lambda *a, **k: {})
    # This test hashes simulated runtimes; kernel/device checks have their own tests.
    monkeypatch.setattr(classifier, "select_device", lambda device: device)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"checkpoint-a")
    meta = {"classes": classifier.CLASSES, "temperature": 1.0}
    (tmp_path / "meta.json").write_text(json.dumps(meta))

    def identity(device="cpu"):
        return classifier.Classifier(tmp_path, device=device).id

    first = identity()
    stamp = weights.stat()
    os.utime(weights, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1000000))
    assert identity() == first
    weights.write_bytes(b"checkpoint-b")
    os.utime(weights, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    second = identity()
    assert second != first
    meta["temperature"] = 2.0
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    third = identity()
    assert third != second
    monkeypatch.setattr(classifier, "PREPROCESSING", "next-pipeline")
    assert identity() != third
    current = identity()
    monkeypatch.setattr(classifier.torch.cuda, "is_available", lambda: False)
    assert identity("cuda:0") != current
    precision = classifier.torch.get_float32_matmul_precision()
    monkeypatch.setattr(
        classifier.torch,
        "get_float32_matmul_precision",
        lambda: "high" if precision != "high" else "highest",
    )
    assert identity() != current


def test_classifier_requires_declared_calibration_temperature(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"classes": classifier.CLASSES}))
    with pytest.raises(KeyError, match="temperature"):
        classifier.Classifier(tmp_path, device="cpu")
