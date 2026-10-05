# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Changing calibrated model content must invalidate evidence, irrespective of paths."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Self

import numpy as np
import pytest

from video2tenhou.perception import classifier


def test_identity_tracks_checkpoint_metadata_and_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Model:
        def load_state_dict(self, state: dict) -> None:
            pass

        def to(self, device: str) -> Self:
            return self

        def eval(self) -> Self:
            return self

    monkeypatch.setattr(classifier, "make_model", lambda _n: Model())
    monkeypatch.setattr(classifier.torch, "load", lambda *_unused_a, **_unused_k: {})
    # This test hashes simulated runtimes; kernel/device checks have their own tests.
    monkeypatch.setattr(classifier, "select_device", lambda device: device)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"checkpoint-a")
    meta = {"classes": classifier.CLASSES, "temperature": 1.0}
    (tmp_path / "meta.json").write_text(json.dumps(meta))

    def identity(device: str = "cpu") -> str:
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


def test_classifier_requires_declared_calibration_temperature(tmp_path: Path) -> None:
    (tmp_path / "meta.json").write_text(json.dumps({"classes": classifier.CLASSES}))
    with pytest.raises(KeyError, match="temperature"):
        classifier.Classifier(tmp_path, device="cpu")


def test_lazy_classifier_loads_weights_only_to_classify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache checks need no weights; a changed checkpoint cannot be loaded later."""
    monkeypatch.setattr(classifier, "select_device", lambda _device: "cpu")
    (tmp_path / "weights.pt").write_bytes(b"checkpoint-a")
    meta = {"classes": classifier.CLASSES, "temperature": 0.7}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    loaded = []

    def load(model_dir: Path) -> SimpleNamespace:
        loaded.append(model_dir)
        return SimpleNamespace(
            id=classifier.classifier_config(model_dir).id,
            classify=lambda crops, _sideways: np.ones((len(crops), 1)),
        )

    monkeypatch.setattr(classifier, "Classifier", load)
    lazy = classifier.LazyClassifier(tmp_path)
    assert (lazy.classes, lazy.T, loaded) == (classifier.CLASSES, 0.7, [])
    crop = np.zeros((96, 64, 3), np.uint8)
    assert lazy.classify([crop, crop]).shape == (2, 1)
    lazy.classify([crop])
    assert loaded == [tmp_path]
    stale = classifier.LazyClassifier(tmp_path)
    (tmp_path / "weights.pt").write_bytes(b"checkpoint-b")
    with pytest.raises(RuntimeError, match="changed after evidence was validated"):
        stale.classify([crop])
