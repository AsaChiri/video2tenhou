# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Changing calibrated model content must invalidate evidence, irrespective of paths."""

import json
import os
from typing import TYPE_CHECKING

import pytest

from video2tenhou.perception import classifier

if TYPE_CHECKING:
    from typing import Self


if TYPE_CHECKING:
    from pathlib import Path


def test_identity_tracks_checkpoint_metadata_and_preprocessing(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify identity tracks checkpoint metadata and preprocessing."""

    class Model:
        def load_state_dict(self, state: "dict") -> None:
            pass

        def to(self, device: "str") -> "Self":
            return self

        def eval(self) -> "Self":
            return self

    monkeypatch.setattr(classifier, "make_model", lambda _n: Model())
    monkeypatch.setattr(classifier.torch, "load", lambda *_unused_a, **_unused_k: {})
    # This test hashes simulated runtimes; kernel/device checks have their own tests.
    monkeypatch.setattr(classifier, "select_device", lambda device: device)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"checkpoint-a")
    meta = {"classes": classifier.CLASSES, "temperature": 1.0}
    (tmp_path / "meta.json").write_text(json.dumps(meta))

    def identity(device: str = "cpu") -> "str":
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


def test_classifier_requires_declared_calibration_temperature(tmp_path: "Path") -> None:
    """Verify classifier requires declared calibration temperature."""
    (tmp_path / "meta.json").write_text(json.dumps({"classes": classifier.CLASSES}))
    with pytest.raises(KeyError, match="temperature"):
        classifier.Classifier(tmp_path, device="cpu")
