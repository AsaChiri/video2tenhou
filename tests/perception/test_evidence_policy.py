# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Retention changes derived evidence without rewriting raw recognition."""

import hashlib
import json
import sys
from copy import deepcopy
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from tests.paths import DATA
from video2tenhou.eval import evaluate_retained_reading
from video2tenhou.perception import detector
from video2tenhou.perception.detector_metadata import InferenceOptions
from video2tenhou.perception.evidence_policy import (
    DEFAULT_POLICY,
    load_policy,
    prepare_reading,
    resolve_policy,
)
from video2tenhou.train.data import CLASSES
from video2tenhou.train.export_detector import write_runtime_metadata

if TYPE_CHECKING:
    from video2tenhou.perception.evidence_policy import EvidencePolicy


if TYPE_CHECKING:
    from pathlib import Path


def _policy(
    stage: str = "sparse", kind: str = "hand", value: float = 0.1
) -> "EvidencePolicy":
    data = DEFAULT_POLICY.to_dict()
    data[stage][kind] = value
    return resolve_policy(data)


def _box(x: "float", confidence: "float", *, none: float = 0.0) -> dict:
    posterior = [0.0] * len(CLASSES)
    posterior[CLASSES.index("1s")] = 1 - none
    posterior[CLASSES.index("none")] = none
    return {
        "xyxy": [x, 10.0, x + 10.0, 30.0],
        "conf": confidence,
        "sideways": False,
        "p": posterior,
        "role": "tile",
    }


def test_default_stage_boundaries_and_none_filter_preserve_raw_input() -> None:
    """Verify default stage boundaries and none filter preserve raw input."""
    reading = {
        "t": 1.0,
        "region": "meld:TL",
        "size": [100, 60],
        "rejected": False,
        "boxes": [_box(10.0, 0.2), _box(21.0, 0.35), _box(32.0, 0.5, none=0.5)],
    }
    before = deepcopy(reading)
    sparse = prepare_reading(
        "meld", reading, stage="sparse", none_index=CLASSES.index("none")
    )
    dense = prepare_reading("meld", reading, stage="dense")
    assert [box["conf"] for box in sparse["boxes"]] == [0.35]
    assert [box["conf"] for box in dense["boxes"]] == [0.2, 0.35, 0.5]
    assert all(box["role"] == "tile" and box["group"] == 0 for box in dense["boxes"])
    assert reading == before
    assert sparse is not reading
    assert sparse["boxes"][0] is not reading["boxes"][1]


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -0.1, 1.1, "0.2"])
def test_invalid_numeric_policy_never_silently_falls_back(bad: "float") -> None:
    """Verify invalid numeric policy never silently falls back."""
    data = DEFAULT_POLICY.to_dict()
    data["sparse"]["hand"] = bad
    with pytest.raises(ValueError, match="Evidence floors must be finite numbers"):
        resolve_policy(data)


def test_metadata_schema_and_stage_identity_are_complete_and_independent(
    tmp_path: "Path",
) -> None:
    """Verify metadata schema and stage identity are complete and independent."""
    sparse = _policy()
    dense = _policy("dense")
    assert sparse.fingerprint_for("dense") == DEFAULT_POLICY.fingerprint_for("dense")
    assert dense.fingerprint_for("sparse") == DEFAULT_POLICY.fingerprint_for("sparse")
    assert sparse.fingerprint != DEFAULT_POLICY.fingerprint != dense.fingerprint
    data = sparse.to_dict()
    data["sparse"]["hand"] = 0.9
    assert sparse.minimum("sparse", "hand") == 0.1
    for bad in (
        {},
        dict(DEFAULT_POLICY.to_dict(), schema_version=True),
        dict(DEFAULT_POLICY.to_dict(), unexpected=True),
    ):
        with pytest.raises(
            ValueError, match="Unsupported or incomplete evidence policy"
        ):
            resolve_policy(bad)
    incomplete = DEFAULT_POLICY.to_dict()
    del incomplete["dense"]["meld"]
    with pytest.raises(ValueError, match="must specify exactly hand, pond and meld"):
        resolve_policy(incomplete)
    path = tmp_path / "meta.json"
    assert load_policy(path) is DEFAULT_POLICY
    path.write_text(
        json.dumps({"schema_version": 1, "evidence_policy": sparse.to_dict()})
    )
    assert load_policy(path) == sparse
    path.write_text('{"schema_version":1,"evidence_policy":{}}')
    with pytest.raises(ValueError, match="Unsupported or incomplete evidence policy"):
        load_policy(path)
    path.write_text("broken")
    with pytest.raises(json.JSONDecodeError, match="Expecting value"):
        load_policy(path)


def test_known_human_1s_exposes_reader_vs_retained_loss_without_raw_mutation() -> None:
    """Verify known human 1s exposes reader vs retained loss without raw mutation."""
    fixture = json.loads((DATA / "retention_known_1s.json").read_text())
    reading, ground_truth = fixture["reading"], fixture["ground_truth"]
    original = deepcopy(reading)
    default = evaluate_retained_reading(reading, ground_truth, CLASSES, stage="sparse")
    # Synthetic lower floor proves plumbing only; it does not qualify this policy.
    candidate = evaluate_retained_reading(
        reading, ground_truth, CLASSES, stage="sparse", policy=_policy()
    )
    assert ground_truth[13]["tile"] == "1s"
    assert 13 not in default["correct_gt_indices"]
    assert 13 in candidate["correct_gt_indices"]
    assert set(default["correct_gt_indices"]) <= set(candidate["correct_gt_indices"])
    assert reading == original
    assert not candidate["rejected"]


def test_restructure_changes_role_before_evaluation_matches() -> None:
    """Verify restructure changes role before evaluation matches."""
    reading = {
        "t": 1.0,
        "region": "hand:TR",
        "size": [150, 60],
        "rejected": False,
        "boxes": [_box(10.0, 0.9), _box(90.0, 0.3), _box(101.0, 0.3)],
    }
    gt = [{"xyxy": reading["boxes"][0]["xyxy"], "tile": "1s", "role": "tile"}]
    # The unfiltered longest group is rightmost, so the left human target is extra.
    loose = evaluate_retained_reading(reading, gt, CLASSES, stage="sparse")
    strict = evaluate_retained_reading(
        reading, gt, CLASSES, stage="sparse", policy=_policy(value=0.4)
    )
    assert loose["correct"] == 0
    assert strict["correct"] == 1
    assert loose["reading"]["boxes"][-1]["role"] == "extra"


def test_policy_only_metadata_change_keeps_detector_recognition_identity(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify policy only metadata change keeps detector recognition identity."""
    model = SimpleNamespace(names={0: "face"}, family="yolo9", size="s")
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(LibreYOLO=lambda *_unused_a, **_unused_k: model),
    )
    monkeypatch.setattr(detector.metadata, "version", lambda _: "1.5.0")
    monkeypatch.setattr(detector, "runtime_signature", lambda _: {})
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face")
    metadata = {
        "schema_version": 1,
        "backend": "libreyolo",
        "architecture": "yolo9-s",
        "classes": {"0": "face"},
        "weights_sha256": hashlib.sha256(b"face").hexdigest(),
    }
    path = weights.with_name("meta.json")
    path.write_text(json.dumps(metadata))
    first = detector.Detector(weights, device="cpu")
    metadata["evidence_policy"] = _policy().to_dict()
    path.write_text(json.dumps(metadata))
    second = detector.Detector(weights, device="cpu")
    assert first.id == second.id
    assert first.evidence_policy.fingerprint_for(
        "sparse"
    ) != second.evidence_policy.fingerprint_for("sparse")


def test_export_records_explicit_policy_with_checkpoint_hash(tmp_path: "Path") -> None:
    """Verify export records explicit policy with checkpoint hash."""
    weights = tmp_path / "weights.pt"
    torch.save(
        {
            "model_family": "yolo9",
            "size": "s",
            "names": {0: "face"},
            "nc": 1,
            "task": "detect",
        },
        weights,
    )
    path = write_runtime_metadata(
        weights, settings=InferenceOptions(confidence=0.08), evidence_policy=_policy()
    )
    saved = json.loads(path.read_text())
    assert saved["evidence_policy"] == _policy().to_dict()
    assert saved["weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
