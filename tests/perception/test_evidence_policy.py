"""Retention changes derived evidence without rewriting raw recognition."""

from tests.paths import DATA
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from video2tenhou.eval import evaluate_retained_reading
from video2tenhou.perception.evidence_policy import (
    DEFAULT_POLICY, load_policy, prepare_reading, resolve_policy,
)
from video2tenhou.train.data import CLASSES


def _policy(stage="sparse", kind="hand", value=.1):
    data = DEFAULT_POLICY.to_dict()
    data[stage][kind] = value
    return resolve_policy(data)


def _box(x, confidence, *, none=0.):
    posterior = [0.] * len(CLASSES)
    posterior[CLASSES.index("1s")] = 1 - none
    posterior[CLASSES.index("none")] = none
    return dict(xyxy=[x, 10., x + 10., 30.], conf=confidence, sideways=False, p=posterior, role="tile")


def test_default_stage_boundaries_and_none_filter_preserve_raw_input():
    reading = dict(t=1., region="meld:TL", size=[100, 60], rejected=False,
                   boxes=[_box(10., .2), _box(21., .35), _box(32., .5, none=.5)])
    before = deepcopy(reading)
    sparse = prepare_reading("meld", reading, stage="sparse", none_index=CLASSES.index("none"))
    dense = prepare_reading("meld", reading, stage="dense")
    assert [box["conf"] for box in sparse["boxes"]] == [.35]
    assert [box["conf"] for box in dense["boxes"]] == [.2, .35, .5]
    assert all(box["role"] == "tile" and box["group"] == 0 for box in dense["boxes"])
    assert reading == before
    assert sparse is not reading and sparse["boxes"][0] is not reading["boxes"][1]


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), -.1, 1.1, "0.2"])
def test_invalid_numeric_policy_never_silently_falls_back(bad):
    data = DEFAULT_POLICY.to_dict()
    data["sparse"]["hand"] = bad
    with pytest.raises(ValueError):
        resolve_policy(data)


def test_metadata_schema_and_stage_identity_are_complete_and_independent(tmp_path):
    sparse = _policy()
    dense = _policy("dense")
    assert sparse.fingerprint_for("dense") == DEFAULT_POLICY.fingerprint_for("dense")
    assert dense.fingerprint_for("sparse") == DEFAULT_POLICY.fingerprint_for("sparse")
    assert sparse.fingerprint != DEFAULT_POLICY.fingerprint != dense.fingerprint
    data = sparse.to_dict()
    data["sparse"]["hand"] = .9
    assert sparse.minimum("sparse", "hand") == .1
    for bad in ({}, dict(DEFAULT_POLICY.to_dict(), schema_version=True),
                dict(DEFAULT_POLICY.to_dict(), unexpected=True)):
        with pytest.raises(ValueError):
            resolve_policy(bad)
    incomplete = DEFAULT_POLICY.to_dict()
    del incomplete["dense"]["meld"]
    with pytest.raises(ValueError):
        resolve_policy(incomplete)
    path = tmp_path / "meta.json"
    assert load_policy(path) is DEFAULT_POLICY
    path.write_text(json.dumps(dict(schema_version=1, evidence_policy=sparse.to_dict())))
    assert load_policy(path) == sparse
    path.write_text('{"schema_version":1,"evidence_policy":{}}')
    with pytest.raises(ValueError):
        load_policy(path)
    path.write_text("broken")
    with pytest.raises(ValueError):
        load_policy(path)


def test_known_human_1s_exposes_reader_vs_retained_loss_without_raw_mutation():
    fixture = json.loads((DATA / "retention_known_1s.json").read_text())
    reading, ground_truth = fixture["reading"], fixture["ground_truth"]
    original = deepcopy(reading)
    default = evaluate_retained_reading("hand", reading, ground_truth, CLASSES, stage="sparse")
    # Synthetic lower floor proves plumbing only; it does not qualify this policy.
    candidate = evaluate_retained_reading("hand", reading, ground_truth, CLASSES,
                                          stage="sparse", policy=_policy())
    assert ground_truth[13]["tile"] == "1s"
    assert 13 not in default["correct_gt_indices"] and 13 in candidate["correct_gt_indices"]
    assert set(default["correct_gt_indices"]) <= set(candidate["correct_gt_indices"])
    assert reading == original
    assert not candidate["rejected"]


def test_restructure_changes_role_before_evaluation_matches():
    reading = dict(t=1., region="hand:TR", size=[150, 60], rejected=False,
                   boxes=[_box(10., .9), _box(90., .3), _box(101., .3)])
    gt = [dict(xyxy=reading["boxes"][0]["xyxy"], tile="1s", role="tile")]
    # The unfiltered longest group is rightmost, so the left human target is extra.
    loose = evaluate_retained_reading("hand", reading, gt, CLASSES, stage="sparse")
    strict = evaluate_retained_reading("hand", reading, gt, CLASSES, stage="sparse",
                                      policy=_policy(value=.4))
    assert loose["correct"] == 0 and strict["correct"] == 1
    assert loose["reading"]["boxes"][-1]["role"] == "extra"


def test_policy_only_metadata_change_keeps_detector_recognition_identity(tmp_path, monkeypatch):
    from video2tenhou.perception import detector
    model = SimpleNamespace(names={0: "face"}, family="yolo9", size="s")
    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=lambda *a, **k: model))
    monkeypatch.setattr(detector.metadata, "version", lambda _: "1.5.0")
    monkeypatch.setattr(detector, "runtime_signature", lambda _: {})
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"face")
    metadata = dict(schema_version=1, backend="libreyolo", architecture="yolo9-s",
                    classes={"0": "face"}, weights_sha256=hashlib.sha256(b"face").hexdigest())
    path = weights.with_name("meta.json")
    path.write_text(json.dumps(metadata))
    first = detector.Detector(weights, device="cpu")
    metadata["evidence_policy"] = _policy().to_dict()
    path.write_text(json.dumps(metadata))
    second = detector.Detector(weights, device="cpu")
    assert first.id == second.id
    assert first.evidence_policy.fingerprint_for("sparse") != second.evidence_policy.fingerprint_for("sparse")


def test_export_records_explicit_policy_with_checkpoint_hash(tmp_path):
    import torch
    from video2tenhou.train.export_detector import write_runtime_metadata
    weights = tmp_path / "weights.pt"
    torch.save(dict(model_family="yolo9", size="s", names={0: "face"}, nc=1), weights)
    path = write_runtime_metadata(weights, confidence=.08, evidence_policy=_policy())
    saved = json.loads(path.read_text())
    assert saved["evidence_policy"] == _policy().to_dict()
    assert saved["weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
