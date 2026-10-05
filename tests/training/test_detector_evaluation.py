# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation must count wrong classes and missing annotations honestly."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tests.recognition import RecognitionStub
from video2tenhou import eval as evaluation
from video2tenhou import layout
from video2tenhou.eval import eval_perception, evaluate_detector_images
from video2tenhou.perception import classifier, detector, reader
from video2tenhou.perception.detector import Det
from video2tenhou.perception.evidence_policy import DEFAULT_POLICY
from video2tenhou.perception.reader import Reading
from video2tenhou.perception.tiles import CLASSES
from video2tenhou.train import data


@dataclass
class FixedDetector(RecognitionStub):
    """Return controlled detections for evaluator count and label checks."""

    detections: list[Det] = field(default_factory=list)

    def predict(self, _img: np.ndarray, /) -> list[Det]:
        """Supply the same detections for the test's single image."""
        return self.detections


def test_face_boxes_match_one_to_one(tmp_path: Path) -> None:
    """An extra or displaced box is a false positive; non-face labels are refused."""
    (tmp_path / "images/val").mkdir(parents=True)
    (tmp_path / "labels/val").mkdir(parents=True)
    cv2.imwrite(
        str(tmp_path / "images/val/hand_1.png"), np.zeros((10, 10, 3), np.uint8)
    )
    (tmp_path / "labels/val/hand_1.txt").write_text("0 .5 .5 .4 .4\n")
    found = [Det((3, 3, 7, 7), 0.9), Det((0, 0, 2, 2), 0.8)]
    det = FixedDetector("faces", detections=found)
    audit = tmp_path / "predictions.jsonl"
    metrics = evaluate_detector_images(det, tmp_path, predictions_path=audit)
    assert (metrics["hand"]["precision"], metrics["hand"]["recall"]) == (0.5, 1)
    assert json.loads(audit.read_text())["matches"] == [[0, 0]]
    (tmp_path / "labels/val/hand_1.txt").write_text("1 .5 .5 .4 .4\n")
    with pytest.raises(ValueError, match="Invalid face annotation"):
        evaluate_detector_images(det, tmp_path)


def test_missing_label_is_not_a_negative(tmp_path: Path) -> None:
    (tmp_path / "images/val").mkdir(parents=True)
    (tmp_path / "labels/val").mkdir(parents=True)
    cv2.imwrite(
        str(tmp_path / "images/val/meld_1.png"), np.zeros((10, 10, 3), np.uint8)
    )
    det = FixedDetector("empty")
    with pytest.raises(FileNotFoundError):
        evaluate_detector_images(det, tmp_path)
    (tmp_path / "labels/val/meld_1.txt").write_text("")
    assert evaluate_detector_images(det, tmp_path)["meld"]["images"] == 1


@pytest.mark.parametrize("rejected", [False, True])
def test_perception_counts_misses_and_reviewed_negatives(
    *, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rejected: bool
) -> None:
    frame_dir = tmp_path / "recording" / "frames"
    frame_dir.mkdir(parents=True)
    cv2.imwrite(str(frame_dir / "1.000.png"), np.zeros((20, 40, 3), np.uint8))
    boxes = [
        {"tile": "1m", "quad": [[1, 1], [10, 1], [10, 15], [1, 15]]},
        {"tile": "2m", "quad": [[20, 1], [30, 1], [30, 15], [20, 15]]},
    ]
    labels = [
        {"t": 1.0, "kind": "hand", "corner": "TL", "boxes": boxes},
        {"t": 1.0, "kind": "meld", "corner": "TL", "boxes": [], "melds": []},
        {
            "t": 1.0,
            "kind": "pond",
            "corner": "TL",
            "boxes": [dict(boxes[0], role="tile", row=0, col=0)],
        },
    ]
    monkeypatch.setattr(data, "load_labels", lambda _: labels)
    monkeypatch.setattr(data, "hand_table", lambda *_: [])
    monkeypatch.setattr(data, "hand_of", lambda *_: 4)
    monkeypatch.setattr(
        evaluation, "region_upright", lambda frame, *_: (frame, np.eye(3))
    )
    monkeypatch.setattr(layout.Calibration, "load", lambda *_: None)
    detector_inputs = []
    monkeypatch.setattr(
        detector,
        "Detector",
        lambda *args, **kwargs: (
            detector_inputs.append((args, kwargs))
            or SimpleNamespace(evidence_policy=DEFAULT_POLICY)
        ),
    )
    classifier_inputs = []
    monkeypatch.setattr(
        classifier,
        "Classifier",
        lambda model_dir: (
            classifier_inputs.append(model_dir) or SimpleNamespace(classes=CLASSES)
        ),
    )
    probability = np.zeros(len(CLASSES))
    probability[0] = 1

    def reading(
        _image: object, name: str, *_models: object, **_kwargs: object
    ) -> Reading:
        found = [reader.Box((1, 1, 10, 15), 0.9, sideways=False, p=probability.copy())]
        if name.startswith("hand"):
            # A revealed extra should not count as part of the annotated standing row.
            found.append(
                reader.Box(
                    (31, 1, 39, 15),
                    0.9,
                    sideways=False,
                    p=probability.copy(),
                    role="extra",
                )
            )
        if name.startswith("pond"):
            found[0].role = "other"
            found[0].row, found[0].col = 0, 0
        return reader.Reading(1.0, name, (40, 20), found, rejected=rejected)

    monkeypatch.setattr(reader, "read_region", reading)
    audit = tmp_path / "readings.jsonl"
    result = eval_perception(
        "recording.mp4",
        tmp_path,
        predictions_path=audit,
        classifier_dir=tmp_path / "candidate",
    )
    assert classifier_inputs == [tmp_path / "candidate"]
    assert detector_inputs == [
        ((detector.DEFAULT_WEIGHTS,), {"settings": detector.InferenceOptions()})
    ]
    assert result["hand"]["identity"] == 1
    assert result["hand"]["correct_of_gt"] == 0.5
    assert result["hand"]["usable_correct_of_gt"] == (0 if rejected else 0.5)
    assert result["hand"]["predicted"] == 1
    assert result["meld"]["gt"] == 0
    assert result["meld"]["predicted"] == 1
    assert result["pond"]["correct_of_gt"] == 1
    assert (
        result["pond"]["usable_correct_of_gt"]
        == result["pond"]["structured_slot_recall"]
        == 0
    )
    assert len(audit.read_text().splitlines()) == 3
    hand_audit = json.loads(audit.read_text().splitlines()[0])
    assert len(hand_audit["reading"]["boxes"]) == 1
    assert len(hand_audit["raw_reading"]["boxes"]) == 2
    assert hand_audit["raw_reading"]["boxes"][1]["role"] == "extra"


def test_observation_command_honors_explicit_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen = []
    monkeypatch.setattr(
        evaluation,
        "eval_observations",
        lambda video, work: seen.append((video, work)) or {},
    )
    evaluation.main(["observations", "recording.mp4", "--work", str(tmp_path)])
    assert seen == [("recording.mp4", tmp_path)]
    assert json.loads(capsys.readouterr().out) == {}
