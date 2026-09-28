"""Evaluation must count wrong classes and missing annotations honestly."""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from video2tenhou.eval import evaluate_detector_images, eval_perception
from video2tenhou.perception.detector import Det
from video2tenhou.perception.evidence_policy import DEFAULT_POLICY


def test_class_mismatch_is_not_a_localization_true_positive(tmp_path):
    (tmp_path / "images/val").mkdir(parents=True)
    (tmp_path / "labels/val").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images/val/hand_1.png"), np.zeros((10, 10, 3), np.uint8))
    (tmp_path / "labels/val/hand_1.txt").write_text("0 .5 .5 .4 .4\n")
    det = SimpleNamespace(predict=lambda image: [Det((3, 3, 7, 7), .9, True)])
    audit = tmp_path / "predictions.jsonl"
    metrics = evaluate_detector_images(det, tmp_path, predictions_path=audit)
    assert metrics["hand"]["precision"] == metrics["hand"]["recall"] == 0
    assert metrics["hand"]["gt"] == metrics["hand"]["predicted"] == 1
    assert json.loads(audit.read_text())["matches"] == []


def test_missing_label_is_not_a_negative(tmp_path):
    (tmp_path / "images/val").mkdir(parents=True)
    (tmp_path / "labels/val").mkdir(parents=True)
    cv2.imwrite(str(tmp_path / "images/val/meld_1.png"), np.zeros((10, 10, 3), np.uint8))
    det = SimpleNamespace(predict=lambda image: [])
    with pytest.raises(FileNotFoundError):
        evaluate_detector_images(det, tmp_path)
    (tmp_path / "labels/val/meld_1.txt").write_text("")
    assert evaluate_detector_images(det, tmp_path)["meld"]["images"] == 1


@pytest.mark.parametrize("rejected", [False, True])
def test_perception_counts_misses_and_reviewed_negatives(tmp_path, monkeypatch, rejected):
    from video2tenhou import layout
    from video2tenhou.perception import classifier, detector, reader
    from video2tenhou.train import data

    frame_dir = tmp_path / "recording" / "frames"
    frame_dir.mkdir(parents=True)
    cv2.imwrite(str(frame_dir / "1.000.png"), np.zeros((20, 40, 3), np.uint8))
    boxes = [dict(tile="1m", quad=[[1, 1], [10, 1], [10, 15], [1, 15]]),
             dict(tile="2m", quad=[[20, 1], [30, 1], [30, 15], [20, 15]])]
    labels = [dict(t=1., kind="hand", corner="TL", boxes=boxes),
              dict(t=1., kind="meld", corner="TL", boxes=[], melds=[]),
              dict(t=1., kind="pond", corner="TL", boxes=[dict(boxes[0], role="tile", row=0, col=0)])]
    monkeypatch.setattr(data, "load_labels", lambda _: labels)
    monkeypatch.setattr(data, "hand_table", lambda *_: [])
    monkeypatch.setattr(data, "hand_of", lambda *_: 4)
    monkeypatch.setattr(data, "region_upright", lambda frame, *_: (frame, np.eye(3)))
    monkeypatch.setattr(layout.Calibration, "load", lambda *_: None)
    detector_inputs = []
    monkeypatch.setattr(detector, "Detector", lambda **kwargs: detector_inputs.append(kwargs) or SimpleNamespace(evidence_policy=DEFAULT_POLICY))
    classifier_inputs = []
    monkeypatch.setattr(classifier, "Classifier", lambda model_dir: classifier_inputs.append(model_dir) or SimpleNamespace(classes=data.CLASSES))
    probability = np.zeros(len(data.CLASSES)); probability[0] = 1

    def reading(_frame, _cal, name, *_models, **_kwargs):
        found = [reader.Box((1, 1, 10, 15), .9, False, probability.copy())]
        if name.startswith("hand"):
            # A revealed extra should not count as part of the annotated standing row.
            found.append(reader.Box((31, 1, 39, 15), .9, False, probability.copy(), role="extra"))
        if name.startswith("pond"):
            found[0].role = "other"
            found[0].row, found[0].col = 0, 0
        return reader.Reading(1., name, (40, 20), found, rejected=rejected)

    monkeypatch.setattr(reader, "read_region", reading)
    audit = tmp_path / "readings.jsonl"
    result = eval_perception("recording.mp4", tmp_path, predictions_path=audit, classifier_dir=tmp_path / "candidate")
    assert classifier_inputs == [tmp_path / "candidate"]
    assert detector_inputs == [dict(backend=None, conf=None)]
    assert result["hand"]["identity"] == 1
    assert result["hand"]["correct_of_gt"] == .5
    assert result["hand"]["usable_correct_of_gt"] == (0 if rejected else .5)
    assert result["hand"]["predicted"] == 1
    assert result["meld"]["gt"] == 0 and result["meld"]["predicted"] == 1
    assert result["pond"]["correct_of_gt"] == 1
    assert result["pond"]["usable_correct_of_gt"] == result["pond"]["structured_slot_recall"] == 0
    assert len(audit.read_text().splitlines()) == 3
    hand_audit = json.loads(audit.read_text().splitlines()[0])
    assert len(hand_audit["reading"]["boxes"]) == 1
    assert len(hand_audit["raw_reading"]["boxes"]) == 2
    assert hand_audit["raw_reading"]["boxes"][1]["role"] == "extra"


def test_observation_command_honors_explicit_workspace(tmp_path, monkeypatch, capsys):
    from video2tenhou import eval as evaluation
    seen = []
    monkeypatch.setattr(evaluation, "eval_observations", lambda video, work: seen.append((video, work)) or {})
    evaluation.main(["observations", "recording.mp4", "--work", str(tmp_path)])
    assert seen == [("recording.mp4", tmp_path)]
    assert json.loads(capsys.readouterr().out) == {}
