# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve draft provenance and prevent implicit acceptance as training labels."""

import hashlib
import json
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pytest

from tools import pseudolabel_images as drafts
from video2tenhou.perception import detector
from video2tenhou.perception.detector import Det

if TYPE_CHECKING:
    from pathlib import Path


def detection(
    box: tuple[float, float, float, float] = (-2, 2, 30, 8),
    confidence: float = 0.8,
    *,
    back: bool = False,
) -> Det:
    """Create a face-detector result with controlled box and confidence."""
    return Det(xyxy=box, conf=confidence, back=back)


def archive_fixture(tmp_path: "Path", members: "list[tuple[str, bytes]]") -> tuple:
    """Create an image archive from explicitly supplied member bytes."""
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        for name, content in members:
            stream.writestr(name, content)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"test-checkpoint-identity")
    return archive, weights


def png() -> "bytes":
    """Encode a small synthetic image as PNG bytes."""
    ok, image = cv2.imencode(".png", np.zeros((10, 20, 3), dtype=np.uint8))
    assert ok
    return image.tobytes()


@dataclass
class PredictionModel:
    """Localization labels accompanying controlled test predictions."""

    names: dict[int, str]


@dataclass
class Predictor:
    """Controlled face predictions implementing the draft export interface."""

    predictions: list[Det]
    model: PredictionModel
    device: str = "fake"
    id: str | None = None
    imgsz: int | None = None
    conf: float | None = None
    iou: float | None = None
    cuda_graph: bool | None = None

    def predict(self, _image: np.ndarray, /) -> list[Det]:
        """Return the supplied predictions without model inference."""
        return self.predictions


def predictor(predictions: list[Det], names: dict[int, str] | None = None) -> Predictor:
    """Return controlled face predictions and model class metadata."""
    return Predictor(predictions, PredictionModel(names or drafts.CLASS_NAMES))


def test_box_clipping_normalization_and_raw_coordinates() -> None:
    """Verify box clipping normalization and raw coordinates."""
    box = drafts.normalized_box(detection(), 20, 10)
    assert box["raw_xyxy"] == [-2, 2, 30, 8]
    assert box["xyxy"] == [0, 2, 20, 8]
    assert box["yolo"] == [0.5, 0.5, 1.0, 0.6]
    assert box["class_name"] == "face"
    assert box["back"] is False


@pytest.mark.parametrize(
    ("box", "confidence"),
    [
        ((0, 0, float("nan"), 5), 0.8),
        ((4, 0, 2, 5), 0.8),
        ((30, 0, 40, 5), 0.8),
        ((0, 0, 4, 5), 1.2),
    ],
)
def test_invalid_prediction_is_rejected(
    box: "tuple[float, float, float, float]", confidence: float
) -> None:
    """Verify invalid prediction is rejected."""
    with pytest.raises(
        ValueError,
        match=r"must be finite|Invalid detection area|no area inside the image",
    ):
        drafts.normalized_box(detection(box, confidence), 20, 10)


def test_face_only_schema_rejects_identity_classes_and_back_predictions() -> None:
    """Verify face only schema rejects identity classes and back predictions."""
    with pytest.raises(ValueError, match="Expected detector classes"):
        drafts.validate_classes({0: "1m", 1: "unknown"})
    drafts.validate_classes({0: "face"})
    with pytest.raises(ValueError, match="cannot export a back"):
        drafts.normalized_box(detection(back=True), 20, 10)


@pytest.mark.parametrize(
    "members",
    [
        [("../outside.png", b"x")],
        [("C:/outside.png", b"x")],
        [("same.png", b"x"), ("SAME.png", b"y")],
    ],
)
def test_zip_traversal_and_duplicate_members_fail_closed(
    tmp_path: "Path", members: list[tuple[str, bytes]]
) -> None:
    """Verify zip traversal and duplicate members fail closed."""
    archive, weights = archive_fixture(tmp_path, members)
    output = tmp_path / "draft"
    with pytest.raises(ValueError, match=r"Unsafe image member|Duplicate image member"):
        drafts.export_drafts(archive, weights, output, predictor=predictor([]))
    assert not json.loads((output / "summary.json").read_text())["complete"]
    assert not list((output / "images").iterdir())
    assert not (tmp_path / "outside.png").exists()


def test_zero_predictions_duplicate_groups_and_source_bytes_are_preserved(
    tmp_path: "Path",
) -> None:
    """Verify zero predictions duplicate groups and source bytes are preserved."""
    content = png()
    archive, weights = archive_fixture(
        tmp_path,
        [
            ("a/tile.png", content),
            ("b/tile.png", content),
            ("ignored.txt", b"not an image"),
        ],
    )
    output = tmp_path / "draft"
    summary = drafts.export_drafts(archive, weights, output, predictor=predictor([]))
    assert summary["complete"]
    assert summary["images_completed"] == 2
    assert summary["images_zero_detections"] == 2
    assert summary["class_counts"] == {0: 0}
    assert summary["image_dimensions"] == {
        "min_width": 20,
        "max_width": 20,
        "min_height": 10,
        "max_height": 10,
    }
    assert summary["exact_duplicate_groups"] == [
        {
            "sha256": hashlib.sha256(content).hexdigest(),
            "members": ["a/tile.png", "b/tile.png"],
        }
    ]
    rows = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text().splitlines()
    ]
    assert len({row["image"] for row in rows}) == 2
    for row in rows:
        assert row["reviewed"] is False
        assert row["status"] == "unreviewed_zero_detections"
        assert (output / row["image"]).read_bytes() == content
        assert (output / row["draft_labels"]).read_text() == ""
    assert not (output / "data.yaml").exists()
    assert not (output / "labels").exists()
    with pytest.raises(FileExistsError):
        drafts.export_drafts(archive, weights, output, predictor=predictor([]))


def test_face_export_and_partial_failure_manifest(tmp_path: "Path") -> None:
    """Verify face export and partial failure manifest."""
    archive, weights = archive_fixture(
        tmp_path, [("good.png", png()), ("bad.png", b"invalid image")]
    )
    names = {0: "face"}
    box = detection((1, 1, 8, 8), 0.9)
    output = tmp_path / "draft"
    with pytest.raises(ValueError, match="Cannot decode"):
        drafts.export_drafts(
            archive, weights, output, predictor=predictor([box], names)
        )
    summary = json.loads((output / "summary.json").read_text())
    assert not summary["complete"]
    assert summary["images_completed"] == 1
    assert summary["class_counts"] == {"0": 1}
    assert summary["classes"]["0"] == "face"
    rows = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text().splitlines()
    ]
    assert rows[0]["status"] == "unreviewed_predictions"
    assert rows[1]["status"] == "error"
    assert (output / rows[0]["draft_labels"]).read_text().startswith("0 ")


def test_export_uses_checkpoint_defaults_and_records_recognition_identity(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify export uses checkpoint defaults and records recognition identity."""
    archive, weights = archive_fixture(tmp_path, [("photo.png", png())])
    metadata = weights.with_name("meta.json")
    metadata.write_text('{"inference":{"confidence":0.08}}')
    calls = []
    model = predictor([detection()])
    model.id, model.imgsz, model.conf, model.iou, model.cuda_graph = (
        "detector:fixture",
        1024,
        0.08,
        0.5,
        True,
    )

    def load(path: "Path", **kwargs: object) -> Predictor:
        calls.append((path, kwargs))
        return model

    monkeypatch.setattr(detector, "Detector", load)
    summary = drafts.export_drafts(archive, weights, tmp_path / "draft", device="cpu")
    assert calls == [(weights.resolve(), {"device": "cpu"})]
    assert summary["settings"]["conf"] == 0.08
    assert summary["recognition_id"] == "detector:fixture"
    assert (
        summary["metadata_sha256"] == hashlib.sha256(metadata.read_bytes()).hexdigest()
    )
    assert summary["classes"] == {0: "face"}
    assert summary["class_counts"] == {0: 1}
