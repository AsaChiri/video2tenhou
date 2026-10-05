# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Draft exports keep provenance and never pass as reviewed training labels."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import pseudolabel_images as drafts
from video2tenhou.perception import detector
from video2tenhou.perception.detector import Det


@dataclass
class FixedDetector:
    """Return the same face boxes for every image."""

    boxes: list[Det] = field(default_factory=list)
    id: str = "detector:fixture"
    conf: float = 0.15

    def predict(self, _image: np.ndarray, /) -> list[Det]:
        """Return the configured predictions without inference."""
        return self.boxes


def png() -> bytes:
    """Encode a 20x10 black image."""
    ok, image = cv2.imencode(".png", np.zeros((10, 20, 3), dtype=np.uint8))
    assert ok
    return image.tobytes()


def archive(tmp_path: Path, members: dict[str, bytes]) -> Path:
    """Write a ZIP with the given members."""
    path = tmp_path / "images.zip"
    with zipfile.ZipFile(path, "w") as stream:
        for name, content in members.items():
            stream.writestr(name, content)
    return path


def test_boxes_are_clipped_and_normalized_with_raw_coordinates_kept() -> None:
    """Out-of-image edges are clipped; raw coordinates stay in predictions."""
    box = drafts.draft_box(Det((-2, 2, 30, 8), 0.8), 20, 10)
    assert box["raw_xyxy"] == [-2, 2, 30, 8]
    assert box["xyxy"] == [0, 2, 20, 8]
    assert box["yolo"] == [0.5, 0.5, 1.0, 0.6]
    with pytest.raises(ValueError, match="Invalid detection"):
        drafts.draft_box(Det((30, 0, 40, 5), 0.8), 20, 10)


def test_export_preserves_bytes_and_marks_every_draft_unreviewed(
    tmp_path: Path,
) -> None:
    """Members with one name in two folders keep separate files and hashes."""
    content = png()
    source = archive(
        tmp_path,
        {"a/tile.png": content, "b/tile.png": content, "notes.txt": b"ignored"},
    )
    output = tmp_path / "draft"
    model = FixedDetector([Det((1, 1, 8, 8), 0.9)])
    summary = drafts.export_drafts(source, tmp_path / "w.pt", output, model=model)
    assert summary["complete"]
    assert summary["archive_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert summary["recognition_id"] == "detector:fixture"
    assert (summary["images"], summary["detections"]) == (2, 2)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").open()]
    assert len({row["image"] for row in rows}) == 2
    for row in rows:
        assert row["reviewed"] is False
        assert row["status"] == "unreviewed_predictions"
        assert row["sha256"] == hashlib.sha256(content).hexdigest()
        assert (output / row["image"]).read_bytes() == content
        assert (output / row["draft_labels"]).read_text().startswith("0 ")
    assert not (output / "data.yaml").exists()
    with pytest.raises(FileExistsError):
        drafts.export_drafts(source, tmp_path / "w.pt", output, model=model)


def test_failed_image_leaves_an_incomplete_export(tmp_path: Path) -> None:
    """An undecodable image fails the run but keeps the drafts written before it."""
    source = archive(tmp_path, {"good.png": png(), "bad.png": b"not an image"})
    output = tmp_path / "draft"
    with pytest.raises(ValueError, match="Cannot decode"):
        drafts.export_drafts(source, tmp_path / "w.pt", output, model=FixedDetector())
    summary = json.loads((output / "summary.json").read_text())
    assert not summary["complete"]
    assert summary["images_without_detections"] == 1
    rows = [json.loads(line) for line in (output / "manifest.jsonl").open()]
    assert [row["status"] for row in rows] == ["unreviewed_zero_detections", "error"]
    assert (output / rows[0]["draft_labels"]).read_text() == ""


def test_production_detector_uses_checkpoint_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a substitute, the deployed detector loads the given weights."""
    loaded = []

    def load(weights: Path, **kwargs: object) -> FixedDetector:
        loaded.append((weights, kwargs))
        return FixedDetector()

    monkeypatch.setattr(detector, "Detector", load)
    weights = tmp_path / "weights.pt"
    summary = drafts.export_drafts(
        archive(tmp_path, {"photo.png": png()}),
        weights,
        tmp_path / "draft",
        device="cpu",
    )
    assert loaded == [(weights, {"device": "cpu"})]
    assert summary["confidence"] == 0.15
