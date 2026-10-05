# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Training crop geometry and human annotation handling."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from video2tenhou.layout import Calibration
from video2tenhou.train import data
from video2tenhou.train.data import hand_of


def test_training_assigns_labels_by_physical_table_windows() -> None:
    hands = [
        {"game": 0, "t_start": 40, "t_end": 300},
        {"game": 0, "t_start": 300.5, "t_end": 600},
    ]
    assert hand_of(39, hands) is None
    assert hand_of(300, hands) == 0
    assert hand_of(300.25, hands) is None
    assert hand_of(300.5, hands) == 1
    assert hand_of(601, hands) is None


def test_training_requires_pipeline_hand_table_even_if_old_label_table_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(data, "ROOT", tmp_path)
    labels = tmp_path / "labels" / "recording"
    labels.mkdir(parents=True)
    (labels / "legacy_hands.json").write_text("[]")
    work = tmp_path / "work"
    with pytest.raises(FileNotFoundError):
        data.hand_table("recording.mp4", work)
    current = work / "recording" / "hands.json"
    current.parent.mkdir(parents=True)
    hands = [{"hand": 0, "game": 0, "t_start": 100.0, "t_end": 400.0}]
    current.write_text(json.dumps(hands))
    assert data.hand_table("recording.mp4", work) == hands


def test_build_keeps_reviewed_tiles_and_empty_melds_in_whole_hand_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise annotation loading through detector labels and classifier images."""
    monkeypatch.setattr(data, "ROOT", tmp_path)
    labels = tmp_path / "labels/recording/boxes"
    labels.mkdir(parents=True)
    tile = {
        "tile": "2p",
        "quad": [[20, 20], [80, 20], [80, 60], [20, 60]],
        "sideways": True,
    }
    back = {"tile": "X", "quad": [[100, 100], [160, 100], [160, 140], [100, 140]]}
    for stamp, boxes in ((5, [tile]), (15, [tile, back]), (45, [])):
        (labels / f"{stamp}.json").write_text(
            json.dumps(
                {
                    "t": stamp,
                    "kind": "meld",
                    "corner": "TL",
                    "boxes": boxes,
                    "melds": [],
                }
            )
        )
    work = tmp_path / "work"
    (work / "recording").mkdir(parents=True)
    (work / "recording/hands.json").write_text(
        json.dumps([{"t_start": i * 10, "t_end": i * 10 + 9} for i in range(5)])
    )
    frame = np.full((240, 320, 3), 150, dtype=np.uint8)
    monkeypatch.setattr(data.video, "frame_at", lambda *_args: frame.copy())
    monkeypatch.setattr(
        data, "region_upright", lambda image, *_args: (image, np.eye(3))
    )
    out = tmp_path / "dataset"
    stats = data.build("recording.mp4", Calibration.load("pml"), work, out)
    assert stats["det_train_boxes"] == 1
    assert stats["det_val_negatives"] == 1
    # The face detector never learns a reviewed face-down tile as background.
    assert stats["det_train_back_images_excluded"] == 1
    assert not (out / "detector/labels/train/meld_TL_15.txt").exists()
    assert len(list((out / "classifier/train/X").glob("*.png"))) == 1
    assert (out / "detector/data.yaml").read_text().endswith("names:\n  0: face\n")
    assert (out / "detector/labels/train/meld_TL_5.txt").read_text().split() == [
        "0",
        "0.15625",
        "0.16667",
        "0.18750",
        "0.16667",
    ]
    assert (out / "detector/labels/val/meld_TL_45.txt").read_text() == ""
    crops = list((out / "classifier/train/2p").glob("meld_TL_5_*.png"))
    assert len(crops) == 2
    for path in crops:
        image = cv2.imread(str(path))
        assert image is not None
        assert image.shape == (96, 64, 3)
    assert not list((out / "classifier/val/2p").glob("*.png"))
    assert len(list((out / "classifier/val/none").glob("*.png"))) == 3
    assert json.loads((out / "meta.json").read_text())["stats"] == stats
