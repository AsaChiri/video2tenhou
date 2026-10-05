# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Experimental face data preserves human holdouts and pseudo-label provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from video2tenhou.train import face_data


def inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    duplicate: bool = False,
    empty: bool = False,
) -> tuple:
    """Create human and draft face annotations for dataset assembly tests."""
    human, drafts = tmp_path / "human", tmp_path / "drafts"
    for split, stamp, content in (
        ("train", 1, b"training"),
        ("val", 41, b"training" if duplicate else b"validation"),
    ):
        (human / "images" / split).mkdir(parents=True)
        (human / "labels" / split).mkdir(parents=True)
        (human / "images" / split / f"hand_TL_{stamp}.jpg").write_bytes(content)
        (human / "labels" / split / f"hand_TL_{stamp}.txt").write_text(
            "0 .5 .5 .2 .4\n"
        )
    monkeypatch.setattr(
        face_data,
        "load_labels",
        lambda _: [
            {"kind": "hand", "corner": "TL", "t": 1.0},
            {"kind": "hand", "corner": "TL", "t": 41.0},
        ],
    )
    drafts.mkdir()
    (drafts / "summary.json").write_text(
        json.dumps({"complete": True, "archive_sha256": "archivehash"})
    )
    (drafts / "photo.png").write_bytes(b"unreviewed photo")
    boxes = [] if empty else [{"class_id": 0, "yolo": [0.5, 0.5, 0.2, 0.4]}]
    (drafts / "boxes.json").write_text(json.dumps({"boxes": boxes}))
    row = {
        "status": "unreviewed_predictions",
        "image": "photo.png",
        "predictions": "boxes.json",
        "sha256": hashlib.sha256(b"unreviewed photo").hexdigest(),
    }
    (drafts / "manifest.jsonl").write_text(
        json.dumps(row) + "\n" + json.dumps(row) + "\n"
    )
    hands = [
        {"t_start": index * 10.0, "t_end": index * 10.0 + 9.0} for index in range(5)
    ]
    return human, drafts, hands


def test_human_holdout_and_unique_pseudo_train_are_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    human, drafts, hands = inputs(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    report = face_data.build_face_dataset(
        human, drafts, output, video=tmp_path / "broadcast.mp4", hands=hands
    )
    rows = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 3
    assert [
        (row["origin"], row["reviewed"]) for row in rows if row["split"] == "val"
    ] == [("human", True)]
    pseudo = next(row for row in rows if row["origin"] == "pseudo")
    assert pseudo["split"] == "train"
    assert pseudo["reviewed"] is False
    assert pseudo["group"] == "archive:archivehash"
    assert report["stats"]["pseudo_exact_duplicates_skipped"] == 1


def test_cross_split_duplicate_fails_instead_of_contaminating_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    human, drafts, hands = inputs(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(ValueError, match="duplicated across"):
        face_data.build_face_dataset(
            human,
            drafts,
            tmp_path / "prepared",
            video=tmp_path / "broadcast.mp4",
            hands=hands,
        )


def test_empty_draft_never_becomes_a_training_negative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero teacher detections do not show an empty scene."""
    human, drafts, hands = inputs(tmp_path, monkeypatch, empty=True)
    output = tmp_path / "prepared"
    result = face_data.build_face_dataset(
        human, drafts, output, video=tmp_path / "broadcast.mp4", hands=hands
    )
    assert result["stats"]["pseudo_empty_images_excluded"] == 2
    assert not any(
        json.loads(line)["origin"] == "pseudo"
        for line in (output / "manifest.jsonl").read_text().splitlines()
    )


def test_human_labels_must_be_face_boxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second detector class means the dataset predates face-only training."""
    human, _, hands = inputs(tmp_path, monkeypatch)
    (human / "labels/train/hand_TL_1.txt").write_text("1 .5 .5 .2 .4\n")
    with pytest.raises(ValueError, match="must be face boxes"):
        face_data.build_face_dataset(
            human, None, tmp_path / "out", video=tmp_path / "broadcast.mp4", hands=hands
        )


def test_human_only_refinement_keeps_identical_holdout_and_no_pseudo_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    human, _, hands = inputs(tmp_path, monkeypatch)
    output = tmp_path / "human-only"
    result = face_data.build_face_dataset(
        human, None, output, video=tmp_path / "broadcast.mp4", hands=hands
    )
    rows = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 2
    assert all(row["origin"] == "human" and row["reviewed"] for row in rows)
    assert [(row["split"], row["group"]) for row in rows] == [
        ("val", "broadcast:hand:4"),
        ("train", "broadcast:hand:0"),
    ]
    assert result["archive_teacher"] is None
