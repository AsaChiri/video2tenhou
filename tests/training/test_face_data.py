# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Experimental face data preserves human holdouts and pseudo-label provenance."""

import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from video2tenhou.train import face_data

if TYPE_CHECKING:
    from pathlib import Path


def inputs(
    tmp_path: "Path",
    monkeypatch: "pytest.MonkeyPatch",
    *,
    duplicate: bool = False,
    unknown: bool = False,
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
    names = {0: "face"}
    (drafts / "summary.json").write_text(
        json.dumps(
            {
                "complete": True,
                "class_mode": "face",
                "classes": names,
                "archive_sha256": "archivehash",
            }
        )
    )
    (drafts / "photo.png").write_bytes(b"unreviewed photo")
    (drafts / "boxes.json").write_text(
        json.dumps(
            {"boxes": [{"class_id": 1 if unknown else 0, "yolo": [0.5, 0.5, 0.2, 0.4]}]}
        )
    )
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
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify human holdout and unique pseudo train are separate."""
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
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify cross split duplicate fails instead of contaminating validation."""
    human, drafts, hands = inputs(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(ValueError, match="duplicated across"):
        face_data.build_face_dataset(
            human,
            drafts,
            tmp_path / "prepared",
            video=tmp_path / "broadcast.mp4",
            hands=hands,
        )


def test_unknown_image_never_becomes_an_empty_face_training_negative(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify unknown image never becomes an empty face training negative."""
    human, drafts, hands = inputs(tmp_path, monkeypatch, unknown=True)
    output = tmp_path / "prepared"
    result = face_data.build_face_dataset(
        human, drafts, output, video=tmp_path / "broadcast.mp4", hands=hands
    )
    assert result["stats"]["pseudo_empty_or_unknown_images_excluded"] == 2
    assert not any(
        json.loads(line)["origin"] == "pseudo"
        for line in (output / "manifest.jsonl").read_text().splitlines()
    )


def test_human_only_refinement_keeps_identical_holdout_and_no_pseudo_rows(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify human only refinement keeps identical holdout and no pseudo rows."""
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
