"""Experimental face data preserves human holdouts and pseudo-label provenance."""
import hashlib
import json

import pytest

from video2tenhou.train import face_data


def inputs(tmp_path, monkeypatch, *, duplicate=False, unknown=False):
    human, drafts = tmp_path / "human", tmp_path / "drafts"
    for split, stamp, content in (("train", 1, b"training"), ("val", 41, b"training" if duplicate else b"validation")):
        (human / "images" / split).mkdir(parents=True)
        (human / "labels" / split).mkdir(parents=True)
        (human / "images" / split / f"hand_TL_{stamp}.jpg").write_bytes(content)
        (human / "labels" / split / f"hand_TL_{stamp}.txt").write_text("0 .5 .5 .2 .4\n")
    monkeypatch.setattr(face_data, "load_labels", lambda _: [dict(kind="hand", corner="TL", t=1.), dict(kind="hand", corner="TL", t=41.)])
    drafts.mkdir()
    names = {0: "face"}
    (drafts / "summary.json").write_text(json.dumps(dict(complete=True, class_mode="face", classes=names, archive_sha256="archivehash")))
    (drafts / "photo.png").write_bytes(b"unreviewed photo")
    (drafts / "boxes.json").write_text(json.dumps(dict(boxes=[dict(class_id=1 if unknown else 0, yolo=[.5, .5, .2, .4])])))
    row = dict(status="unreviewed_predictions", image="photo.png", predictions="boxes.json", sha256=hashlib.sha256(b"unreviewed photo").hexdigest())
    (drafts / "manifest.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    hands = [dict(t_overlay=[index * 10., index * 10. + 9.]) for index in range(5)]
    return human, drafts, hands


def test_human_holdout_and_unique_pseudo_train_are_separate(tmp_path, monkeypatch):
    human, drafts, hands = inputs(tmp_path, monkeypatch)
    output = tmp_path / "prepared"
    report = face_data.build_face_dataset(human, drafts, output, video=tmp_path / "broadcast.mp4", hands=hands)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert [(row["origin"], row["reviewed"]) for row in rows if row["split"] == "val"] == [("human", True)]
    pseudo = next(row for row in rows if row["origin"] == "pseudo")
    assert pseudo["split"] == "train" and pseudo["reviewed"] is False
    assert pseudo["group"] == "archive:archivehash"
    assert report["stats"]["pseudo_exact_duplicates_skipped"] == 1


def test_cross_split_duplicate_fails_instead_of_contaminating_validation(tmp_path, monkeypatch):
    human, drafts, hands = inputs(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(ValueError, match="duplicated across"):
        face_data.build_face_dataset(human, drafts, tmp_path / "prepared", video=tmp_path / "broadcast.mp4", hands=hands)


def test_unknown_image_never_becomes_an_empty_face_training_negative(tmp_path, monkeypatch):
    human, drafts, hands = inputs(tmp_path, monkeypatch, unknown=True)
    output = tmp_path / "prepared"
    result = face_data.build_face_dataset(human, drafts, output, video=tmp_path / "broadcast.mp4", hands=hands)
    assert result["stats"]["pseudo_empty_or_unknown_images_excluded"] == 2
    assert not any(json.loads(line)["origin"] == "pseudo" for line in (output / "manifest.jsonl").read_text().splitlines())


def test_human_only_refinement_keeps_identical_holdout_and_no_pseudo_rows(tmp_path, monkeypatch):
    human, _, hands = inputs(tmp_path, monkeypatch)
    output = tmp_path / "human-only"
    result = face_data.build_face_dataset(human, None, output, video=tmp_path / "broadcast.mp4", hands=hands)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(row["origin"] == "human" and row["reviewed"] for row in rows)
    assert [(row["split"], row["group"]) for row in rows] == [("val", "broadcast:hand:4"), ("train", "broadcast:hand:0")]
    assert result["archive_teacher"] is None
