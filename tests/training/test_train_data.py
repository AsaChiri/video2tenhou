import json

import pytest

from video2tenhou.train import data
from video2tenhou.train.data import hand_of


def test_hand_of_goes_by_the_overlay_segment():
    # read windows start 60 s before the overlay switch and overlap; the overlay segments do not
    hands = [
        {"hand": 0, "game": 0, "t_start": 40.0, "t_end": 400.0, "t_overlay": [100.0, 400.0]},
        {"hand": 1, "game": 0, "t_start": 340.0, "t_end": 700.0, "t_overlay": [402.0, 700.0]},
        {"hand": 2, "game": 1, "t_start": 900.0, "t_end": 1200.0, "t_overlay": [960.0, 1200.0]},
    ]
    assert hand_of(350.0, hands) == 0            # inside the overlap: still hand 0's segment
    assert hand_of(401.0, hands) == 1            # between the segments (the overlay switching): the table is on hand 1
    assert hand_of(500.0, hands) == 1
    assert hand_of(800.0, hands) is None         # the break between hanchan belongs to no hand
    assert hand_of(950.0, hands) is None         # nor the window before the first segment of a hanchan
    assert hand_of(50.0, hands) is None


def test_training_requires_overlay_segments_instead_of_overlapping_read_windows():
    hands = [{"game": 0, "t_start": 380.0, "t_end": 760.0}, {"game": 0, "t_start": 765.0, "t_end": 1060.0}]
    with pytest.raises(KeyError, match="t_overlay"):
        hand_of(400.0, hands)


def test_training_requires_pipeline_hand_table_even_if_old_label_table_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "ROOT", tmp_path)
    labels = tmp_path / "labels" / "recording"
    labels.mkdir(parents=True)
    (labels / "legacy_hands.json").write_text("[]")
    work = tmp_path / "work"
    with pytest.raises(FileNotFoundError):
        data.hand_table("recording.mp4", work)
    current = work / "recording" / "hands.json"
    current.parent.mkdir(parents=True)
    hands = [{"hand": 0, "game": 0, "t_overlay": [100., 400.]}]
    current.write_text(json.dumps(hands))
    assert data.hand_table("recording.mp4", work) == hands
