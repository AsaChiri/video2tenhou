import json

import pytest

from video2tenhou.train import data
from video2tenhou.train.data import hand_of


def test_training_assigns_labels_by_physical_table_windows():
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
    tmp_path, monkeypatch
):
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
