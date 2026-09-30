# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Table timing supplies timestamps; scoremj supplies all hand metadata."""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from tests.recognition import RecognitionStub
from video2tenhou import timeline as timing
from video2tenhou.layout import Calibration
from video2tenhou.perception import detector
from video2tenhou.perception.detector import Det
from video2tenhou.record import SEATS, Game, HandResult

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def game() -> "Game":
    """Create an authoritative synthetic game with starting-seat metadata."""
    return Game(
        123,
        dict(zip(SEATS, ("east", "south", "west", "north"), strict=True)),
        {},
        [
            HandResult(
                0, 0, 0, dict(zip(SEATS, (1000, -1000, 0, 0), strict=True)), "ron"
            ),
            HandResult(1, 0, 0, dict.fromkeys(SEATS, 0), "draw"),
        ],
    )


def observations() -> "dict[str, list[dict]]":
    # Static old ponds at the opening, then two hands and an empty tail.
    """Create timed pond counts covering an opening, two hands and a tail."""
    return {
        corner: [
            {"t0": t, "t1": t + 1, "count": count, "partial": False, "n_used": 1}
            for t, count in [
                (0, 12),
                (20, 12),
                (40, 0),
                (50, 1),
                (60, 4),
                (80, 8),
                (100, 0),
                (110, 1),
                (120, 4),
                (140, 8),
                (160, 0),
                (180, 0),
            ]
        ]
        for corner in ("TL", "TR", "BL", "BR")
    }


def test_static_opening_and_empty_tail_do_not_become_hands() -> None:
    """Verify static opening and empty tail do not become hands."""
    assert timing.hand_windows(observations(), 200) == [(21.5, 81), (81.5, 141)]


def test_counts_exclude_indicators_backs_partial_views_and_invalid_rows(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Verify counts exclude indicators backs partial views and invalid rows."""
    image = np.zeros((200, 220, 3), np.uint8)
    ivs = [
        timing.calm.Interval("pond:TL", 0, 2, 5, calm=True, motion=0, skin=0),
        timing.calm.Interval(
            "pond:TR", 0, 2, 5, calm=True, motion=0, skin=0, partial=True
        ),
        timing.calm.Interval("hand:TL", 0, 2, 5, calm=True, motion=0, skin=0),
        timing.calm.Interval("pond:TL", 3, 5, 5, calm=True, motion=0, skin=0),
    ]

    def detection(
        *, x: "int", y: int = 30, confidence: float = 0.9, back: bool = False
    ) -> "Det":
        return Det(xyxy=(x, y, x + 20, y + 30), conf=confidence, back=back)

    batches = iter(
        [
            [
                [
                    detection(x=20),
                    detection(x=42),
                    detection(x=20, y=170),
                    detection(x=70, back=True),
                    detection(x=95, confidence=0.2),
                ]
            ],
            [[detection(x=20 + i * 22) for i in range(7)]],
        ]
    )
    det = RecognitionStub("timing")
    monkeypatch.setattr(det, "predict_batch", lambda _images: next(batches))
    cal = Calibration.load("pml")
    monkeypatch.setattr(cal, "region", lambda _frame, _name: (image, None))
    monkeypatch.setattr(
        timing.video,
        "sample",
        lambda *_unused_a, **_unused_kw: [(1, image), (4, image)],
    )
    counts = timing.read_pond_counts("source.mp4", cal, ivs, det)
    assert counts["TL"][0]["count"] == 2
    assert not counts["TL"][0]["partial"]
    assert counts["TL"][1]["partial"]
    assert counts["TR"] == []


def test_recording_starting_with_active_play_keeps_first_hand() -> None:
    """Verify recording starting with active play keeps first hand."""
    rows = {
        c: [
            {"t0": t, "t1": t + 1, "count": n, "n_used": 1}
            for t, n in [(0, 0), (20, 1), (40, 4), (60, 7)]
        ]
        for c in timing.CORNERS
    }
    assert timing.hand_windows(rows, 90) == [(0, 90)]


def test_site_metadata_uses_fixed_chairs_and_rotates_winds_without_ocr() -> None:
    """Verify site metadata uses fixed chairs and rotates winds without ocr."""
    entries, problems = timing.site_entries([(0, 100), (100.5, 200)], [game()])
    assert not problems
    assert entries[0]["corner_wind"] == {"TL": "E", "BL": "S", "BR": "W", "TR": "N"}
    assert entries[1]["corner_wind"] == {"TL": "N", "BL": "E", "BR": "S", "TR": "W"}
    assert entries[1]["scores"] == {"N": 26000, "E": 24000, "S": 25000, "W": 25000}
    assert entries[1]["nicks"]["TL"] == "east"
    assert entries[1]["corner_site"] == entries[0]["corner_site"]


def test_repeats_keep_the_dealer_and_new_games_reset_starting_seats() -> None:
    """Verify repeats keep the dealer and new games reset starting seats."""
    first = game()
    first.hands[1].kyoku = 0
    first.hands[1].honba = 1
    entries, problems = timing.site_entries(
        [(0, 100), (101, 200), (201, 300), (301, 400)], [first, game()]
    )
    assert not problems
    assert (
        entries[0]["corner_wind"]
        == entries[1]["corner_wind"]
        == entries[2]["corner_wind"]
    )
    assert entries[2]["scores"] == dict.fromkeys("ESWN", 25000)


def test_mismatched_hand_count_does_not_invent_alignment() -> None:
    """Verify mismatched hand count does not invent alignment."""
    entries, problems = timing.site_entries([(0, 100)], [game()])
    assert entries == []
    assert "1 hands" in problems[0]
    assert "2" in problems[0]


def test_production_header_needs_no_overlay_and_invalidates_table_cache(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify production header needs no overlay and invalidates table cache."""
    source = tmp_path / "video.mp4"
    source.write_bytes(b"source")
    data = dict(Calibration.load("pml").data)
    cal = Calibration(data)
    assert "overlay" not in data
    assert cal.region(np.zeros((1080, 1920, 3), np.uint8), "pond:TL")[0].size
    monkeypatch.setattr(timing.calm, "run_calm", lambda *_unused_a, **_unused_kw: [])
    monkeypatch.setattr(
        timing.video, "probe", lambda *_unused_a: SimpleNamespace(duration=200)
    )
    monkeypatch.setattr(detector, "Detector", lambda: SimpleNamespace(id="detector-v1"))
    scans = []
    monkeypatch.setattr(
        timing,
        "read_pond_counts",
        lambda *_unused_a, **_unused_kw: scans.append(True) or observations(),
    )
    work = tmp_path / "work"
    for _ in range(2):
        entries, problems = timing.run_header(source, cal, [game()], work)
        assert len(entries) == 2
        assert not problems
    assert len(scans) == 1
    source.write_bytes(b"replacement")
    timing.run_header(source, cal, [game()], work)
    assert len(scans) == 2
    saved = json.loads((work / "table-timing.json").read_text())
    saved["observations"]["TL"][0]["count"] = 123
    (work / "table-timing.json").write_text(json.dumps(saved))
    timing.run_header(source, cal, [game()], work)
    assert len(scans) == 3


def test_failed_timing_preserves_existing_hand_metadata(
    tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify failed timing preserves existing hand metadata."""
    source = tmp_path / "video.mp4"
    source.write_bytes(b"source")
    previous = '[{"human":"retained"}]'
    (tmp_path / "hands.json").write_text(previous)
    monkeypatch.setattr(timing.calm, "run_calm", lambda *_unused_a, **_unused_kw: [])
    monkeypatch.setattr(
        timing.video, "probe", lambda *_unused_a: SimpleNamespace(duration=200)
    )
    monkeypatch.setattr(detector, "Detector", lambda: SimpleNamespace(id="test"))
    monkeypatch.setattr(
        timing,
        "read_pond_counts",
        lambda *_unused_a, **_unused_kw: {c: [] for c in timing.CORNERS},
    )
    entries, problems = timing.run_header(
        source, Calibration.load("pml"), [game()], tmp_path
    )
    assert not entries
    assert problems
    assert (tmp_path / "hands.json").read_text() == previous
