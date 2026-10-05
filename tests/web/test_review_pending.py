# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""A hand is pending until its decode and the exports reflect its current inputs.

Pending means: no decode, a decode made from other facts (dismissals excluded),
hand metadata or site result, or exports not built from the current decode file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.web.analysis import publish, write_analysis
from video2tenhou import layout
from video2tenhou.tool import review_state
from video2tenhou.tool.processes import ProcessOwner


@pytest.fixture
def review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[review_state.ReviewState]:
    """Open a recording whose three hands (two games) are decoded and exported."""
    monkeypatch.setattr(review_state, "ROOT", tmp_path)
    monkeypatch.setattr(layout, "LABEL_DIR", tmp_path / "labels")
    write_analysis(tmp_path, "recording", [101, 102], hands_per_game=2)
    owner = ProcessOwner()
    yield review_state.ReviewState(
        tmp_path / "recording.mp4",
        tmp_path / "work",
        "pml",
        tmp_path / "out",
        owner,
    )
    owner.shutdown()


def root(state: review_state.ReviewState) -> Path:
    """Return the data directory that holds this recording's folders."""
    return state.work.parent.parent


def test_answers_and_deletions_are_pending_until_decoded_and_exported(
    review: review_state.ReviewState,
) -> None:
    """Adding or deleting an answer needs a new decode and new exports."""
    assert review.pending() == []
    fact = review.add_fact({"hand": 1, "kind": "draw", "seat": "E", "tile": "2p"})
    assert review.pending() == [1]
    publish(root(review), "recording", [1])
    assert review.pending() == []
    assert review.delete_fact(fact["ts"]) == 1
    assert review.pending() == [1]
    publish(root(review), "recording", [1])
    assert review.pending() == []


def test_a_decode_without_matching_exports_is_pending(
    review: review_state.ReviewState,
) -> None:
    """Exports must have been built from the current decode file."""
    decode = review.decode_path(2)
    decode.write_text(decode.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert review.pending() == [2]
    (review.out / "export-inputs.json").unlink()
    assert review.pending() == [0, 1, 2, 3]
    publish(root(review), "recording", [])
    assert review.pending() == []


def test_a_decode_by_another_decoder_version_is_stale_and_never_shown(
    review: review_state.ReviewState,
) -> None:
    """An older decode format is pending and is neither rendered nor queried."""
    decode = review.decode_path(1)
    data = json.loads(decode.read_text(encoding="utf-8"))
    data.update(
        decoder_version=data["decoder_version"] - 1,
        items=[{"kind": "draw", "seat": "E", "j": 0, "text": "Old format"}],
        score={"yaku": ["Riichi (1)"]},
    )
    decode.write_text(json.dumps(data), encoding="utf-8")
    publish(root(review), "recording", [])
    assert review.pending() == [1]
    assert review.all_items() == []
    assert review.hand_view(1)["decode"] is None
    row = review.hand_summary()[1]
    assert (row["pending"], row["status"], row["score"]) == (True, None, None)
    publish(root(review), "recording", [1])
    assert review.pending() == []


def test_updating_one_hand_leaves_another_answer_pending(
    review: review_state.ReviewState,
) -> None:
    """A rebuild of hand 0 does not acknowledge an answer for hand 1."""
    for hand in (0, 1):
        review.add_fact({"hand": hand, "kind": "draw", "seat": "E", "tile": "2p"})
    publish(root(review), "recording", [0])
    assert review.pending() == [1]


def test_dismissing_a_question_never_requires_an_update(
    review: review_state.ReviewState,
) -> None:
    """A dismissal is review state, not a reconstruction input."""
    decode = review.decode_path(3)
    data = json.loads(decode.read_text(encoding="utf-8"))
    data["items"] = [{"kind": "conflict", "t": 120.4, "text": "Leave or fix."}]
    decode.write_text(json.dumps(data), encoding="utf-8")
    publish(root(review), "recording", [])
    assert [item["id"] for item in review.all_items()] == ["conflict::120"]
    review.add_fact({"hand": 3, "kind": "dismiss", "item": "conflict::120"})
    assert review.all_items() == []
    assert review.pending() == []


def test_unambiguous_legacy_notes_dismiss_and_others_are_ignored(
    review: review_state.ReviewState,
) -> None:
    """Saved notes hide only the one question with their text; the journal stays."""
    decode = review.decode_path(0)
    data = json.loads(decode.read_text(encoding="utf-8"))
    data["items"] = [
        {"kind": "order", "seat": "E", "t": 30, "text": "Which tile?"},
        {"kind": "call", "seat": "S", "t": 40, "text": "Same"},
        {"kind": "call", "seat": "W", "t": 50, "text": "Same"},
    ]
    decode.write_text(json.dumps(data), encoding="utf-8")
    journal = review.labels / "facts.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    notes = [
        {"game": 0, "kyoku": 0, "honba": 0, "kind": "note", "text": text}
        for text in ("Which tile?", "Same")
    ]
    journal.write_text("".join(json.dumps(n) + "\n" for n in notes))
    before = journal.read_bytes()
    assert [item["id"] for item in review.all_items()] == ["call:S:40", "call:W:50"]
    assert journal.read_bytes() == before


def test_new_kinds_are_validated_and_notes_are_not_accepted(
    review: review_state.ReviewState,
) -> None:
    """The studio saves answers and dismissals only."""
    with pytest.raises(ValueError, match="Unsupported answer"):
        review.add_fact({"hand": 0, "kind": "note", "text": "OK"})
    with pytest.raises(ValueError, match="question to dismiss"):
        review.add_fact({"hand": 0, "kind": "dismiss"})


def test_changed_inputs_hide_hands_until_analysis(
    review: review_state.ReviewState,
) -> None:
    """Changed settings retire the analysis without touching saved answers."""
    review.add_fact({"hand": 0, "kind": "draw", "seat": "E", "tile": "2p"})
    (review.work / "inputs.changed").write_text("Analyze again.\n")
    assert review.hands == []
    assert review.pending() == []
    assert len(review.all_facts()) == 1


def test_summary_marks_pending_hands_and_reads_each_decode_once(
    review: review_state.ReviewState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hand rows carry the pending flag; unchanged files are not parsed again."""
    review.add_fact({"hand": 2, "kind": "draw", "seat": "E", "tile": "2p"})
    rows = review.hand_summary()
    assert [row["pending"] for row in rows] == [False, False, True, False]
    assert set(rows[0]) == {
        "hand",
        "game",
        "kyoku",
        "honba",
        "t_start",
        "t_end",
        "status",
        "pending",
        "turns",
        "score",
    }
    reads = []
    original = Path.read_text

    def read_text(path: Path, encoding: str | None = None) -> str:
        if path.parent == review.work / "decode":
            reads.append(path.name)
        return original(path, encoding)

    monkeypatch.setattr(Path, "read_text", read_text)
    review.hand_summary()
    review.hand_summary()
    os.utime(review.decode_path(0), ns=(1, 1))
    review.hand_summary()
    assert reads == ["00.json"]
