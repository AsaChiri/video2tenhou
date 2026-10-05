# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted dense partial views reach reconstruction without requiring a pinned draw."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.builders import posterior
from tests.engine import factories
from tests.recognition import models_stub
from video2tenhou.engine import dense, rules
from video2tenhou.engine.confidence import Certificate
from video2tenhou.engine.dense import DenseContext
from video2tenhou.engine.questions import Report
from video2tenhou.engine.reconstruct import fit_evidence
from video2tenhou.engine.solver import HandModel, SeatTurn, Solution
from video2tenhou.engine.turns import Turn


def _frame(t: float, count: int) -> dict:
    tiles = [
        "1m",
        "2m",
        "3m",
        "4p",
        "5p",
        "6p",
        "7s",
        "8s",
        "9s",
        "1z",
        "1z",
        "3z",
        "9m",
        "2z",
    ]
    boxes = [
        {
            "role": "tile",
            "xyxy": [index * 40, 0, index * 40 + 38, 58],
            "p": posterior(tile, 0.95, floor=0.001).tolist(),
            "conf": 0.95,
        }
        for index, tile in enumerate(tiles[:count])
    ]
    return {"t": t, "boxes": boxes}


@pytest.mark.parametrize(
    ("case", "expected_solves", "pinned"),
    [
        ("subset", 2, 0),
        ("hidden", 2, 0),
        ("full", 2, 1),
        ("draw", 2, 1),
        ("empty", 1, 0),
        ("moving", 1, 0),
        ("not_selected", 1, None),  # nothing to reread: no note
        ("offline", 1, None),
        ("kan", 2, 1),
        ("kan_after", 2, 1),
        ("kan_empty", 1, 0),
    ],
)
def test_real_dense_evidence_changes_objective_before_resolve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    case: str,
    expected_solves: int,
    pinned: int | None,
) -> None:
    """Exercise still-run aggregation and evidence mapping; stub only I/O and search."""
    frames = _dense_frames(case)
    reads = []

    def read_window(
        _context: object,
        lo: float,
        hi: float,
        regions: list[str],
        **_unused_kwargs: object,
    ) -> dict:
        reads.append((lo, hi, regions))
        return {"hand:TL": [frame for frame in frames if lo <= frame["t"] <= hi]}

    monkeypatch.setattr(dense, "dense_reads", read_window)
    seat_turns = [SeatTurn(0, "draw", "6z", 10, 20), SeatTurn(1, "draw", "9m", 40, 50)]
    if case.startswith("kan"):
        seat_turns[1].kind, seat_turns[1].kan = "kan", "daiminkan"
        seat_turns[1].removed = ["1m"] * 3
    model = HandModel("E", {s: seat_turns if s == "S" else [] for s in rules.SEATS}, [])
    melds_before = {
        s: {1: 1} if case.startswith("kan") and s == "S" else {} for s in rules.SEATS
    }
    # An ambiguous draw selects the window; a certified open-kan draw still does.
    certificate = (
        Certificate(10.0)
        if case.startswith("kan")
        else Certificate(0.0, 10.0 if case == "not_selected" else 0.0)
    )
    first = Solution(
        "optimal",
        0.0,
        {},
        {},
        {},
        certificates={("draw", "S", 1): certificate},
        certified=True,
    )
    second = Solution("optimal", 1.0, {}, {}, {})
    turns = [
        Turn(0, "S", "draw", None, 20.0),
        Turn(1, "W", "draw", None, 30.0),
        Turn(2, "S", "draw", None, 50.0),
        Turn(3, "W", "draw", None, 60.0),
    ]
    searcher = factories.search(model, turns)
    searched = []

    def search(prior: Solution | None = None) -> Solution:
        # Confirm the accepted observations really enter the CP objective;
        # do not rely on a list-length assertion alone to establish usefulness.
        program = model.build().model
        searched.append({"prior": prior, "terms": len(program.proto.objective.vars)})
        return first if len(searched) == 1 else second

    monkeypatch.setattr(searcher, "solve", search)
    report = Report()
    sol, _ = fit_evidence(
        searcher,
        factories.hand(t1=65.0),
        factories.sequence(turns),
        factories.riichi(),
        melds_before=melds_before,
        context=DenseContext(
            entry={"corner_wind": {"TL": "S", "TR": "W", "BL": "E", "BR": "N"}},
            models=None if case == "offline" else models_stub(),
            work_dir=tmp_path,
            t0=0.0,
            t1=65.0,
            diagnostics=report.diagnostics,
        ),
        report=report,
    )
    assert len(searched) == expected_solves
    # the diagnosis says how many of the reread draws a still view pins
    assert report.diagnostics == (
        []
        if pinned is None
        else [
            f"dense hand reads for 1 uncertain draws: still views pin {pinned} of them"
        ]
    )
    assert report.items == report.notes == []
    assert sol is (second if expected_solves == 2 else first)
    if expected_solves == 2:
        assert searched[1]["prior"] is first
        assert searched[1]["terms"] > searched[0]["terms"]
        if case == "subset":
            assert model.hand_ev
            assert all(e.subset for e in model.hand_ev)
        elif case == "hidden":
            assert all(e.hidden == 2 for e in model.hand_ev)
        elif case == "draw":
            assert model.draw_ev
    else:
        assert not model.hand_ev
        assert not model.draw_ev
    if case in ("not_selected", "offline"):
        assert reads == []


def _dense_frames(case: str) -> list[dict]:
    """Provide the acquisition sequence for each evidence scenario."""
    counts = {
        "subset": 8,
        "hidden": 11,
        "full": 13,
        "draw": 14,
        "kan": 13,
        "kan_after": 10,
    }
    start = 52 if case == "kan_after" else 42 if case == "draw" else 22
    frames = [_frame(start + 0.2 * k, counts.get(case, 8)) for k in range(6)]
    if case in ("empty", "kan_empty"):
        frames = []
    elif case == "moving":
        frames = [_frame(22 + 0.2 * k, 8 + k % 3) for k in range(6)]
    return frames
