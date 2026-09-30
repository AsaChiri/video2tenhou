# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted dense partial views reach reconstruction without requiring a pinned draw."""

from pathlib import Path

import numpy as np
import pytest

from tests.engine.factories import hand_decoder
from tests.recognition import models_stub
from tests.spies import record_results
from video2tenhou.engine import decode, dense
from video2tenhou.engine.solver import SeatTurn, Solution
from video2tenhou.engine.turns import Turn
from video2tenhou.train.data import CLASS_INDEX, CLASSES


def _frame(t: "float", count: "int") -> "dict":
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
    boxes = []
    for index, tile in enumerate(tiles[:count]):
        posterior = np.full(len(CLASSES), 0.001)
        posterior[CLASS_INDEX[tile]] = 0.95
        boxes.append(
            {
                "role": "tile",
                "xyxy": [index * 40, 0, index * 40 + 38, 58],
                "p": (posterior / posterior.sum()).tolist(),
                "conf": 0.95,
            }
        )
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
        ("not_selected", 1, 0),
        ("offline", 1, None),
        ("kan", 2, 1),
        ("kan_after", 2, 1),
        ("kan_empty", 1, 0),
    ],
)
def test_real_dense_evidence_changes_objective_before_resolve(
    monkeypatch: "pytest.MonkeyPatch",
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
    monkeypatch.setattr(
        decode,
        "seat_turns_of",
        lambda _turns, seat, _dealer: (
            seat_turns if seat == "S" else [],
            {1: 1} if case.startswith("kan") and seat == "S" else {},
        ),
    )
    first = Solution(
        "optimal",
        0.0,
        {},
        {},
        {},
        margins={("S", 1): 0.0},
        alternative_gaps={("S", 1): 10.0 if case == "not_selected" else 0.0},
    )
    if case.startswith("kan"):
        first.margins[("S", 1)] = first.alternative_gaps[("S", 1)] = 10.0
    second = Solution("optimal", 1.0, {}, {}, {})
    decoder = hand_decoder(
        dealer="E",
        dora=[],
        ura=[],
        tsumo_winner=None,
        calls=[],
        entry={"corner_wind": {"TL": "S", "TR": "W", "BL": "E", "BR": "N"}},
        obs={},
        t0=0.0,
        t1=65.0,
        problems=[],
        work_dir=tmp_path,
        models=None if case == "offline" else models_stub(),
        turns=[
            Turn(0, "S", "draw", None, 20.0),
            Turn(1, "W", "draw", None, 30.0),
            Turn(2, "S", "draw", None, 50.0),
            Turn(3, "W", "draw", None, 60.0),
        ],
        riichi_alternatives=[],
        _apply_hand_facts=lambda _model: None,
        _result_constraint=lambda _model: None,
    )
    searched = []

    def search(prior: "Solution | None" = None) -> "Solution":
        # Confirm the accepted observations really enter the CP objective;
        # do not rely on a list-length assertion alone to establish usefulness.
        program = decoder.model.build()[0]
        searched.append({"prior": prior, "terms": len(program.proto.objective.vars)})
        return first if len(searched) == 1 else second

    monkeypatch.setattr(decoder, "_solve", search)
    real_draws = dense.draws
    pin_counts = []

    monkeypatch.setattr(dense, "draws", record_results(real_draws, pin_counts))
    decode.HandDecoder._fit_evidence(decoder)
    assert len(searched) == expected_solves
    assert pin_counts == ([] if pinned is None else [pinned])
    assert decoder.sol is (second if expected_solves == 2 else first)
    if expected_solves == 2:
        assert searched[1]["prior"] is first
        assert searched[1]["terms"] > searched[0]["terms"]
        if case == "subset":
            assert decoder.model.hand_ev
            assert all(e.subset for e in decoder.model.hand_ev)
        elif case == "hidden":
            assert all(e.hidden == 2 for e in decoder.model.hand_ev)
        elif case == "draw":
            assert decoder.model.draw_ev
    else:
        assert not decoder.model.hand_ev
        assert not decoder.model.draw_ev
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
