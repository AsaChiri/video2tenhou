# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Broken acquisition and dependencies must not become uncertainty or illegal hands."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from mahjong.shanten import Shanten

from tests.engine import factories
from tests.recognition import models_stub
from video2tenhou import tenhou6
from video2tenhou.engine import dense, pond_evidence, scoring
from video2tenhou.engine.dense import DenseContext
from video2tenhou.engine.events import acquire_replacements
from video2tenhou.engine.hand import site_seat, site_seat_name
from video2tenhou.engine.ponds import PondSlot
from video2tenhou.engine.questions import Report
from video2tenhou.engine.review import facts_for_hand
from video2tenhou.engine.solver import HandModel, SeatTurn
from video2tenhou.engine.turns import Skip, Turn
from video2tenhou.perception.tiles import CLASSES

ENTRY = {"corner_wind": {"TL": "E", "TR": "S", "BL": "W", "BR": "N"}}
CONCEALED = [f"{n}m" for n in range(1, 10)] + ["1p", "2p", "3p", "5p"]


SCORE_CONTEXT = scoring.WinContext(
    tsumo=False, riichi=True, seat="S", round_wind="E", dora=[], ura=[]
)


@pytest.mark.parametrize(
    "operation", ["taken", "skipped", "draw", "riichi", "replacement"]
)
@pytest.mark.parametrize("failure", [OSError, RuntimeError, KeyError])
def test_acquisition_failure_stops_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
    failure: type[Exception],
) -> None:
    def broken(*_unused_args: object, **_unused_kwargs: object) -> None:
        raise failure("acquisition failed")

    monkeypatch.setattr(dense, "dense_reads", broken)
    monkeypatch.setattr(dense, "dense_pond_reads", broken)
    monkeypatch.setattr(pond_evidence, "read_replacement", broken)
    slot = PondSlot(
        1, 0, 0, np.zeros(len(CLASSES)), 20, (10, 20), 20, xyxy=(0, 0, 10, 14)
    )
    turns = [Turn(0, "S", "draw", slot, 20.0)]
    models, report = models_stub(), Report()
    model = HandModel("E", {"S": [SeatTurn(0, "draw", "1m", 10, 20)]}, [])

    def acquire() -> None:
        if operation == "taken":
            dense.taken_discards(
                "S",
                10,
                20,
                {},
                context=DenseContext(entry=ENTRY, models=models, work_dir=tmp_path),
            )
        elif operation == "skipped":
            dense.skipped_turns(
                [Skip(0, "S", before=20.0)],
                {},
                context=DenseContext(
                    entry=ENTRY,
                    models=models,
                    work_dir=tmp_path,
                    t0=0,
                    t1=30,
                    diagnostics=report.diagnostics,
                ),
            )
        elif operation == "draw":
            dense.draws(
                [("S", 0)],
                model,
                turns,
                {},
                context=DenseContext(
                    entry=ENTRY,
                    models=models,
                    work_dir=tmp_path,
                    t0=0,
                    diagnostics=report.diagnostics,
                ),
            )
        elif operation == "riichi":
            dense.turned_tile(
                "S",
                turns,
                context=DenseContext(
                    entry=ENTRY,
                    models=models,
                    work_dir=tmp_path,
                    t0=0,
                    diagnostics=report.diagnostics,
                ),
            )
        else:
            monkeypatch.setattr(
                pond_evidence,
                "replacement_requests",
                lambda *_unused_args: [
                    {
                        "acquired": False,
                        "window": [10, 20],
                        "seat": "S",
                        "slot_id": 1,
                        "t": 20,
                    }
                ],
            )
            hand = factories.hand(entry=ENTRY, t1=30.0)
            acquire_replacements(hand, turns, models, tmp_path, report)

    with pytest.raises(failure, match="acquisition failed"):
        acquire()
    assert report == Report()  # A failed read is no question about missing tiles.


@pytest.mark.parametrize("operation", ["score", "tenpai", "replay"])
def test_scoring_dependency_failures_propagate(
    monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    def broken(*_unused_args: object, **_unused_kwargs: object) -> None:
        raise RuntimeError("scoring dependency failed")

    monkeypatch.setattr(scoring.HandCalculator, "estimate_hand_value", broken)
    monkeypatch.setattr(Shanten, "calculate_shanten", broken)
    operations = {
        "score": lambda: scoring.score_hand(CONCEALED, "5p", [], context=SCORE_CONTEXT),
        "tenpai": lambda: scoring.is_tenpai(CONCEALED),
        "replay": lambda: tenhou6._wins(
            Counter(tenhou6.tile(t) for t in [*CONCEALED, "5p"])
        ),
    }
    with pytest.raises(RuntimeError, match="scoring dependency failed"):
        operations[operation]()


def test_malformed_meld_schema_is_not_a_scoring_result() -> None:
    with pytest.raises(KeyError, match="unknown"):
        scoring.score_hand(
            CONCEALED,
            "5p",
            [{"type": "unknown", "tiles": ["1z"] * 3}],
            context=SCORE_CONTEXT,
        )


def test_domain_rejections_remain_explicit() -> None:
    assert not scoring.score_hand(["1m"] * 4, "1m", [], context=SCORE_CONTEXT).ok
    result = scoring.score_hand(CONCEALED, "9p", [], context=SCORE_CONTEXT)
    assert not result.ok
    assert result.error
    assert not tenhou6._wins(Counter({11: 3}))
    assert not tenhou6._wins(Counter({11: 15}))


def test_missing_seat_mapping_cannot_assume_starting_winds() -> None:
    entry = {
        **ENTRY,
        "corner_site": {"TL": "SOUTH", "TR": "WEST", "BL": "NORTH", "BR": "EAST"},
    }
    assert site_seat("EAST", entry) == "N"
    assert site_seat_name("N", entry) == "EAST"
    del entry["corner_site"]["BR"]
    with pytest.raises(StopIteration):
        site_seat("EAST", entry)
    with pytest.raises(StopIteration):
        site_seat_name("invalid", entry)


def test_obsolete_result_annotation_requires_explicit_current_schema() -> None:
    entry = {**ENTRY, "game": 0, "kyoku": 0, "honba": 0}
    fact = {"game": 0, "kyoku": 0, "honba": 0, "kind": "result", "ura": ["1p"]}
    with pytest.raises(ValueError, match="Unsupported result annotation"):
        facts_for_hand([fact], entry)
    assert fact["ura"] == ["1p"]
    assert fact["kind"] == "result"
    current = {"game": 0, "kyoku": 0, "honba": 0, "kind": "ura", "tiles": ["1p"]}
    assert facts_for_hand([current], entry)["ura"] == ["1p"]
