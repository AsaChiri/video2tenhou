# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Recorded evidence honors narrow human answers without inventing a full hand fact."""

import copy
import gzip
import json

import pytest

from tests.engine.factories import hand_decoder
from tests.integration.helpers import restore_model
from tests.paths import DATA
from video2tenhou.engine.assemble import kyoku_from_decode
from video2tenhou.engine.decode import HandDecoder
from video2tenhou.engine.review import facts_for_hand
from video2tenhou.engine.scoring import WinContext, score_hand
from video2tenhou.engine.solver import Solution
from video2tenhou.record import HandResult
from video2tenhou.tenhou6 import replay_kyoku


def recorded_case(index: "int") -> "dict":
    """Load one adjudicated case from the curated recognition fixture."""
    with gzip.open(
        DATA / "week11_adjudicated_models.json.gz", "rt", encoding="utf-8"
    ) as stream:
        return next(case for case in json.load(stream) if case["hand"] == index)


@pytest.mark.parametrize("index", [7, 24])
def test_confirmed_fields_survive_recorded_model_and_legal_export(index: int) -> None:
    """Verify confirmed fields survive recorded model and legal export."""
    case = recorded_case(index)
    model = restore_model(case["model"])
    facts = facts_for_hand(case["new_facts"], case["entry"])
    decoder = hand_decoder(facts=facts, problems=[])
    HandDecoder._apply_hand_facts(decoder, model)
    reference = case["reference"]
    prior = Solution(
        "optimal",
        reference["solver"]["objective"],
        reference["haipai"],
        {
            (key.split(":")[0], int(key.split(":")[1])): tile
            for key, tile in reference["draws"].items()
        },
        {},
    )
    sol = model.solve(time_limit=60, workers=2, margins=False, prior=prior)
    assert sol.ok
    for key, tile in model.facts.draws.items():
        assert sol.draws[key] == tile
    if index == 7:
        assert model.turns["S"][17].discard == "1m"
        assert model.turns["S"][17].discard_p is None
        assert ("S", 17) not in sol.discards
    else:
        assert sol.draws[("S", 0)] == "4z"
        assert sol.haipai["S"].count("2p") == sol.haipai["S"].count("4z") == 1
        assert "S" not in model.facts.haipai  # only two counts were reviewed
        assert model.win is not None
        winner = model.win.seat
        concealed = list(sol.hands[(winner, model.win.j)])
        winning = sol.draws[(winner, model.win.j)]
        concealed.remove(winning)
        context = reference["score"]["context"]
        score = score_hand(
            concealed,
            winning,
            reference["score"]["melds"],
            context=WinContext(
                tsumo=True,
                riichi=winner in reference["riichi"],
                seat=winner,
                round_wind="S",
                dora=reference["dora"],
                ura=context.get("ura", reference["ura"]),
                ippatsu=context["ippatsu"],
                haitei=context["haitei"],
                rinshan=context["rinshan"],
            ),
        )
        assert score.ok
        assert (score.han, score.fu) == (6, 20)

    decoded = copy.deepcopy(reference)
    decoded["haipai"] = sol.haipai
    decoded["draws"] = {f"{seat}:{j}": tile for (seat, j), tile in sol.draws.items()}
    for turn in decoded["turns"]:
        key = turn["seat"], turn["j"]
        turn["draw"] = sol.draws.get(key)
        turn["draw2"] = sol.draws2.get(key)
        turn["discard"] = sol.discards.get(key, model._discard_of(key))
        turn["tsumogiri"] = (turn["draw2"] or turn["draw"]) == turn["discard"]
    kyoku, _ = kyoku_from_decode(decoded, case["entry"], HandResult(**case["result"]))
    assert replay_kyoku(kyoku.dump()) == []
