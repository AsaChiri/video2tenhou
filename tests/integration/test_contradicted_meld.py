# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""A later partial camera view must not erase an independently witnessed call."""

import copy
import gzip
import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.paths import DATA
from video2tenhou.engine.calls import CallAnchor
from video2tenhou.engine.dense import DenseContext, place_taken
from video2tenhou.engine.melds import Call, track_melds
from video2tenhou.engine.ponds import tail_runs, track_pond
from video2tenhou.train.data import CLASS_INDEX, CLASSES

if TYPE_CHECKING:
    from video2tenhou.engine.ponds import PondSlot


def recorded() -> "dict":
    """Load the recorded contradictory-chi evidence fixture."""
    return json.loads(
        gzip.decompress((DATA / "week11_contradicted_chi.json.gz").read_bytes())
    )


def hypothesis(data: "dict") -> "Call":
    """Select the recorded chi hypothesis that needs further verification."""
    calls = track_melds("E", data["meld_observations"], include_contradicted=True)
    return next(c for c in calls if c.type == "chi" and c.t_first == 9818)


def source_logs(data: "dict") -> "tuple[dict[str, list[PondSlot]], PondSlot]":
    """Build pond logs including the recorded short-lived called discard."""
    logs = {s: [] for s in "ESWN"}
    logs["N"] = track_pond(data["pond_observations"])
    runs = tail_runs(logs["N"], data["source_window"]["readings"])
    source = next(
        r
        for r in runs
        if r["gone"] and r["tile"] == "0p" and 9810 <= r["t_first"] <= 9812
    )
    slot = place_taken("N", source, logs)
    return logs, slot


def test_recorded_chi_requires_independent_removed_discard() -> None:
    """Verify recorded chi requires independent removed discard."""
    data = recorded()
    ordinary = track_melds("E", data["meld_observations"])
    assert not any(c.type == "chi" and c.t_first == 9818 for c in ordinary)
    ev = hypothesis(data)
    assert ev.contradicted
    assert ev.seen == 3
    assert ev.absent == 5
    assert sorted(ev.tiles) == ["0p", "3p", "4p"]
    assert ev.to_dict()["contradicted"] is True
    assert (
        CallAnchor(
            {s: [] for s in "ESWN"},
            context=DenseContext(
                entry={}, models=None, work_dir=None, t0=0, problems=[]
            ),
        ).anchor([ev], 9500, 9970, None)
        == []
    )
    logs, slot = source_logs(data)
    anchor = CallAnchor(
        logs,
        context=DenseContext(entry={}, models=None, work_dir=None, t0=0, problems=[]),
    )
    calls = anchor.anchor([ev, copy.deepcopy(ev)], 9500, 9970, None)
    assert len(calls) == 1  # the competing hypothesis cannot claim the same tile
    call = calls[0]
    assert call.type == "chi"
    assert call.source == "kamicha"
    assert call.called_tile == "0p"
    assert sorted(call.tiles) == ["0p", "3p", "4p"]
    assert call.anchor == "discard"
    assert call.contradicted
    assert call.absent == 5
    assert id(slot) in anchor.claimed


@pytest.mark.parametrize("mismatch", ["not_removed", "wrong_source", "wrong_tile"])
def test_contradicted_camera_event_cannot_claim_unrelated_evidence(
    mismatch: str,
) -> None:
    """Verify contradicted camera event cannot claim unrelated evidence."""
    data = recorded()
    ev = hypothesis(data)
    logs, slot = source_logs(data)
    if mismatch == "not_removed":
        slot.t_removed = None
    elif mismatch == "wrong_source":
        logs["W"] = [slot]
        logs["N"] = []
    else:
        slot.p = np.eye(len(CLASSES))[CLASS_INDEX["7p"]]
    assert (
        CallAnchor(
            logs,
            context=DenseContext(
                entry={}, models=None, work_dir=None, t0=0, problems=[]
            ),
        ).anchor([ev], 9500, 9970, None)
        == []
    )


def test_contradicted_external_kan_has_no_self_kan_fallback() -> None:
    """Verify contradicted external kan has no self kan fallback."""
    p = np.eye(len(CLASSES))[CLASS_INDEX["2m"]]
    ev = Call(
        "E",
        50,
        (45, 50),
        "kan",
        ["2m"] * 4,
        0,
        "kamicha",
        "2m",
        [p] * 4,
        1,
        seen=2,
        absent=3,
        contradicted=True,
    )
    anchor = CallAnchor(
        {s: [] for s in "ESWN"},
        context=DenseContext(entry={}, models=None, work_dir=None, t0=0, problems=[]),
    )
    assert anchor.anchor([ev], 10, 90, None) == []
