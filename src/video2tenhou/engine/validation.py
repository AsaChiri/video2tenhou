# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""One review view of a reconstruction, including its export legality."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from video2tenhou.record import HandResult
from video2tenhou.tenhou6 import replay_kyoku

from . import rules
from .assemble import kyoku_from_decode, stream_times
from .hand import corner_of

if TYPE_CHECKING:
    from video2tenhou.tenhou6 import Violation


def violation_rows(
    decoded: dict, entry: dict, violations: list[Violation]
) -> list[dict]:
    """Name each replay violation's seat by its wind in this hand, with its time.

    The time is the turn of the seat's draw or discard the replay stopped at, when
    the violation sits at one.
    """
    times = stream_times(decoded)
    rows = []
    for v in violations:
        seat = (
            None if v.player is None else rules.SEATS[(v.player - entry["kyoku"]) % 4]
        )
        t = None
        if seat is not None:
            draws, discards = times[seat]
            if v.discard is not None and v.discard < len(discards):
                t = discards[v.discard]
            elif v.draw is not None and v.draw < len(draws):
                t = draws[v.draw]
        rows.append(
            {"kind": v.kind, "seat": seat, "tile": v.tile, "t": t, "text": v.text}
        )
    return rows


def export_conflicts(decoded: dict, entry: dict, rows: list[dict]) -> list[dict]:
    """Ask about a log the replayer rejects, locating over-counted tiles once each."""
    if not rows:
        return []
    sources = defaultdict(list)
    start = decoded.get("play_window", [entry.get("t_start", 0)])[0]
    for seat, tiles in decoded.get("haipai", {}).items():
        for tile in tiles:
            sources[tile].append(
                {
                    "kind": "haipai",
                    "seat": seat,
                    "tile": tile,
                    "tiles": tiles,
                    "t": start,
                    "corner": corner_of(entry, seat),
                }
            )
    turns = {(t["seat"], t["j"]): t for t in decoded.get("turns", [])}
    for key, tile in decoded.get("draws", {}).items():
        seat, j = key.split(":")
        turn = turns.get((seat, int(j)), {})
        sources[tile].append(
            {
                "kind": "draw",
                "seat": seat,
                "j": int(j),
                "tile": tile,
                "t": turn.get("t", decoded.get("t_last", start)),
                "corner": corner_of(entry, seat),
            }
        )
    for turn in turns.values():
        if turn.get("draw2"):
            sources[turn["draw2"]].append(
                {
                    "kind": "replacement_draw",
                    "seat": turn["seat"],
                    "tile": turn["draw2"],
                    "t": turn["t"],
                    "corner": corner_of(entry, turn["seat"]),
                }
            )
    for field in ("dora", "ura"):
        for index, tile in enumerate(decoded.get(field, [])):
            sources[tile].append(
                {
                    "kind": "indicator",
                    "field": field,
                    "index": index,
                    "tile": tile,
                    "t": decoded.get("t_last", start),
                }
            )
    over = [
        {
            "tile": tile,
            "count": len(locations),
            "limit": rules.max_count(tile),
            "sources": locations,
        }
        for tile, locations in sources.items()
        if tile in rules.KINDS or tile in rules.PLAIN_OF
        if len(locations) > rules.max_count(tile)
    ]
    item = {
        "kind": "conflict",
        "stage": "export",
        "over": over,
        "violations": rows,
        "text": (
            "The log of this hand breaks the rules of play listed below, so it is not "
            "exported. Correct the reading that causes it."
        ),
    }
    times = [row["t"] for row in rows if row["t"] is not None]
    if times:
        item["t"] = min(times)
    return [item]


def review_artifact(
    decoded: dict, entry: dict, violations: list[Violation] | None = None
) -> dict:
    """Refresh the export check of a reconstruction without changing it.

    ``violations`` are the replayer's, when the caller already replayed the log.
    """
    out = dict(decoded)
    out["items"] = [
        item for item in decoded.get("items", []) if item.get("stage") != "export"
    ]
    if decoded.get("solver", {}).get("status") in ("optimal", "feasible", "repaired"):
        if violations is None:
            result = decoded["result"]
            han, fu = result["site"]
            hand_result = HandResult(
                entry["kyoku"],
                entry["honba"],
                entry["sticks"],
                result["deltas"],
                result["outcome"],
                han=han,
                fu=fu,
                riichi=result["riichi"],
            )
            kyoku, _ = kyoku_from_decode(decoded, entry, hand_result)
            violations = replay_kyoku(kyoku.dump())
        rows = violation_rows(decoded, entry, violations)
        out["items"].extend(export_conflicts(decoded, entry, rows))
        out["validation"] = {"ok": not rows, "errors": rows}
    return out
