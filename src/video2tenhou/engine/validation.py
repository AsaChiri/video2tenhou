# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""One review view of a reconstruction, including export legality and hand notes."""

from collections import defaultdict

from video2tenhou.record import HandResult
from video2tenhou.tenhou6 import replay_kyoku

from . import rules
from .assemble import kyoku_from_decode
from .hand import corner_of, site_seat_name


def export_conflicts(decoded: dict, entry: dict, violations: list[str]) -> list[dict]:
    """Locate export failures without counting discards twice."""
    if not violations:
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
        if tile in rules.KINDS or tile in rules.REDS
        if len(locations) > rules.max_count(tile)
    ]
    return [
        {
            "kind": "conflict",
            "stage": "export",
            "over": over,
            "text": "This hand cannot be exported: " + "; ".join(violations),
        }
    ]


def review_artifact(
    decoded: dict, entry: dict, violations: list[str] | None = None
) -> dict:
    """Refresh review from current data, including saved results from earlier runs."""
    out = dict(decoded)
    notices = [
        item for item in decoded.get("items", []) if item["kind"] == "solver_incomplete"
    ]
    out["items"] = [
        item
        for item in decoded.get("items", [])
        if item["kind"] != "solver_incomplete" and item.get("stage") != "export"
    ]
    out["notes"] = list(decoded.get("notes", []))
    if notices or any(
        row.get("state") == "unresolvable" for row in decoded.get("confidence", [])
    ):
        note = "Some automatic checks reached their time limit."
        if note not in out["notes"]:
            out["notes"].append(note)
    if decoded.get("solver", {}).get("status") in ("optimal", "feasible", "repaired"):
        if violations is None:
            result = decoded["result"]
            site = result.get("site", [result.get("han"), result.get("fu")])
            hand_result = HandResult(
                entry["kyoku"],
                entry["honba"],
                entry["sticks"],
                result["deltas"],
                result["outcome"],
                han=site[0],
                fu=site[1],
                riichi=result["riichi"]
                if "riichi" in result
                else [
                    site_seat_name(seat, entry) for seat in decoded.get("riichi", [])
                ],
            )
            kyoku, _ = kyoku_from_decode(decoded, entry, hand_result)
            violations = replay_kyoku(kyoku.dump())
        out["items"].extend(export_conflicts(decoded, entry, violations))
        out["validation"] = {"ok": not violations, "errors": violations}
    return out
