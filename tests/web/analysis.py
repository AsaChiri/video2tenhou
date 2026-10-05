# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Publish small, consistent analysis outputs the way the pipeline commands do."""

from __future__ import annotations

import json
from pathlib import Path

from video2tenhou import record
from video2tenhou.engine.decode import DECODER_VERSION
from video2tenhou.engine.review import decode_context, load_facts
from video2tenhou.export import EXPORT_INPUTS, decode_identity
from video2tenhou.files import atomic_write_json

SEATS = {"TL": "E", "TR": "S", "BL": "W", "BR": "N"}


def write_analysis(
    root: Path, name: str, game_ids: list[int], hands_per_game: int = 1
) -> list[dict]:
    """Write hands, record, decodes and exports of a completed analysis."""
    work = root / "work" / name
    work.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "hand": game * hands_per_game + i,
            "game": game,
            "kyoku": i,
            "honba": 0,
            "site_index": i,
            "t_start": 100.0 * (game * hands_per_game + i),
            "t_end": 100.0 * (game * hands_per_game + i) + 90,
            "corner_wind": SEATS,
        }
        for game in range(len(game_ids))
        for i in range(hands_per_game)
    ]
    games = [
        record.Game(
            game_id,
            {},
            {},
            [record.HandResult(i, 0, 0, {}, "draw") for i in range(hands_per_game)],
        )
        for game_id in game_ids
    ]
    (work / "hands.json").write_text(json.dumps(entries), encoding="utf-8")
    (work / "record.json").write_text(
        json.dumps([record.to_dict(g) for g in games]), encoding="utf-8"
    )
    publish(root, name)
    return entries


def publish(
    root: Path,
    name: str,
    hands: list[int] | None = None,
    facts: list[dict] | None = None,
) -> None:
    """Decode hands with saved answers and rewrite every export.

    Mirrors `video2tenhou rebuild`: decodes bind their context, and the export
    records the decode files it was built from. `facts` are the answers read
    when the command started (default: the current ones).
    """
    work, out = root / "work" / name, root / "out" / name
    entries = json.loads((work / "hands.json").read_text(encoding="utf-8"))
    games = [
        record.from_dict(row)
        for row in json.loads((work / "record.json").read_text(encoding="utf-8"))
    ]
    if facts is None:
        facts = load_facts(root / "labels" / name)
    for entry in entries:
        if hands is not None and entry["hand"] not in hands:
            continue
        result = games[entry["game"]].hands[entry["site_index"]]
        atomic_write_json(
            work / "decode" / f"{entry['hand']:02d}.json",
            {
                **{key: entry[key] for key in ("hand", "game", "kyoku", "honba")},
                "decoder_version": DECODER_VERSION,
                "items": [],
                "notes": [],
                "stats": {"turns": 0, "calls": 0},
                "score": None,
                "decode_context": decode_context(entry, result, facts),
            },
        )
    out.mkdir(parents=True, exist_ok=True)
    for game in range(len(games)):
        (out / f"g{game}.json").write_text('{"log": []}', encoding="utf-8")
        (out / f"g{game}.html").write_text("<h1>Replay links</h1>", encoding="utf-8")
    (out / "review.json").write_text("[]", encoding="utf-8")
    atomic_write_json(
        out / EXPORT_INPUTS,
        {
            "decodes": {
                str(e["hand"]): decode_identity(
                    work / "decode" / f"{e['hand']:02d}.json"
                )
                for e in entries
            }
        },
    )
