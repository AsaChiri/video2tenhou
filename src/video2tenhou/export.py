# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 6: game logs, viewer links, confidence, review queue and report.

Every file is published atomically because the studio may read it while a rebuild
writes it. `export-inputs.json` records the decode files each export was built from,
so a reader can tell whether the exports reflect the current reconstructions.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from .engine.assemble import game_from_decodes
from .engine.validation import review_artifact
from .files import atomic_write_json, atomic_write_text, read_published_text

if TYPE_CHECKING:
    from pathlib import Path

    from .record import Game

LOGGER = logging.getLogger(__name__)
EXPORT_INPUTS = "export-inputs.json"
MISSING_DECODE = (
    "No current reconstruction is available. Analyze this recording again to "
    "rebuild missing or outdated evidence."
)


def decode_identity(path: Path) -> list[int] | None:
    """Identify one decode file version by modification time and size."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return [stat.st_mtime_ns, stat.st_size]


def export_inputs(out: Path) -> dict[str, list[int] | None]:
    """Return the decode identities the current exports were built from, by hand."""
    try:
        return json.loads(read_published_text(out / EXPORT_INPUTS))["decodes"]
    except FileNotFoundError:
        return {}


def hand_status(d: dict, *, left_out: bool) -> str:
    """Classify a hand as complete, needing review or in conflict.

    A conflict has no legal reconstruction or a log the replayer rejects, so nothing
    is written for it.
    """
    if left_out or any(i["kind"] == "conflict" for i in d["items"]):
        return "conflict"
    return "review" if d["items"] else "complete"


def _score(d: dict) -> str:
    sc = d["score"]
    if not sc:
        return "-"
    if sc["match"]:
        return "ok"
    return sc["error"] or f"{sc['han']}/{sc['fu']} vs {sc['site'][0]}/{sc['site'][1]}"


def write_outputs(
    out: Path,
    games: list[Game],
    decodes: list[dict],
    entries: list[dict],
    *,
    decode_dir: Path,
) -> None:
    """Write g<k>.json/.html/.confidence.json, review.json and report.md atomically.

    Logs are titled with the recording name (`out`'s name). Hands left out of a log
    (conflicts) and hands without a decode become conflict review items. `decode_dir`
    holds the decode files `decodes` were read from; their identities are recorded
    last, so an interrupted export never looks current.
    """
    out.mkdir(parents=True, exist_ok=True)
    report, review, diagnostics = [], [], []
    written = 0
    for gi, game in enumerate(games):
        ds = [d for d in decodes if d["game"] == gi]
        expected = [e for e in entries if e["game"] == gi]
        g, conf, left_out = game_from_decodes(ds, entries, game, out.name)
        written += len(g.kyokus)
        atomic_write_text(out / f"g{gi}.json", g.dumps(), retry_windows=True)
        atomic_write_text(out / f"g{gi}.html", g.links_html(), retry_windows=True)
        atomic_write_json(
            out / f"g{gi}.confidence.json",
            {str(h): rows for h, rows in conf.items()},
            indent=1,
            retry_windows=True,
        )
        for decoded in sorted(ds, key=lambda d: d["hand"]):
            entry = next(e for e in entries if e["hand"] == decoded["hand"])
            d = review_artifact(decoded, entry, left_out.get(decoded["hand"], []))
            # Unseen tiles (draws, kan indicators) are written as the rules' guess and
            # stay open questions until the reviewer supplies them or says Can't tell.
            lost = sum(1 for r in d.get("confidence", []) if r.get("lost"))
            report.append(
                f"| {d['hand'] + 1} | {gi + 1} | {d['kyoku']}/{d['honba']} | "
                f"{hand_status(d, left_out=d['hand'] in left_out)} | "
                f"{('no' if d['hand'] in left_out else 'yes')} | "
                f"{d['stats']['turns']} | {d['stats']['calls']} | {len(d['items'])} "
                f"| {lost} | {_score(d)} |"
            )
            diagnostics += [
                f"- hand {d['hand'] + 1}: {line}" for line in d.get("diagnostics", [])
            ]
            for it in d["items"]:
                # An item's own `hand` is a list of tiles, not the hand number.
                row = dict(it)
                if isinstance(row.get("hand"), list):
                    row["tiles"] = row.pop("hand")
                review.append({**row, "hand": d["hand"]})
        decoded_ids = {d["hand"] for d in ds}
        for entry in expected:
            if entry["hand"] in decoded_ids:
                continue
            review.append(
                {"kind": "conflict", "hand": entry["hand"], "text": MISSING_DECODE}
            )
            report.append(
                f"| {entry['hand'] + 1} | {gi + 1} | "
                f"{entry['kyoku']}/{entry['honba']} | conflict | no "
                "| 0 | 0 | 1 | 0 | - |"
            )
        report.append(
            f"\nhanchan {gi + 1}: {len(g.kyokus)} of {len(expected)} hands written"
        )
        report += [
            f"  - hand {h + 1} left out: "
            + (f"{v[0]}" if v else "no legal reconstruction (a conflict)")
            + (f" (+{len(v) - 1} more)" if len(v) > 1 else "")
            for h, v in sorted(left_out.items())
        ]
        report.append("")
    atomic_write_json(out / "review.json", review, indent=1, retry_windows=True)
    head = (
        "| hand | game | kyoku/honba | status | written | turns | calls | open "
        "items | lost | score |\n|---|---|---|---|---|---|---|---|---|---|\n"
    )
    if diagnostics:
        report += [
            "<details><summary>Diagnostics</summary>\n",
            *diagnostics,
            "\n</details>",
        ]
    atomic_write_text(
        out / "report.md", head + "\n".join(report) + "\n", retry_windows=True
    )
    atomic_write_json(
        out / EXPORT_INPUTS,
        {
            "decodes": {
                str(d["hand"]): decode_identity(decode_dir / f"{d['hand']:02d}.json")
                for d in decodes
            }
        },
        retry_windows=True,
    )
    LOGGER.info(
        "[6 write] %s of %s hands written, %s open questions -> %s",
        written,
        len(entries),
        len(review),
        out,
    )
