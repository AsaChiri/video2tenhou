# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Locate hands from table clearings; take identities and scores from scoremj."""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from . import calm, video
from .cache import source_identity
from .engine.ponds import clearings
from .files import atomic_write_json
from .layout import CORNERS, Calibration
from .perception.reader import Box, assign_pond
from .perception.tiles import CLASSES
from .record import SEAT_LETTER, SEATS, Game

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from video2tenhou.perception.reader import RegionDetector
    from video2tenhou.read import ReadContext

POND_DETECTION_CONFIDENCE = 0.4
MIN_HAND_WINDOW = 30


# PML's first dealer is top left; physical chairs stay fixed within a game.
STARTING_SEATS = {"TL": "EAST", "BL": "SOUTH", "BR": "WEST", "TR": "NORTH"}
FPS = 2.0


LOGGER = logging.getLogger("video2tenhou.timeline")


def read_pond_counts(
    path: str | Path,
    cal: Calibration,
    intervals: list[calm.Interval],
    detector: RegionDetector,
    log: Callable[[str], None] = LOGGER.info,
) -> dict[str, list[dict]]:
    """Count discards on calm pond frames without classifying broadcast text."""
    requests = defaultdict(list)
    for iv in intervals:
        if iv.region.startswith("pond:") and iv.calm and not iv.partial:
            tick = round((iv.t0 + iv.t1) * FPS / 2)
            requests[tick].append(iv)
    out = {corner: [] for corner in CORNERS}
    if not requests:
        return out
    done = 0
    for t, frame in video.sample(path, fps=FPS):
        rows = requests.get(round(t * FPS))
        if not rows:
            continue
        images = [cal.region(frame, iv.region)[0] for iv in rows]
        for iv, image, detections in zip(
            rows, images, detector.predict_batch(images), strict=True
        ):
            # The face detector establishes face-up tiles. Row assignment only
            # needs that fact, box geometry, and the wall band, not tile identity.
            boxes = [
                Box(
                    d.xyxy,
                    d.conf,
                    sideways=d.xyxy[2] - d.xyxy[0] > 1.15 * (d.xyxy[3] - d.xyxy[1]),
                    p=np.zeros(len(CLASSES)),
                )
                for d in detections
                if d.conf >= POND_DETECTION_CONFIDENCE
            ]
            rejected = assign_pond(boxes, image.shape[0], image.shape[1])
            out[iv.region.partition(":")[2]].append(
                {
                    "t0": iv.t0,
                    "t1": iv.t1,
                    "count": sum(box.role == "tile" for box in boxes),
                    "partial": rejected,
                    "n_used": 1,
                }
            )
        done += len(rows)
        if done % 100 < len(rows):
            log(f"  table timing: {t:.0f} s")
    return out


def hand_windows(
    observations: dict[str, list[dict]], duration: float
) -> list[tuple[float, float]]:
    """Split at corroborated pond clearings, excluding empty lead-in/out spans."""
    cuts = clearings(observations, 0, duration)
    bounds = [0.0, *[t for t in cuts if 0 < t < duration], duration]
    windows = []
    for index, (start, end) in enumerate(itertools.pairwise(bounds)):
        active = [
            row
            for rows in observations.values()
            for row in rows
            if not row.get("partial")
            and row.get("n_used")
            and row["count"] > 0
            and start <= row["t0"] < end
        ]
        if index == 0:
            # A recording can begin with the previous game's finished ponds
            # still on screen. Require observed play before the first clearing;
            # a static table of old discards is not a new hand.
            growing = False
            for rows in observations.values():
                counts = [row["count"] for row in rows if row in active]
                if any(
                    count >= min(counts[:i]) + 2 for i, count in enumerate(counts) if i
                ):
                    growing = True
                    break
            if not growing:
                continue
        if end - start >= MIN_HAND_WINDOW and active:
            windows.append((start + (1 / FPS if index else 0), end))
    return windows


def site_entries(
    windows: list[tuple[float, float]], games: list[Game]
) -> tuple[list[dict], list[str]]:
    """Map observed hand windows in order to authoritative game/hand records."""
    expected = sum(len(game.hands) for game in games)
    if len(windows) != expected:
        return [], [
            (
                f"Table clearings identify {len(windows)} hands, but scoremj "
                f"lists {expected}. Check the pond calibration, recording range"
                " and game IDs before retrying."
            )
        ]
    entries = []
    for gi, game in enumerate(games):
        scores = dict.fromkeys(SEATS, 25000)
        for site_index, hand in enumerate(game.hands):
            start, end = windows[len(entries)]
            winds = {
                c: SEAT_LETTER[SEATS[(SEATS.index(s) - hand.kyoku) % 4]]
                for c, s in STARTING_SEATS.items()
            }
            entries.append(
                {
                    "hand": len(entries),
                    "game": gi,
                    "game_id": game.id,
                    "kyoku": hand.kyoku,
                    "honba": hand.honba,
                    "sticks": hand.sticks,
                    "t_start": start,
                    "t_end": end,
                    "corner_wind": winds,
                    "corner_site": dict(STARTING_SEATS),
                    "scores": {winds[c]: scores[s] for c, s in STARTING_SEATS.items()},
                    "nicks": {
                        c: game.players.get(s, "") for c, s in STARTING_SEATS.items()
                    },
                    "site_index": site_index,
                }
            )
            scores = {seat: scores[seat] + hand.deltas[seat] for seat in SEATS}
    return entries, []


def run_header(
    context: ReadContext, games: list[Game], *, force: bool = False
) -> tuple[list[dict], list[str]]:
    """Cache table evidence, then align physical hand order to scoremj records.

    Only the context's detector is used: it counts pond tiles when the
    table-timing cache, which binds its identity, is missing or stale.
    """
    path, cal, work = context.path, context.calibration, context.work
    detector = context.detector
    work.mkdir(parents=True, exist_ok=True)
    intervals = calm.run_calm(path, cal, work, force=force, log=LOGGER.info)
    signature = {
        "version": 1,
        "source": source_identity(path),
        "geometry": calm.geometry_key(cal),
        "detector": detector.id,
        "intervals": [
            iv.to_dict() for iv in intervals if iv.region.startswith("pond:")
        ],
        "fps": FPS,
    }
    cache_path = work / "table-timing.json"
    saved: dict | None = None
    if not force:
        with contextlib.suppress(FileNotFoundError, json.JSONDecodeError):
            saved = json.loads(cache_path.read_text(encoding="utf-8"))
    if not (isinstance(saved, dict) and saved.get("signature") == signature):
        observations = read_pond_counts(path, cal, intervals, detector, log=LOGGER.info)
        saved = {"signature": signature, "observations": observations}
        atomic_write_json(cache_path, saved)
    windows = hand_windows(saved["observations"], video.probe(str(path)).duration)
    entries, problems = site_entries(windows, games)
    if not problems:
        atomic_write_json(work / "hands.json", entries, indent=1)
    return entries, problems
