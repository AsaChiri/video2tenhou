# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic posteriors, voted slots and observations shared across tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from video2tenhou.engine.solver import TI, TILES
from video2tenhou.files import sha256_text
from video2tenhou.perception.tiles import CLASS_INDEX, CLASSES


def posterior(
    tile: str,
    peak: float = 0.9,
    *,
    floor: float = 0.002,
    second: str | None = None,
    second_peak: float = 0.0,
    normalized: bool = True,
) -> np.ndarray:
    """Classifier posterior peaked on `tile`, optionally competing with `second`."""
    p = np.full(len(CLASSES), floor)
    p[CLASS_INDEX[tile]] = peak
    if second is not None:
        p[CLASS_INDEX[second]] = second_peak
    return p / p.sum() if normalized else p


def one_hot(tile: str) -> list[float]:
    """Certain classifier posterior as stored in reading files."""
    return posterior(tile, 1.0, floor=0.0, normalized=False).tolist()


def tile_counts(tiles: Sequence[str]) -> np.ndarray:
    """Count tiles in the solver's red-five-aware vocabulary."""
    e = np.zeros(len(TILES))
    for tile in tiles:
        e[TI[tile]] += 1
    return e


def kinds(probabilities: dict[str, float]) -> np.ndarray:
    """Solver-vocabulary posterior from tile probabilities."""
    p = np.zeros(len(TILES))
    for tile, value in probabilities.items():
        p[TI[tile]] = value
    return p


def slot(
    tile: str,
    *,
    key: list,
    xyxy: list[float],
    p: np.ndarray | None = None,
    conf: float = 0.9,
    sideways: float = 0.0,
    seen: int = 3,
) -> dict:
    """Voted observation slot; `p` defaults to ``posterior(tile)``."""
    return {
        "key": key,
        "tile": tile,
        "conf": conf,
        "seen": seen,
        "sideways": sideways,
        "disagree": False,
        "xyxy": xyxy,
        "p": (posterior(tile) if p is None else p).tolist(),
    }


def observation(
    region: str,
    t0: float,
    t1: float,
    slots: Sequence[dict] = (),
    *,
    indicators: Sequence[dict] = (),
    n_used: int = 3,
    partial: bool = False,
) -> dict:
    """Voted observation of one calm interval."""
    return {
        "region": region,
        "t0": t0,
        "t1": t1,
        "n_readings": n_used,
        "n_used": n_used,
        "count": len(slots),
        "quality": 0.9,
        "slots": list(slots),
        "indicators": list(indicators),
        "partial": partial,
    }


def pond_slot(
    tile: str, row: int, col: int, sideways: float = 0.0, seen: int = 3
) -> dict:
    """Pond slot at its (row, column) stack position on a 40 x 60 px grid."""
    return slot(
        tile,
        key=[row, col],
        xyxy=[col * 40, row * 60, col * 40 + 38, row * 60 + 58],
        sideways=sideways,
        seen=seen,
    )


def row(tiles: Sequence[str], r: int = 0, start: int = 0) -> list[dict]:
    """Pond slots laid consecutively in row `r` from column `start`."""
    return [pond_slot(t, r, start + i) for i, t in enumerate(tiles)]


def obs(
    t0: float,
    t1: float,
    slots: Sequence[dict],
    *,
    n_used: int = 3,
    partial: bool = False,
) -> dict:
    """Observation of the top-left pond."""
    return observation("pond:TL", t0, t1, slots, n_used=n_used, partial=partial)


def publish_reads(hand_dir: Path, tile: str = "1m", **manifest: object) -> None:
    """Publish three certain top-left pond readings, then a manifest naming them."""
    box = {"xyxy": [10, 10, 50, 70], "conf": 0.9, "sideways": False, "p": one_hot(tile)}
    text = "".join(
        json.dumps({"t": t, "region": "pond:TL", "size": [400, 700], "boxes": [box]})
        + "\n"
        for t in (0.0, 1.0, 2.0)
    )
    hand_dir.mkdir(parents=True, exist_ok=True)
    (hand_dir / "pond_TL.jsonl").write_text(text, encoding="utf-8")
    (hand_dir / "done.json").write_text(
        json.dumps({**manifest, "outputs": {"pond:TL": sha256_text(text)}})
    )
