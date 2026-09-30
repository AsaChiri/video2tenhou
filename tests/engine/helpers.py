# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic pond observations shared by tracker and acquisition tests."""

import numpy as np

from video2tenhou.train.data import CLASS_INDEX, CLASSES


def slot(
    tile: "str", row: "int", col: "int", sideways: float = 0.0, seen: int = 3
) -> "dict":
    """Build a normalized synthetic tracked-slot observation."""
    p = np.full(len(CLASSES), 0.002)
    p[CLASS_INDEX[tile]] = 0.9
    p /= p.sum()
    return {
        "key": [row, col],
        "tile": tile,
        "conf": 0.9,
        "seen": seen,
        "sideways": sideways,
        "disagree": False,
        "xyxy": [col * 40, row * 60, col * 40 + 38, row * 60 + 58],
        "p": p.tolist(),
    }


def obs(
    t0: "float",
    t1: "float",
    slots: "list[dict]",
    n_used: int = 3,
    *,
    partial: bool = False,
) -> "dict":
    """Build one pond observation interval with explicit support counts."""
    return {
        "region": "pond:TL",
        "t0": t0,
        "t1": t1,
        "n_readings": n_used,
        "n_used": n_used,
        "count": len(slots),
        "quality": 0.9,
        "slots": slots,
        "indicators": [],
        "partial": partial,
    }


def row(tiles: "list[str]", r: int = 0, start: int = 0) -> "list[dict]":
    """Place synthetic tile slots consecutively in a pond row."""
    return [slot(t, r, start + i) for i, t in enumerate(tiles)]
