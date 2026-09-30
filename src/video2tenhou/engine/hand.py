# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""The hand being decoded: who sits where, and which observations are its own.

Seats are the winds of this hand (E deals); a player's identity is the corner of the
table (a fixed chair and camera). The site record names players by the wind they held in
the hanchan's first hand.
"""

from __future__ import annotations

from collections import Counter
from typing import overload


def seat_of(entry: dict, corner: str) -> str:
    """Map a fixed camera corner to its wind in this hand, after dealer rotation."""
    return entry["corner_wind"][corner]


def corner_of(entry: dict, seat: str) -> str:
    """Find a hand-relative wind's camera, rejecting missing calibration."""
    return next(c for c, s in entry["corner_wind"].items() if s == seat)


@overload
def site_seat(name: str, entry: dict) -> str: ...


@overload
def site_seat(name: None, entry: dict) -> None: ...


def site_seat(name: str | None, entry: dict) -> str | None:
    """Map a site's starting-seat name to the wind held in this hand.

    This hand's wind of the player the site record names (EAST..NORTH, the winds of the
    first hand).
    """
    if name is None:
        return None
    corner = next(c for c, n in entry["corner_site"].items() if n == name)
    return entry["corner_wind"][corner]


def site_seat_name(seat: str, entry: dict) -> str:
    """Map a hand's wind to the site's starting-seat name.

    The site record's name (EAST..NORTH) for the player who holds this wind in this
    hand.
    """
    return entry["corner_site"][corner_of(entry, seat)]


def in_window(obs: list[dict], t0: float, t1: float) -> list[dict]:
    """Select observations within the hand's clearing and play window.

    The observations of the hand's window. The window starts with the clearing of the
    previous hand (`ponds.play_window`): a view that began before it shows that hand's
    tiles, or what the push left.
    """
    return [o for o in obs if t0 < o["t0"] <= t1]


# s after a meld event over which the hand camera is asked whether a meld was laid
MELD_LOOK = 30.0


def melds_shown(obs: dict, entry: dict, seat: str, lo: float, hi: float) -> int | None:
    """Infer the number of laid melds from the observed resting hand size.

    How many melds (chi, pon, kan) the seat's hand camera says were laid by then: a
    resting hand of 13, 10, 7 ... tiles (14, 11, 8 ... just after a draw) is 0, 1, 2 ...
    melds. The most frequent over the full views in [lo, hi]; None when there is none.
    """
    if not obs:
        return None
    n: Counter = Counter()
    for o in in_window(obs.get(f"hand:{corner_of(entry, seat)}", []), lo, hi):
        c = o["count"]
        if o["n_used"] and not o.get("partial") and c > 0 and c % 3 in (1, 2):
            n[(13 - c) // 3 if c % 3 == 1 else (14 - c) // 3] += 1
    return n.most_common(1)[0][0] if n else None
