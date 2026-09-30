# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""The rules of section 1 of docs/DESIGN.md as checkable functions and constants."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

MELD_SIZE = 3


SEATS = "ESWN"
KINDS = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + [
    f"{n}z" for n in range(1, 8)
]  # 34
REDS = {"0m": "5m", "0p": "5p", "0s": "5s"}


def plain(tile: str) -> str:
    """Red five -> plain five; everything else unchanged."""
    return REDS.get(tile, tile)


def suit(tile: str) -> str:
    """Return m/p/s/z for a validated two-character tile token."""
    return tile[1]


def number(tile: str) -> int:
    """Return rank 1-9 (honors 1-7), treating a red zero as rank five."""
    n = int(tile[0])
    return 5 if n == 0 else n


def max_count(tile: str) -> int:
    """Return the inventory limit of a tile, distinguishing red fives.

    How many copies of exactly this token exist: red fives once, plain fives three,
    others four.
    """
    if tile in REDS:
        return 1
    if tile in ("5m", "5p", "5s"):
        return 3
    return 4


def count_ok(tiles: Iterable[str]) -> bool:
    """Check physical copy limits while keeping red and plain fives distinct."""
    c = Counter(tiles)
    return all(c[t] <= max_count(t) for t in c)


def next_seat(seat: str) -> str:
    """Advance one draw turn in E/S/W/N order, wrapping from North to East."""
    return SEATS[(SEATS.index(seat) + 1) % 4]


def relative(me: str, other: str) -> str:
    """Kamicha (the player before me), toimen, shimocha (after me)."""
    d = (SEATS.index(other) - SEATS.index(me)) % 4
    return {0: "self", 1: "shimocha", 2: "toimen", 3: "kamicha"}[d]


def is_chi(tiles: list[str]) -> bool:
    """Check a three-tile sequence, accepting red fives and excluding honors."""
    if len(tiles) != MELD_SIZE:
        return False
    s = {suit(t) for t in tiles}
    if len(s) != 1 or "z" in s:
        return False
    ns = sorted(number(t) for t in tiles)
    return ns[1] == ns[0] + 1 and ns[2] == ns[1] + 1


def is_pon(tiles: list[str]) -> bool:
    """Check three matching ranks; physical copy limits are checked separately."""
    return len(tiles) == MELD_SIZE and len({plain(t) for t in tiles}) == 1


def kan_tiles(kind: str) -> list[str]:
    """Return all four tiles of a kan, preserving its single red five.

    The four tiles of any kan of `kind`: a kan of fives is all four fives, three plain
    and the red one.
    """
    k = plain(kind)
    red = next((r for r, p in REDS.items() if p == k), None)
    return [k] * 3 + [red] if red else [k] * 4


DEALER = (
    "E"  # seats are the winds of the hand being decoded, so the dealer is always East
)


LIVE_WALL = 136 - 13 * 4 - 14  # 70 draws in a hand without kans
