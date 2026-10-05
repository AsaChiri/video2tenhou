# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""The rules of section 1 of docs/DESIGN.md as checkable functions and constants."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

SEATS = "ESWN"
# seats are the winds of the hand being decoded, so the dealer is always East
DEALER = "E"
# the seat at each offset after a player, in turn order
RELATIONS = ("self", "shimocha", "toimen", "kamicha")
# the 34 kinds: numbered suits, then honors
KINDS = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + [
    f"{n}z" for n in range(1, 8)
]
RED_OF = {"5m": "0m", "5p": "0p", "5s": "0s"}  # plain five -> the red five of its suit
PLAIN_OF = {red: five for five, red in RED_OF.items()}  # red five -> plain five
OTHER_FIVE = RED_OF | PLAIN_OF  # a five read as the other one
LIVE_WALL = 136 - 13 * 4 - 14  # 70 draws in a hand without kans


def plain(tile: str) -> str:
    """Red five -> plain five; everything else unchanged."""
    return PLAIN_OF.get(tile, tile)


def suit(tile: str) -> str:
    """Return m/p/s/z for a validated two-character tile token."""
    return tile[1]


def number(tile: str) -> int:
    """Return rank 1-9 (honors 1-7), treating a red zero as rank five."""
    n = int(tile[0])
    return 5 if n == 0 else n


def max_count(tile: str) -> int:
    """How many copies of exactly this token exist: red fives one, plain fives three."""
    if tile in PLAIN_OF:
        return 1
    if tile in RED_OF:
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
    return RELATIONS[(SEATS.index(other) - SEATS.index(me)) % 4]


def seat_at(me: str, relation: str) -> str:
    """Return the seat that is `relation` (kamicha, toimen, shimocha) of `me`."""
    return SEATS[(SEATS.index(me) + RELATIONS.index(relation)) % 4]


def kan_tiles(kind: str) -> list[str]:
    """Return the four tiles of a kan of `kind`: a kan of fives holds the red one."""
    k = plain(kind)
    red = RED_OF.get(k)
    return [k] * 3 + [red] if red else [k] * 4
