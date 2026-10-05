# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""What a decode tells the reviewer: questions, notes and answers it ignored.

A question (review item) is a short prompt with structured fields (seat, time in
seconds, tiles, candidates) that the browser renders without parsing the text. A
note is a short statement the reviewer benefits from on the hand page. An ignored
answer is a reviewer fact the decoder could not apply. Diagnostics are developer
reasoning: report.md keeps them and the browser never shows them. The decoding
stages' notes and prompts are written here (DESIGN.md 4.8 `review.py`, section 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import rules
from .confidence import confidence_state

if TYPE_CHECKING:
    from .confidence import Certificate
    from .melds import Call
    from .scoring import ScoreResult
    from .solver import Culprit

WINDS = {"E": "East", "S": "South", "W": "West", "N": "North"}
CONFIRMED_DISCARD_WINDOW = 3  # s between a reviewed discard and the repaired one
# Individual questions first; covered but uncertified tile choices are grouped
# after them, so a hand cannot look complete merely because images cover it.
PRIORITY = [
    "ura",
    "conflict",
    "result",
    "call",
    "order",
    "riichi",
    "dora",
    "kan",
    "draw",
]
FACT_NAMES = {"haipai": "starting hand", "draw": "draw", "final": "final hand"}


@dataclass
class Report:
    """The questions, notes, ignored answers and diagnostics of one decode."""

    items: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    ignored_facts: list[dict] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)

    def ignore(self, kind: str, fact: dict, reason: str) -> None:
        """Record a reviewer answer that the reconstruction could not apply."""
        self.ignored_facts.append(
            {
                "kind": kind,
                "seat": fact.get("seat"),
                "t": fact.get("t"),
                "reason": reason,
            }
        )


def clock(t: float) -> str:
    """Format video seconds as m:ss, the way the browser shows times."""
    s = int(max(0.0, t) + 0.5)
    return f"{s // 60}:{s % 60:02d}"


def ranked(items: list[dict]) -> list[dict]:
    """Order a hand's questions: the ura first, then a conflict, a result, a call."""
    return sorted(
        items,
        key=lambda it: (
            PRIORITY.index(it["kind"]) if it["kind"] in PRIORITY else len(PRIORITY)
        ),
    )


# ---- notes -------------------------------------------------------------------------


def site_corrected(site: tuple[int | None, int | None], used: tuple) -> str:
    """Note the han/fu the reviewer confirmed in place of the site's."""
    return (
        f"Scored as {used[0]}/{used[1]} instead of the site's {site[0]}/{site[1]}, "
        "as confirmed in review."
    )


def wall_overused(used: int) -> str:
    """Note a turn sequence that draws more tiles than the live wall holds."""
    return (
        f"The turn sequence uses {used} wall tiles, more than the {rules.LIVE_WALL} "
        "of the live wall: some of its turns did not happen."
    )


def wall_short(used: int) -> str:
    """Note an exhaustive draw whose turn sequence misses wall draws."""
    return (
        "The hand ended in an exhaustive draw, but the turn sequence uses only "
        f"{used} of the {rules.LIVE_WALL} wall tiles."
    )


def unexplained_removal(seat: str, tile: str, t: float) -> str:
    """Note a tile that left a pond with no call to take it."""
    return (
        f"{WINDS[seat]}'s {tile} left the pond at {clock(t)}, and no call explains it."
    )


def kan_unplaced(tile: str, t: float) -> str:
    """Note a new dora indicator whose kan no discard follows."""
    return (
        f"The dora indicator {tile} appeared at {clock(t)}, but no discard follows "
        "it, so its kan is not in the log."
    )


# ---- questions: the turn sequence --------------------------------------------------


def missed_discard(seat: str, t: float, i: int) -> dict:
    """Ask for a discard the merge needed and no read found."""
    return {
        "kind": "order",
        "seat": seat,
        "t": t,
        "i": i,
        "text": (
            "This player's turn came, but no read of the pond found the discard. "
            "Add the missed discard, or confirm that nothing is missing."
        ),
    }


def uncounted_meld(c: Call, i: int) -> dict:
    """Ask for the discard that would let a seen meld into the log."""
    return {
        "kind": "order",
        "seat": c.seat,
        "t": c.t_first,
        "i": i,
        "type": c.type,
        "tiles": list(c.tiles),
        "text": (
            "This player's meld was seen, but no discard follows it, so the log "
            "leaves the meld out. Add the discard made after the call, or confirm "
            "that nothing is missing."
        ),
    }


def unanchored_call(e: Call) -> dict:
    """Ask about a clear meld whose taken discard no pond read shows."""
    return {
        "kind": "call",
        "seat": e.seat,
        "t": e.t_first,
        "tile": e.tiles[e.called_pos or 0],
        "type": e.type,
        "tiles": e.tiles,
        "source": e.source,
        "text": (
            "The meld camera shows this meld, but no pond shows the discard it took."
            " What is the meld, and whose discard did it take?"
        ),
    }


def hidden_call(c: Call) -> dict:
    """Ask for the tiles of a call that only the turn order shows."""
    return {
        **_call_fields(c),
        "text": (
            "No camera shows this call; the turn order implies it, and the shown "
            "meld is the rules' guess. What was the meld?"
        ),
    }


def unnamed_kan(c: Call) -> dict:
    """Ask for the tile of a concealed kan that no camera named."""
    return {
        **_call_fields(c),
        "text": (
            "No camera names this concealed kan's tile; the shown one is the rules' "
            "guess. What was the kan?"
        ),
    }


def _call_fields(c: Call) -> dict:
    return {
        "kind": "call",
        "seat": c.seat,
        "t": c.t_first,
        "tile": c.called_tile or c.tiles[0],
        "type": c.type,
        "tiles": list(c.tiles),
        "source": c.source,
    }


# ---- questions: riichi, the dead wall ----------------------------------------------


def missing_riichi(seat: str, t: float, tile: str) -> dict:
    """Ask which discard declared a riichi that no turned tile shows."""
    return {
        "kind": "riichi",
        "seat": seat,
        "t": t,
        "tile": tile,
        "text": (
            "No turned tile shows this riichi, so the last discard is assumed. "
            "Which discard declared riichi?"
        ),
    }


def riichi_on_called_tile(seat: str, t: float, tile: str) -> dict:
    """Ask which of a called discard and the turned tile after it declared."""
    return {
        "kind": "riichi",
        "seat": seat,
        "t": t,
        "tile": tile,
        "text": (
            "The turned tile follows a discard that was called, so either declared "
            "riichi, and the hand fits both. Which discard declared riichi?"
        ),
    }


def kan_time(tile: str) -> dict:
    """Ask when the kan of a reviewer-named indicator happened."""
    return {
        "kind": "kan",
        "tile": tile,
        "text": (
            f"No view shows the dora indicator {tile}, so its kan has no time. "
            "Which discard came right after that kan?"
        ),
    }


def missing_indicators(seen: list[str], guesses: list[str], t: float) -> dict:
    """Ask for dora indicators that no view shows, with the log's guesses."""
    what = (
        "the dora indicator"
        if len(guesses) == 1 and not seen
        else "a dora indicator a kan revealed"
        if len(guesses) == 1
        else f"{len(guesses)} dora indicators"
    )
    return {
        "kind": "dora",
        "t": t,
        "tiles": seen,
        "guess": guesses,
        "text": (
            f"No view shows {what}; the log carries {' '.join(guesses)} as a guess. "
            "Enter all the dora indicators in order."
        ),
    }


# ---- questions: conflicts and the result -------------------------------------------


def conflict(
    culprit: Culprit | None, *, t: float, over: list[dict], call: Call | None = None
) -> dict:
    """Ask the one decision a conflict hinges on, with the control that answers it.

    Without a culprit the over-counted kinds and their sources are the question.
    """
    item: dict = {"kind": "conflict", "t": t, "over": over}
    if culprit is None:
        return {
            **item,
            "culprit": None,
            "text": (
                "No reconstruction of this hand is legal, and no single discard, call "
                "or answer explains why. The kinds below appear more than four times;"
                " mark the reading that is wrong."
            ),
        }
    item.update(
        culprit=culprit.kind,
        seat=culprit.seat,
        fact_seat=culprit.seat if culprit.kind == "fact" else None,
        fact_kind=culprit.fact,
    )
    if culprit.kind == "meld" and call is not None:
        return {
            **item,
            "t": call.t_first,
            "type": call.type,
            "tiles": call.tiles,
            "source": call.source,
            "tile": call.called_tile,
            "text": (
                f"This {call.type} cannot be: the hand never holds the tiles it "
                "takes. What is the meld?"
            ),
        }
    if culprit.kind == "result":
        return {
            **item,
            "text": (
                "The winning hand cannot be completed from what was read: a call or "
                "a discard of the winner is read wrong. Enter the hand the winner "
                "revealed."
            ),
        }
    if culprit.kind == "riichi" and culprit.t is not None:
        return {
            **item,
            "t": culprit.t,
            "text": (
                "After this riichi the hand would have to change. Which discard "
                "declared riichi?"
            ),
        }
    if culprit.kind == "fact":
        name = FACT_NAMES.get(culprit.fact or "", "saved")
        return {
            **item,
            "text": f"Your {name} answer contradicts the video. Check that answer.",
        }
    return {
        **item,
        "text": (
            "A call of this player cannot be: the hand never holds the tiles it "
            "takes. Check that call."
        ),
    }


def score_mismatch(sc: ScoreResult, *, same_payment: bool | None) -> str:
    """Prompt for a winning hand that does not score the site's han and fu."""
    found = ", ".join(y.name for y in sc.yaku)
    if not sc.ok:
        why = "it has no yaku" if sc.error == "no_yaku" else "it is not a winning hand"
        return (
            f"The reconstructed hand does not win: {why}. Enter the hand the winner "
            "revealed."
        )
    if same_payment:
        return (
            f"The reconstructed hand ({found}) pays what the site's han/fu pay, so "
            "the site's figures may be a slip. Confirm that the site is wrong, or "
            "enter the hand the winner revealed."
        )
    if same_payment is None:
        return (
            f"The reconstructed hand ({found}) does not score the site's value. Enter"
            " the hand the winner revealed."
        )
    return (
        f"The reconstructed hand ({found}) does not score the site's han/fu. Enter "
        "the hand the winner revealed, or confirm that the site is wrong."
    )


def uncertain_winning_tile(
    seat: str, t: float, concealed: list[str], candidates: list[str], win_tile: str
) -> dict:
    """Ask which of several tiles completing the site's score won."""
    return {
        "kind": "result",
        "seat": seat,
        "t": t,
        "tiles": concealed,
        "win_tile": win_tile,
        "candidates": candidates,
        "text": (
            "The reveal does not show the winning tile clearly, and other tiles "
            "complete the hand with the site's score. Which tile won?"
        ),
    }


def tenpai(seat: str, t: float, hand: list[str], *, tenpai: bool) -> dict:
    """Ask for the final hand of a seat whose tenpai disagrees with the site."""
    mine, site = ("tenpai", "noten") if tenpai else ("noten", "tenpai")
    return {
        "kind": "result",
        "seat": seat,
        "t": t,
        "tiles": hand,
        "tenpai": tenpai,
        "text": (
            f"The reconstruction has this player {mine}, the site {site}. Enter the "
            "player's final hand."
        ),
    }


def ura(seat: str, t: float) -> dict:
    """Ask for the ura indicators of a riichi win."""
    return {
        "kind": "ura",
        "seat": seat,
        "t": t,
        "text": "Enter the ura indicators revealed at the end of the hand.",
    }


# ---- questions: tiles nothing settles ----------------------------------------------


def unseen_draw(
    seat: str, j: int, t: float, tile: str, certificate: Certificate | None
) -> dict:
    """Ask for a draw that no frame shows and nothing later pins."""
    return {
        "kind": "draw",
        "seat": seat,
        "j": j,
        "t": t,
        "tile": tile,
        "runner_up": certificate.runner_up if certificate else None,
        "margin": certificate.margin if certificate else 0.0,
        "text": "No frame shows this draw; the log carries the rules' guess.",
    }


def changed_discard(
    key: tuple[str, int], t: float, observed: str, chosen: str, facts: list[dict]
) -> dict | None:
    """Question pond-contradicting repairs unless a reviewer confirmed the tile."""
    seat, j = key
    if not chosen or chosen == observed:
        return None
    if any(
        f["seat"] == seat
        and abs(float(f["t"]) - t) <= CONFIRMED_DISCARD_WINDOW
        and f["tile"] == chosen
        for f in facts
    ):
        return None
    return {
        "kind": "discard",
        "seat": seat,
        "j": j,
        "t": t,
        "tile": chosen,
        "observed": observed,
        "text": "The pond reading does not fit the hand; the log uses another tile.",
    }


def uncertain_discards(rows: list[dict], items: list[dict]) -> list[dict]:
    """Ask about competing pond choices, even when the solver keeps the raw reading."""
    asked = {
        (item.get("seat"), item.get("j"))
        for item in items
        if item.get("kind") == "discard"
    }
    return [
        {
            "kind": "discard",
            "seat": row["seat"],
            "j": row["turn"],
            "tile": row["value"],
            "runner_up": row.get("runner_up"),
            "margin": row["margin"],
            "t": next(
                (e["t"] for e in row.get("evidence", []) if e.get("t") is not None),
                None,
            ),
            "text": "Two tiles fit this discard about equally well.",
        }
        for row in rows
        if row.get("field") == "discard"
        and confidence_state(row.get("margin"), row.get("alternative_gap"))
        == "ambiguous"
        and not row.get("human")
        and (row.get("seat"), row.get("turn")) not in asked
    ]


def uncertain_tiles(rows: list[dict], items: list[dict]) -> dict | None:
    """Group covered tiles with close witnesses, without duplicating questions or facts.

    Neither image coverage nor an unfinished search establishes ambiguity.
    Preserve explicit Can't tell answers.
    """
    asked = {
        (item.get("seat"), item.get("j"))
        for item in items
        if item.get("kind") == "draw"
    }
    pending = [
        row
        for row in rows
        if row.get("field") in ("draw", "haipai")
        and confidence_state(row.get("margin"), row.get("alternative_gap"))
        == "ambiguous"
        and not row.get("human")
        and not row.get("lost")
        and not row.get("inferred_from")
        and (row.get("seat"), row.get("turn")) not in asked
    ]
    if not pending:
        return None
    return {
        "kind": "uncertain_tiles",
        "count": len(pending),
        "choices": [
            {
                "field": row["field"],
                "seat": row["seat"],
                "j": row["turn"],
                "value": row["value"],
                "runner_up": row.get("runner_up"),
                "margin": row["margin"],
                "t": next(
                    (e["t"] for e in row.get("evidence", []) if e.get("t") is not None),
                    None,
                ),
            }
            for row in pending
        ],
        "text": "Close alternatives remain for these tiles.",
    }
