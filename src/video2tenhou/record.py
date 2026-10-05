# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""PacificML score site (scoremj.com) as an authoritative result source.

The site is a public GraphQL API (`/api/graphql`, no login needed for reads).
Per game it stores every hand result: round/repeat (kyoku index, honba),
riichi sticks on the table, the score delta of each seat and the events
(RON/TSUMO with han/fu and actor/target, EXHAUSTIVE_DRAW with the tenpai
bitmask, RIICHI per declarer). That is exactly the "result" block of a
tenhou/6 log except for the yaku list and the winning hand, so it replaces
score-delta inference. It supplies all names, seats and hand metadata.

    uv run python -m video2tenhou.record --game 21938 21939

Game ids are visible in the site URLs (`/games/<id>`). The result types and parsing
need no HTTP client; only `gql` loads it.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass, field

from video2tenhou.logging_setup import command_logging

ENDPOINT = "https://scoremj.com/api/graphql"
SEATS = ("EAST", "SOUTH", "WEST", "NORTH")
SEAT_LETTER = {"EAST": "E", "SOUTH": "S", "WEST": "W", "NORTH": "N"}

Q_GAME = (
    "query ViewGame($gameId: Int!) { game(id: $gameId) { id createdAt tableNumber "
    "status\n  gameContainer { id name } gameRules { id name }\n  handResults { "
    "scoreDelta { EAST SOUTH WEST NORTH } riichiLeftover honba nextRound nextRepeat"
    " round repeat\n    events { type han fu tenpai actor target } }\n  players { "
    "id startingDirection user { username id } }\n  score { EAST SOUTH WEST NORTH }"
    " } }"
)

LOGGER = logging.getLogger("video2tenhou.record")


def gql(query: str, variables: dict, timeout: float = 30) -> dict:
    """Execute a read query against scoremj.

    HTTP errors propagate and GraphQL errors raise RuntimeError.
    """
    import httpx  # noqa: PLC0415  the engine imports the result types without HTTP

    response = httpx.post(
        ENDPOINT,
        json={"query": query, "variables": variables},
        headers={"User-Agent": "video2tenhou"},
        timeout=timeout,
        follow_redirects=True,
    )
    response.raise_for_status()
    out = response.json()
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["data"]


@dataclass
class HandResult:
    """Authoritative result of one hand, seats named by the hanchan's starting winds."""

    kyoku: int  # 0 = East 1 ... 7 = South 4 (site "round")
    honba: int  # site "honba" ("repeat" is the dealer-repeat count)
    sticks: int  # riichi sticks on the table at the start (site riichiLeftover)
    deltas: dict  # seat -> points, seats by STARTING wind (EAST = initial dealer)
    outcome: str  # "ron" | "tsumo" | "draw"
    winner: str | None = None  # starting-wind seat
    loser: str | None = None
    han: int | None = None
    fu: int | None = None
    tenpai: list = field(default_factory=list)  # seats tenpai at an exhaustive draw
    riichi: list = field(default_factory=list)  # seats that declared riichi

    @property
    def name(self) -> str:
        """Format the zero-based round and honba as a Japanese display label."""
        return f"{'東南'[self.kyoku // 4]}{self.kyoku % 4 + 1}局{self.honba}本場"


@dataclass
class Game:
    """Site record with starting-seat player identities and ordered hand results."""

    id: int
    # the site's seat name (EAST..NORTH, the wind of the hanchan's first hand) ->
    # username
    players: dict
    final: dict
    hands: list[HandResult]
    rules: str = ""
    container: str = ""
    created: str = ""


def _tenpai_seats(mask: int | None) -> list[str]:
    # bitmask in seat order E S W N (verified against score deltas: 3000 split among
    # tenpai players)
    if mask is None:
        return []
    return [s for i, s in enumerate(SEATS) if mask >> i & 1]


def parse_game(g: dict) -> Game:
    """Translate a scoremj game payload, preserving its starting-seat deltas."""
    hands = []
    for h in g["handResults"]:
        hr = HandResult(
            kyoku=h["round"],
            honba=h["honba"],
            sticks=h["riichiLeftover"],
            deltas=dict(h["scoreDelta"]),
            outcome="draw",
        )
        for e in h["events"]:
            t = e["type"]
            if t in ("RON", "TSUMO"):
                hr.outcome, hr.winner, hr.loser, hr.han, hr.fu = (
                    t.lower(),
                    e["actor"],
                    e["target"],
                    e["han"],
                    e["fu"],
                )
            elif t == "EXHAUSTIVE_DRAW":
                hr.tenpai = _tenpai_seats(e["tenpai"])
            elif t == "RIICHI":
                hr.riichi.append(e["actor"])
        hands.append(hr)
    return Game(
        id=g["id"],
        players={p["startingDirection"]: p["user"]["username"] for p in g["players"]},
        final=dict(g["score"]),
        hands=hands,
        rules=(g.get("gameRules") or {}).get("name", ""),
        container=(g.get("gameContainer") or {}).get("name", ""),
        created=g.get("createdAt", ""),
    )


def fetch_game(game_id: int) -> Game:
    """Fetch a game by numeric scoremj ID, propagating network failures."""
    return parse_game(gql(Q_GAME, {"gameId": game_id})["game"])


def to_dict(g: Game) -> dict:
    """JSON-ready cache representation, including each hand result as a mapping."""
    return asdict(g)


def from_dict(d: dict) -> Game:
    """Restore a cached game, including typed HandResult entries."""
    return Game(
        id=d["id"],
        players=d["players"],
        final=d["final"],
        hands=[HandResult(**h) for h in d["hands"]],
        rules=d.get("rules", ""),
        container=d.get("container", ""),
        created=d.get("created", ""),
    )


@command_logging
def main(argv: list[str] | None = None) -> None:
    """Print authoritative game and hand summaries for the requested scoremj IDs."""
    ap = argparse.ArgumentParser(description="print scoremj.com games")
    ap.add_argument("--game", type=int, nargs="+")
    a = ap.parse_args(argv)
    for g in (fetch_game(x) for x in a.game):
        LOGGER.info(
            "game %s (%s, %s) players %s final %s",
            g.id,
            g.container,
            g.rules,
            g.players,
            g.final,
        )
        for i, h in enumerate(g.hands):
            LOGGER.info(
                "  %2d %s sticks=%s %s %s%s %s/%s riichi=%s tenpai=%s %s",
                i,
                h.name,
                h.sticks,
                h.outcome,
                h.winner or "",
                "<-" + h.loser if h.loser else "",
                h.han or "",
                h.fu or "",
                h.riichi,
                h.tenpai,
                h.deltas,
            )


if __name__ == "__main__":
    main()
