"""PacificML score site (scoremj.com) as an authoritative result source.

The site is a public GraphQL API (`/api/graphql`, no login needed for reads).
Per game it stores every hand result: round/repeat (kyoku index, honba),
riichi sticks on the table, the score delta of each seat and the events
(RON/TSUMO with han/fu and actor/target, EXHAUSTIVE_DRAW with the tenpai
bitmask, RIICHI per declarer). That is exactly the "result" block of a
tenhou/6 log except for the yaku list and the winning hand, so it replaces
score-delta inference (`results.py`) whenever the game id is known and it
verifies the overlay reader.

    uv run python -m video2tenhou.record --game 21938 21939

Game ids are visible in the site URLs (`/games/<id>`); `walk_container` lists
the games of an event / week when the id is not known.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

ENDPOINT = "https://scoremj.com/api/graphql"
SEATS = ("EAST", "SOUTH", "WEST", "NORTH")
SEAT_LETTER = {"EAST": "E", "SOUTH": "S", "WEST": "W", "NORTH": "N"}

Q_GAME = """query ViewGame($gameId: Int!) { game(id: $gameId) { id createdAt tableNumber status
  gameContainer { id name } gameRules { id name }
  handResults { scoreDelta { EAST SOUTH WEST NORTH } riichiLeftover honba nextRound nextRepeat round repeat
    events { type han fu tenpai actor target } }
  players { id startingDirection user { username id } }
  score { EAST SOUTH WEST NORTH } } }"""

Q_CONTAINER = """query ViewGameContainer($gameContainerId: Int!, $withGames: Boolean!) {
  gameContainer(id: $gameContainerId) { id type name startsAt children { id name startsAt }
  games @include(if: $withGames) { id createdAt status tableNumber
    players { startingDirection user { username id } } score { EAST SOUTH WEST NORTH } } } }"""


def gql(query: str, variables: dict, timeout: float = 30) -> dict:
    """Execute a read query against scoremj; HTTP errors propagate and GraphQL errors raise RuntimeError."""
    req = urllib.request.Request(ENDPOINT, data=json.dumps({"query": query, "variables": variables}).encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "video2tenhou"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    if out.get("errors"):
        raise RuntimeError(out["errors"])
    return out["data"]


@dataclass
class HandResult:
    """Authoritative result of one hand; seat identifiers refer to the hanchan starting winds."""
    kyoku: int            # 0 = East 1 ... 7 = South 4 (site "round")
    honba: int            # site "honba" ("repeat" is the dealer-repeat count)
    sticks: int           # riichi sticks on the table at the start (site riichiLeftover)
    deltas: dict          # seat -> points, seats by STARTING wind (EAST = initial dealer)
    outcome: str          # "ron" | "tsumo" | "draw"
    winner: Optional[str] = None   # starting-wind seat
    loser: Optional[str] = None
    han: Optional[int] = None
    fu: Optional[int] = None
    tenpai: list = field(default_factory=list)   # seats tenpai at an exhaustive draw
    riichi: list = field(default_factory=list)   # seats that declared riichi

    @property
    def name(self) -> str:
        """Japanese round and honba label for display, using the zero-based kyoku index."""
        return f"{'東南'[self.kyoku // 4]}{self.kyoku % 4 + 1}局{self.honba}本場"


@dataclass
class Game:
    """Site record containing starting-seat player identities and ordered hand results."""
    id: int
    players: dict         # the site's seat name (EAST..NORTH, the wind of the hanchan's first hand) -> username
    final: dict
    hands: list[HandResult]
    rules: str = ""
    container: str = ""
    created: str = ""


def _tenpai_seats(mask: Optional[int]) -> list[str]:
    # bitmask in seat order E S W N (verified against score deltas: 3000 split among tenpai players)
    if mask is None:
        return []
    return [s for i, s in enumerate(SEATS) if mask >> i & 1]


def parse_game(g: dict) -> Game:
    """Translate a scoremj game payload into reconstruction inputs, preserving starting-seat deltas."""
    hands = []
    for h in g["handResults"]:
        hr = HandResult(kyoku=h["round"], honba=h["honba"], sticks=h["riichiLeftover"], deltas=dict(h["scoreDelta"]), outcome="draw")
        for e in h["events"]:
            t = e["type"]
            if t in ("RON", "TSUMO"):
                hr.outcome, hr.winner, hr.loser, hr.han, hr.fu = t.lower(), e["actor"], e["target"], e["han"], e["fu"]
            elif t == "EXHAUSTIVE_DRAW":
                hr.tenpai = _tenpai_seats(e["tenpai"])
            elif t == "RIICHI":
                hr.riichi.append(e["actor"])
        hands.append(hr)
    return Game(id=g["id"], players={p["startingDirection"]: p["user"]["username"] for p in g["players"]},
                final=dict(g["score"]), hands=hands, rules=(g.get("gameRules") or {}).get("name", ""),
                container=(g.get("gameContainer") or {}).get("name", ""), created=g.get("createdAt", ""))


def fetch_game(game_id: int) -> Game:
    """Fetch and normalize a game by its numeric scoremj ID; network failures propagate."""
    return parse_game(gql(Q_GAME, {"gameId": game_id})["game"])


def walk_container(container_id: int) -> list[dict]:
    """All games (site summaries) under a container, depth first."""
    c = gql(Q_CONTAINER, {"gameContainerId": container_id, "withGames": True})["gameContainer"]
    games = [dict(g, container=c["name"]) for g in c["games"]]
    for ch in c["children"]:
        games += walk_container(ch["id"])
    return games


def to_dict(g: Game) -> dict:
    """JSON-ready cache representation, including each hand result as a mapping."""
    return {**g.__dict__, "hands": [h.__dict__ for h in g.hands]}


def from_dict(d: dict) -> Game:
    """Restore a cached game created by to_dict(), including typed HandResult entries."""
    return Game(**{**d, "hands": [HandResult(**h) for h in d["hands"]]})


def main(argv=None):
    """Print authoritative game and hand summaries for the requested scoremj IDs."""
    import argparse
    ap = argparse.ArgumentParser(description="print scoremj.com games")
    ap.add_argument("--game", type=int, nargs="+")
    a = ap.parse_args(argv)
    for g in (fetch_game(x) for x in a.game):
        print(f"game {g.id} ({g.container}, {g.rules}) players {g.players} final {g.final}")
        for i, h in enumerate(g.hands):
            print(f"  {i:2d} {h.name} sticks={h.sticks} {h.outcome} {h.winner or ''}{'<-' + h.loser if h.loser else ''} "
                  f"{h.han or ''}/{h.fu or ''} riichi={h.riichi} tenpai={h.tenpai} {h.deltas}")


if __name__ == "__main__":
    main()
