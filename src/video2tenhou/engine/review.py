# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Review facts as reconstruction constraints, and the confidence of every decision.

The human facts that answer review questions (labels/<video>/facts.jsonl) are
normalized per hand; the next decode applies them as constraints. A confidence row
records every decision of the log with its margin (and a draw's runner-up), whether
a human fixed it and whether nothing covered it (`lost`, DESIGN.md section 6).
Unfinished searches are unresolvable and do not ask for retries. The questions
themselves are written by questions.py.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .confidence import confidence_state
from .hand import corner_of

if TYPE_CHECKING:
    from pathlib import Path

    from video2tenhou.record import HandResult

    from .confidence import Certificate
    from .melds import Call
    from .solver import HandModel, Solution
    from .turns import Turn


@dataclass(frozen=True, kw_only=True)
class ReviewContext:
    """Evidence and human decisions accompanying solver confidence rows."""

    turns: list[Turn]
    calls: list[Call]
    inds: list[dict]
    entry: dict
    facts: dict
    lost_keys: set
    t0: float


def draws_to_reread(
    sol: Solution, model: HandModel | None = None
) -> list[tuple[str, int]]:
    """Select ambiguous draws of a certified solution for video acquisition.

    An unresolvable certificate does not by itself justify re-reading the video:
    proof bounds certify choices; close witnesses establish review questions. An
    open-kan replacement without direct draw evidence also needs a view around the
    kan: a certificate inferred from earlier hand states cannot compensate for that
    acquisition gap. Self-kans have two draws and require separate evidence mapping.
    """
    selected = [
        (seat, j)
        for (field, seat, j), certificate in sol.certificates.items()
        if field == "draw"
        and certificate.state == "ambiguous"
        and (seat, j) not in sol.draw_sources
    ]
    if model is not None:
        observed = {(ev.seat, ev.j) for ev in model.draw_ev}
        for seat, turns in model.turns.items():
            for turn in turns:
                key = (seat, turn.j)
                if (
                    turn.kind == "kan"
                    and turn.kan == "daiminkan"
                    and not turn.two_draws
                    and key not in model.facts.draws
                    and key not in observed
                    and key not in selected
                ):
                    selected.append(key)
    return selected


# ------------------------------------------------------------------------------
# facts (labels/<video>/facts.jsonl) -> constraints for one hand
# ------------------------------------------------------------------------------


def load_facts(labels_dir: Path) -> list[dict]:
    """Read review facts; a new video's missing journal means no facts."""
    p = labels_dir / "facts.jsonl"
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def hand_facts(all_facts: list[dict], entry: dict) -> list[dict]:
    """Return a hand's reconstruction inputs: its facts except question dismissals.

    A `dismiss` fact is review state (a question the reviewer closed without a tile
    answer), so it never constrains or invalidates a reconstruction.
    """
    return [
        fact
        for fact in all_facts
        if fact.get("kind") != "dismiss"
        and all(fact.get(key) == entry[key] for key in ("game", "kyoku", "honba"))
    ]


def decode_context(entry: dict, result: HandResult, all_facts: list[dict]) -> dict:
    """Bind a reconstruction to its seat map, authoritative result and human inputs."""
    return {
        "entry": entry,
        "result": dataclasses.asdict(result),
        "facts": hand_facts(all_facts, entry),
    }


def facts_for_hand(all_facts: list[dict], entry: dict) -> dict:
    """Normalize a hand's facts, keeping soft hand evidence apart from constraints.

    Hand annotations may explicitly set boolean ``soft``. Unsourced review answers
    default to confirmed; imported annotations carrying a ``source`` must declare their
    strength instead of deriving trust from its name.
    """
    out: dict = {
        "haipai": [],
        "final_hand": [],
        "draw": [],
        "discard": [],
        "missing_discard": [],
        "meld_remove": [],
        "meld": [],
        "ura": [],
        "dora": [],
        "riichi_turn": [],
        "kan_time": [],
        "lost": [],
        "lost_haipai": [],
        "lost_dora": False,
        "site_score": None,
    }
    for f in hand_facts(all_facts, entry):
        # the corner is the identity: a fact keeps its meaning whatever the seat letters
        # were when it was saved
        seat = entry["corner_wind"].get(f.get("corner") or "", None)
        kind = f.get("kind")
        if kind in ("haipai", "final_hand"):
            _apply_concealed_fact(out, f, seat)
        elif kind in (
            "draw",
            "discard",
            "meld",
            "meld_remove",
            "missing_discard",
            "riichi_turn",
            "kan_time",
        ):
            _apply_turn_fact(out, f, seat)
        else:
            _apply_result_fact(out, f, seat)
    return out


def _apply_concealed_fact(out: dict, f: dict, seat: str | None) -> None:
    """Validate explicit annotation strength before adding concealed tiles."""
    kind = f.get("kind")
    if (
        kind in ("haipai", "final_hand")
        and seat
        and f.get("tiles")
        and "?" not in f["tiles"]
    ):
        if "source" in f and "soft" not in f:
            raise ValueError(
                f"{kind} annotation from {f['source']!r} must explicitly "
                "set soft=true for evidence or soft=false after human "
                "confirmation. Review the annotation before rebuilding."
            )
        soft = f.get("soft", False)
        if not isinstance(soft, bool):
            raise ValueError(f"{kind} annotation soft must be a boolean.")
        out[kind].append({"seat": seat, "tiles": f["tiles"], "soft": soft})


def _apply_turn_fact(out: dict, f: dict, seat: str | None) -> None:
    """Normalize draw, discard, meld and timing constraints."""
    kind = f.get("kind")
    if kind == "draw" and seat and f.get("tile"):
        t = f.get("t_discard") if f.get("t_discard") is not None else f.get("t")
        out["draw"].append({"seat": seat, "t": t, "tile": f["tile"], "j": f.get("j")})
    elif kind == "discard" and seat and f.get("tile"):
        out["discard"].append({"seat": seat, "t": f.get("t"), "tile": f["tile"]})
    elif (
        kind == "meld"
        and seat
        and f.get("t") is not None
        and f.get("tiles")
        and f.get("type")
    ):
        out["meld"].append(
            {
                "seat": seat,
                "t": float(f["t"]),
                "type": f["type"],
                "tiles": f["tiles"],
                "called_pos": f.get("called_pos"),
                "source": f.get("source"),
            }
        )
    elif kind == "meld_remove" and seat and f.get("t") is not None:
        out["meld_remove"].append(
            {"seat": seat, "t": float(f["t"]), "type": f.get("type")}
        )
    elif (
        kind == "missing_discard" and seat and f.get("tile") and f.get("t") is not None
    ):
        out["missing_discard"].append(
            {"seat": seat, "t": float(f["t"]), "tile": f["tile"]}
        )
    elif kind == "riichi_turn" and seat and f.get("t") is not None:
        out["riichi_turn"].append({"seat": seat, "t": float(f["t"])})
    elif kind == "kan_time" and f.get("t") is not None:
        # the seat is optional
        out["kan_time"].append({"seat": seat, "t": float(f["t"])})


def _apply_result_fact(out: dict, f: dict, seat: str | None) -> None:
    """Normalize result corrections, indicators and unresolved decisions."""
    kind = f.get("kind")
    if kind == "result":
        raise ValueError(
            "Unsupported result annotation. Save ura indicators as "
            "kind='ura' with tiles, or a confirmed concealed hand as "
            "kind='final_hand'. The saved journal was not changed."
        )
    if kind == "ura" and f.get("tiles") is not None:
        out["ura"] = f["tiles"]
    elif kind == "dora" and f.get("tiles"):
        out["dora"] = f["tiles"]
    elif kind == "riichi" and f.get("seats") is not None:
        out["riichi"] = f["seats"]
    elif kind == "lost" and f.get("field") == "dora":
        # Can't tell: the indicator(s) no view shows stay the rules' guess
        out["lost_dora"] = True
    elif kind == "lost" and f.get("field") == "haipai" and seat:
        out["lost_haipai"].append(seat)
    elif kind == "lost" and seat:
        out["lost"].append({"seat": seat, "t": f.get("t"), "j": f.get("j")})
    elif kind == "site_wrong" and f.get("han") is not None and f.get("fu") is not None:
        # the reviewer's han/fu replace the site's
        out["site_score"] = {"han": int(f["han"]), "fu": int(f["fu"])}


def turn_key(model: HandModel, f: dict, tol: float) -> tuple[str, int] | None:
    """Return (seat, j) of the draw a draw or Can't-tell fact is about.

    The turn whose discard is nearest the fact's time (within tol), else the turn
    index the fact carries: the tsumo winner's final draw has no discard.
    """
    seat = f["seat"]
    mine = model.turns.get(seat, [])
    if f.get("t") is not None and mine:
        st = min(mine, key=lambda x: abs(x.t_discard - f["t"]))
        if abs(st.t_discard - f["t"]) <= tol and st.kind in ("draw", "kan"):
            return (seat, st.j)
    if f.get("j") is not None and int(f["j"]) in model.draw_turns(seat):
        return (seat, int(f["j"]))
    return None


# ------------------------------------------------------------------------------
# items and confidence rows
# ------------------------------------------------------------------------------


def _draw_span(
    model: HandModel, turns: list[Turn], seat: str, j: int, t0: float
) -> tuple[float, float]:
    """Return the span from the seat's previous turn to the discard of turn j.

    The tsumo winner's last draw spans to its reveal.
    """
    st = model.turns[seat][j] if j < len(model.turns[seat]) else None
    t_end = st.t_discard if st else (turns[-1].t + 5.0 if turns else t0)
    before = [t.t for t in turns if t.t < t_end - 0.1]
    return (before[-1] if before else t0), t_end


def _covered(model: HandModel, seat: str, j: int) -> bool:
    """Check whether a draw has direct or complete neighbouring hand evidence.

    A row short of the hand by a hidden tile does not cover it: the hidden one may be
    the draw.
    """
    if any(ev.seat == seat and ev.j == j for ev in model.draw_ev):
        return True
    return any(
        ev.seat == seat and not ev.subset and not ev.hidden and (ev.j in (j - 1, j))
        for ev in model.hand_ev
    )


def unseen_draw(model: HandModel, sol: Solution, seat: str, j: int) -> bool:
    """Identify a draw that neither an observation nor certification pins.

    Section 6, `lost`: its certificate is ambiguous, unresolvable or missing.
    """
    certificate = sol.certificates.get(("draw", seat, j))
    return (
        (seat, j) not in sol.draw_sources
        and (certificate is None or certificate.state != "resolved")
        and not _covered(model, seat, j)
    )


def _certificate_fields(certificate: Certificate | None) -> dict:
    """Confidence-row fields of a certificate; an uncertified decision is unmeasured."""
    if certificate is None:
        return {"margin": None, "alternative_gap": None}
    return {"margin": certificate.margin, "alternative_gap": certificate.gap}


def confidence_rows(
    model: HandModel, sol: Solution, *, context: ReviewContext
) -> list[dict]:
    """Build one confidence row per decision of the log (section 5).

    Draws, discards, haipai, calls and indicators.
    """
    turns, calls, inds, entry, facts, lost_keys, t0 = (
        context.turns,
        context.calls,
        context.inds,
        context.entry,
        context.facts,
        context.lost_keys,
        context.t0,
    )
    rows: list[dict] = []
    for (s, j), tile in sorted(sol.draws.items()):
        a, b = _draw_span(model, turns, s, j, t0)
        corner = corner_of(entry, s)
        certificate = sol.certificates.get(("draw", s, j))
        rows.append(
            {
                "seat": s,
                "turn": j,
                "field": "draw",
                "value": tile,
                **_certificate_fields(certificate),
                "runner_up": certificate.runner_up if certificate else None,
                "inferred_from": sol.draw_sources.get((s, j)),
                "human": (s, j) in model.facts.draws,
                "lost": (s, j) in lost_keys or unseen_draw(model, sol, s, j),
                "evidence": [{"region": f"pond:{corner}", "t": b}]
                if sol.draw_sources.get((s, j)) == "discard"
                else [{"region": f"meld:{corner}", "t": b}]
                if sol.draw_sources.get((s, j)) == "ankan"
                else [
                    {"region": f"hand:{corner}", "t": a},
                    {"region": f"hand:{corner}", "t": b},
                ],
            }
        )
    fact_discards = {
        (f["seat"], round(float(f["t"] or 0)))
        for f in facts.get("discard", []) + facts.get("missing_discard", [])
    }
    for t in turns:
        if t.slot is None:
            continue
        st = next((x for x in model.turns[t.seat] if x.t_discard == t.t), None)
        # Repair fixes its selected identity in the model before the final
        # solve; that solution then has no variable-discard override. Keep
        # confidence values aligned with the actual exported reconstruction.
        chosen = sol.discards.get((t.seat, st.j), st.discard) if st else None
        certificate = sol.certificates.get(("discard", t.seat, st.j)) if st else None
        # a virtual discard was never seen in the pond, but its tile is the called tile
        # the meld camera shows
        caller = t.slot.t_removed if t.virtual else None
        if t.virtual:
            calling_seat = next((c.seat for c in calls if c.t_first == caller), t.seat)
            region = f"meld:{corner_of(entry, calling_seat)}"
        else:
            region = f"pond:{corner_of(entry, t.seat)}"
        rows.append(
            {
                "seat": t.seat,
                "turn": st.j if st else None,
                "field": "discard",
                "value": chosen or t.slot.tile,
                **_certificate_fields(certificate),
                "runner_up": certificate.runner_up if certificate else None,
                "conf": round(t.slot.conf, 3),
                "human": (t.seat, round(t.t)) in fact_discards,
                "lost": False,
                "virtual": t.virtual,
                "evidence": [{"region": region, "t": caller if t.virtual else t.t}],
            }
        )
    for s, tiles in sorted(sol.haipai.items()):
        rows.append(
            {
                "seat": s,
                "turn": -1,
                "field": "haipai",
                "value": tiles,
                **_certificate_fields(sol.certificates.get(("haipai", s, -1))),
                "human": s in model.facts.haipai,
                "lost": s in facts.get("lost_haipai", []),
                "evidence": [{"region": f"hand:{corner_of(entry, s)}", "t": t0}],
            }
        )
    rows.extend(
        {
            "seat": c.seat,
            "turn": None,
            "field": "call",
            "value": f"{c.type} {''.join(c.tiles)}",
            "margin": None,
            "seen": c.seen,
            "human": c.human,
            "lost": False,
            "evidence": [
                {"region": f"meld:{corner_of(entry, c.seat)}", "t": c.t_first}
            ],
        }
        for c in calls
    )
    rows.extend(
        {
            "seat": None,
            "turn": None,
            "field": "dora",
            "value": v["tile"],
            "margin": None,
            "seen": v.get("seen"),
            "human": bool(v.get("human")),
            "lost": bool(v.get("lost")),
            "evidence": [{"region": v.get("region") or "", "t": v.get("t_first")}]
            if v.get("t_first") is not None
            else [],
        }
        for v in inds
    )
    for row in rows:
        row["state"] = confidence_state(row.get("margin"), row.get("alternative_gap"))
    return rows
