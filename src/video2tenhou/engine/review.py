"""Review items and confidence rows (DESIGN.md 4.8 `review.py`), and the human facts that answer them.

An item is a question for the tool about one decision the program could not settle and whose answer settles
the most: its kind, the seat and time, the best guess, the candidates with their costs, and where to look
(evidence: region and time). A confidence row records every decision of the log
with its margin (and a draw's runner-up), whether a human fixed it and whether
nothing covered it (`lost`). A tile nothing covered is a question until the reviewer supplies it or says Can't
tell (DESIGN.md section 6). Covered choices with low certified margins are grouped
in one uncertainty item instead of silently marking the hand complete. Facts are
the tool's answers; the next decode applies them as constraints.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .hand import corner_of
from .melds import Call
from .solver import HandModel, Solution
from .turns import Turn

MARGIN_REVIEW = 0.5


def low_margin(margin: Optional[float]) -> bool:
    """Whether a measured choice remains uncertain, including the review boundary.

    Objectives use fixed-point costs but are subtracted as floats. The tiny
    tolerance prevents a mathematical margin of 0.5 rounding just above it.
    Missing margins are not measured decisions (for example call turns).
    """
    return margin is not None and margin <= MARGIN_REVIEW + 1e-9


def draws_to_reread(sol: Solution, model: Optional[HandModel] = None) -> list[tuple[str, int]]:
    """Select draws for video acquisition, keeping search timeouts reviewable separately.

    A distant feasible alternative with a weak proof bound is search uncertainty;
    it does not by itself justify re-reading the video. A search without a
    candidate still requests evidence unless its lower bound already certifies
    the choice. Certified margins alone control review. An open-kan replacement
    without direct draw evidence also needs a view around the kan: a certificate
    inferred from earlier hand states cannot compensate for that acquisition gap.
    Self-kans have two draws and require separate evidence mapping.
    """
    selected = [key for key, margin in sol.margins.items()
                if low_margin(margin) and low_margin(sol.alternative_gaps.get(key, margin))]
    if model is not None:
        observed = {(ev.seat, ev.j) for ev in model.draw_ev}
        for seat, turns in model.turns.items():
            for turn in turns:
                key = (seat, turn.j)
                if (turn.kind == "kan" and turn.kan == "daiminkan" and not turn.two_draws
                        and key not in model.facts.draws and key not in observed and key not in selected):
                    selected.append(key)
    return selected


# ---------------------------------------------------------------------------------------------------------
# facts (labels/<video>/facts.jsonl) -> constraints for one hand
# ---------------------------------------------------------------------------------------------------------

def load_facts(labels_dir: Path) -> list[dict]:
    """Read the append-only review journal; a new video's missing journal means no facts."""
    p = labels_dir / "facts.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def facts_for_hand(all_facts: list[dict], entry: dict) -> dict:
    """Facts of this hand keyed for decode_hand: haipai / final_hand (seat, tiles), draw (seat, t, tile), ura, dora."""
    out: dict = {"haipai": [], "final_hand": [], "draw": [], "discard": [], "missing_discard": [], "meld_remove": [], "meld": [],
                 "ura": [], "dora": [], "riichi_turn": [], "kan_time": [], "lost": [], "lost_haipai": [], "lost_dora": False,
                 "site_score": None}
    for f in all_facts:
        if f.get("game") != entry["game"] or f.get("kyoku") != entry["kyoku"] or f.get("honba") != entry["honba"]:
            continue
        # the corner is the identity: a fact keeps its meaning whatever the seat letters were when it was saved
        seat = entry["corner_wind"].get(f.get("corner") or "", None)
        kind = f.get("kind")
        if kind in ("haipai", "final_hand") and seat and f.get("tiles") and "?" not in f["tiles"]:
            out[kind].append({"seat": seat, "tiles": f["tiles"], "soft": str(f.get("source", "")).startswith("legacy")})
        elif kind == "draw" and seat and f.get("tile"):
            t = f.get("t_discard") if f.get("t_discard") is not None else f.get("t")
            out["draw"].append({"seat": seat, "t": t, "tile": f["tile"], "j": f.get("j")})
        elif kind == "discard" and seat and f.get("tile"):
            out["discard"].append({"seat": seat, "t": f.get("t"), "tile": f["tile"]})
        elif kind == "meld" and seat and f.get("t") is not None and f.get("tiles") and f.get("type"):
            out["meld"].append({"seat": seat, "t": float(f["t"]), "type": f["type"], "tiles": f["tiles"],
                                "called_pos": f.get("called_pos"), "source": f.get("source")})
        elif kind == "meld_remove" and seat and f.get("t") is not None:
            out["meld_remove"].append({"seat": seat, "t": float(f["t"]), "type": f.get("type")})
        elif kind == "missing_discard" and seat and f.get("tile") and f.get("t") is not None:
            out["missing_discard"].append({"seat": seat, "t": float(f["t"]), "tile": f["tile"]})
        elif kind == "result" and f.get("ura"):
            out["ura"] = f["ura"]
        elif kind == "ura" and f.get("tiles") is not None:
            out["ura"] = f["tiles"]
        elif kind == "dora" and f.get("tiles"):
            out["dora"] = f["tiles"]
        elif kind == "riichi" and f.get("seats") is not None:
            out["riichi"] = f["seats"]
        elif kind == "lost" and f.get("field") == "dora":
            out["lost_dora"] = True                 # Can't tell: the indicator(s) no view shows stay the rules' guess
        elif kind == "lost" and f.get("field") == "haipai" and seat:
            out["lost_haipai"].append(seat)
        elif kind == "lost" and seat:
            out["lost"].append({"seat": seat, "t": f.get("t"), "j": f.get("j")})
        elif kind == "riichi_turn" and seat and f.get("t") is not None:
            out["riichi_turn"].append({"seat": seat, "t": float(f["t"])})
        elif kind == "site_wrong" and f.get("han") is not None and f.get("fu") is not None:
            out["site_score"] = {"han": int(f["han"]), "fu": int(f["fu"])}    # the reviewer's han/fu replace the site's
        elif kind == "kan_time" and f.get("t") is not None:
            out["kan_time"].append({"seat": seat, "t": float(f["t"])})       # the seat is optional
    return out


def turn_key(model: HandModel, f: dict, tol: float) -> Optional[tuple[str, int]]:
    """(seat, j) of the draw a draw / lost fact is about: the turn whose discard is nearest the fact's time
    (within tol), else the turn index the fact carries (the tsumo winner's final draw has no discard, so its
    index is its only key)."""
    seat = f["seat"]
    mine = model.turns.get(seat, [])
    if f.get("t") is not None and mine:
        st = min(mine, key=lambda x: abs(x.t_discard - f["t"]))
        if abs(st.t_discard - f["t"]) <= tol and st.kind in ("draw", "kan"):
            return (seat, st.j)
    if f.get("j") is not None and int(f["j"]) in model._draw_turns(seat):
        return (seat, int(f["j"]))
    return None


# ---------------------------------------------------------------------------------------------------------
# items and confidence rows
# ---------------------------------------------------------------------------------------------------------

def _draw_span(model: HandModel, turns: list[Turn], seat: str, j: int, t0: float) -> tuple[float, float]:
    """From the seat's previous turn to the discard of turn j (the tsumo winner's last draw: to its reveal)."""
    st = model.turns[seat][j] if j < len(model.turns[seat]) else None
    t_end = st.t_discard if st else (turns[-1].t + 5.0 if turns else t0)
    before = [t.t for t in turns if t.t < t_end - 0.1]
    return (before[-1] if before else t0), t_end


def _covered(model: HandModel, seat: str, j: int) -> bool:
    """Is a draw covered by any evidence: a direct observation, or a full row of the hand next to it (a row short of
    the hand by a hidden tile is not: the hidden one may be the draw)?"""
    if any(ev.seat == seat and ev.j == j for ev in model.draw_ev):
        return True
    return any(ev.seat == seat and not ev.subset and not ev.hidden and (ev.j in (j - 1, j)) for ev in model.hand_ev)


def unseen_draw(model: HandModel, sol: Solution, seat: str, j: int) -> bool:
    """A draw nothing covers (section 6, `lost`): the solver's margin is low and no reading shows it."""
    margin = sol.margins.get((seat, j))
    return low_margin(margin) and not _covered(model, seat, j)


# Individual questions first; covered but uncertified tile choices are grouped
# after them, so a hand cannot look complete merely because images cover it.
PRIORITY = ["ura", "conflict", "result", "call", "order", "riichi", "dora", "kan", "draw"]


def ranked(items: list[dict]) -> list[dict]:
    """The questions of a hand in the order they settle it: the ura first, then a conflict, a result, a call."""
    return sorted(items, key=lambda it: PRIORITY.index(it["kind"]) if it["kind"] in PRIORITY else len(PRIORITY))


def uncertain_tiles(rows: list[dict], items: list[dict]) -> Optional[dict]:
    """Group unresolved covered tiles without duplicating questions or answered facts.

    Image coverage alone does not establish identity. Keep these alternatives
    visible in completion status, while preserving explicit Can't tell answers.
    """
    asked = {(item.get("seat"), item.get("j")) for item in items if item.get("kind") in ("draw", "lost", "haipai")}
    pending = [row for row in rows if row.get("field") in ("draw", "haipai") and low_margin(row.get("margin"))
               and not row.get("human") and not row.get("lost") and (row.get("seat"), row.get("turn")) not in asked]
    if not pending:
        return None
    return {"kind": "uncertain_tiles", "count": len(pending),
            "choices": [{"field": row["field"], "seat": row["seat"], "j": row["turn"], "value": row["value"],
                         "runner_up": row.get("runner_up"), "margin": row["margin"],
                         "t": next((e["t"] for e in row.get("evidence", []) if e.get("t") is not None), None)} for row in pending],
            "text": f"{len(pending)} tile choices remain uncertain after reconstruction. "
                    "Inspect these alternatives or rebuild the hand; the log remains provisional."}


def changed_discard(seat: str, j: int, t: float, observed: str, chosen: str,
                    facts: list[dict]) -> Optional[dict]:
    """Ask about a solver repair that contradicts the pond, unless a reviewer fixed it."""
    if not chosen or chosen == observed:
        return None
    if any(f["seat"] == seat and abs(float(f["t"]) - t) <= 3 and f["tile"] == chosen for f in facts):
        return None
    return {"kind": "discard", "seat": seat, "j": j, "t": t, "tile": chosen, "observed": observed,
            "text": f"The pond read {observed}, but the reconstruction uses {chosen}. Check this discard before accepting the log."}


def uncertain_discards(rows: list[dict], items: list[dict]) -> list[dict]:
    """Ask about uncertified pond choices, even when the solver keeps the raw reading."""
    asked = {(item.get("seat"), item.get("j")) for item in items if item.get("kind") == "discard"}
    return [{"kind": "discard", "seat": row["seat"], "j": row["turn"], "tile": row["value"],
             "runner_up": row.get("runner_up"), "margin": row["margin"],
             "t": next((e["t"] for e in row.get("evidence", []) if e.get("t") is not None), None),
             "text": "The reconstruction could not certify this discard's identity. Check it in the pond."}
            for row in rows if row.get("field") == "discard" and low_margin(row.get("margin"))
            and not row.get("human") and (row.get("seat"), row.get("turn")) not in asked]


def confidence_rows(model: HandModel, sol: Solution, turns: list[Turn], calls: list[Call], inds: list[dict], entry: dict,
                    facts: dict, lost_keys: set, t0: float) -> list[dict]:
    """One row per decision of the log (section 5): draws, discards, haipai, calls, indicators."""
    rows = []
    for (s, j), tile in sorted(sol.draws.items()):
        a, b = _draw_span(model, turns, s, j, t0)
        corner = corner_of(entry, s)
        margin = sol.margins.get((s, j))
        rows.append({"seat": s, "turn": j, "field": "draw", "value": tile, "margin": margin,
                     "alternative_gap": sol.alternative_gaps.get((s, j)),
                     "runner_up": sol.runner_up.get((s, j)), "human": (s, j) in model.facts.draws,
                     "lost": (s, j) in lost_keys or unseen_draw(model, sol, s, j),
                     "evidence": [{"region": f"hand:{corner}", "t": a}, {"region": f"hand:{corner}", "t": b}]})
    fact_discards = {(f["seat"], round(float(f["t"] or 0)))
                     for f in facts.get("discard", []) + facts.get("missing_discard", [])}
    for t in turns:
        if t.slot is None:
            continue
        st = next((x for x in model.turns[t.seat] if x.t_discard == t.t), None)
        # Repair fixes its selected identity in the model before the final
        # solve; that solution then has no variable-discard override. Keep
        # confidence values aligned with the actual exported reconstruction.
        chosen = sol.discards.get((t.seat, st.j), st.discard) if st else None
        # a virtual discard was never seen in the pond, but its tile is the called tile the meld camera shows
        caller = t.slot.t_removed if t.virtual else None
        region = f"meld:{corner_of(entry, next((c.seat for c in calls if c.t_first == caller), t.seat))}" if t.virtual             else f"pond:{corner_of(entry, t.seat)}"
        rows.append({"seat": t.seat, "turn": st.j if st else None, "field": "discard", "value": chosen or t.slot.tile,
                     "margin": sol.discard_margins.get((t.seat, st.j)) if st else None,
                     "runner_up": sol.discard_runner_up.get((t.seat, st.j)) if st else None,
                     "conf": round(t.slot.conf, 3), "human": (t.seat, round(t.t)) in fact_discards,
                     "lost": False, "virtual": t.virtual, "evidence": [{"region": region, "t": caller if t.virtual else t.t}]})
    for s, tiles in sorted(sol.haipai.items()):
        margin = sol.haipai_margins.get(s)
        rows.append({"seat": s, "turn": -1, "field": "haipai", "value": tiles, "margin": margin,
                     "human": s in model.facts.haipai, "lost": s in facts.get("lost_haipai", []),
                     "evidence": [{"region": f"hand:{corner_of(entry, s)}", "t": t0}]})
    for c in calls:
        rows.append({"seat": c.seat, "turn": None, "field": "call", "value": f"{c.type} {''.join(c.tiles)}", "margin": None,
                     "seen": c.seen, "human": c.human, "lost": False,
                     "evidence": [{"region": f"meld:{corner_of(entry, c.seat)}", "t": c.t_first}]})
    for v in inds:
        rows.append({"seat": None, "turn": None, "field": "dora", "value": v["tile"], "margin": None, "seen": v.get("seen"),
                     "human": bool(v.get("human")), "lost": bool(v.get("lost")),
                     "evidence": [{"region": v.get("region") or "", "t": v.get("t_first")}] if v.get("t_first") is not None else []})
    return rows
