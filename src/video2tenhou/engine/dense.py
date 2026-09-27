"""When unclear, look again more closely (DESIGN.md 4.8): targeted dense reads at 5 fps of the place and the
moment that decide an open question, before a human is asked.

- the discard a call took: the three other ponds over the seconds before the meld appeared;
- a turn the merge had to skip: that player's pond over the gap;
- a draw with a close feasible alternative, or no alternative found before timeout: its hand row from its
  previous discard to the next player's discard, with still runs providing views before the draw, just
  after it, or after the discard;
- an unconfirmed open-kan replacement without direct draw evidence: the same
  bounded hand window, even when earlier hand counts imply a confident guess;
- a riichi no turned tile was read for: the pond at the moment each of the seat's tiles was laid.

Video-reading entrypoints take `models` = (detector, classifier, video path, calibration) and cache reads under
work/<video>/dense/ (read.dense_reads).
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

from ..read import dense_pond_reads, dense_reads
from . import rules
from .hand import corner_of
from .ponds import PondSlot, insert_slot, tail_runs
from .solver import hand_evidence
from .turns import Turn


def _slot(i: int, b: dict, p: np.ndarray) -> PondSlot:
    return PondSlot(i, b["row"], b["col"], p, b["t_first"], (b["t_first"] - 1.0, b["t_first"]), b["t_last"], b["n"], 0.0,
                    xyxy=tuple(b["xyxy"]))


def taken_discards(caller: str, lo: float, hi: float, logs: dict[str, list[PondSlot]], entry: dict, models, work_dir,
                   problems: list[str]) -> list[tuple[str, dict]]:
    """The tiles laid in the three other ponds over [lo, hi] that vanished again before hi: the candidates for
    the discard a call of `caller` took (a called tile rests in the pond for a second or two, often while the
    calm reads see nothing). Returns [(discarder, run)] with run = {t_first, t_last, n, row, col, xyxy, p, tile}."""
    det, clf, video_path, cal = models
    others = [s for s in rules.SEATS if s != caller]
    corners = [corner_of(entry, s) for s in others]
    try:
        reads = dense_pond_reads(video_path, cal, work_dir, det, clf, lo, hi, corners)
    except Exception as ex:  # noqa: BLE001
        problems.append(f"dense pond read for a call of {caller} at {hi:.0f}s failed: {ex}")
        return []
    out = []
    for s, corner in zip(others, corners):
        out += [(s, run) for run in tail_runs(logs[s], reads[corner]) if run["gone"] and run["tile"] not in ("X", "none")]
    return out


def place_taken(seat: str, run: dict, logs: dict[str, list[PondSlot]]) -> PondSlot:
    """A called tile a dense read found enters its pond's log, by its time."""
    sl = _slot(7000 + len(logs[seat]), run, np.asarray(run["p"]))
    sl.t_removed = run["t_last"] + 0.2
    insert_slot(logs[seat], sl)
    return sl


def skipped_turns(order_problems: list[str], turns: list[Turn], logs: dict[str, list[PondSlot]], entry: dict, models,
                  work_dir, t0: float, t1: float, problems: list[str]) -> int:
    """A seat that "skipped" a turn: its discard was probably never in a calm frame. Read its pond densely over
    the gap and add the first new tile found. Returns how many discards were added."""
    det, clf, video_path, cal = models
    added = 0
    for p in order_problems:
        m = re.match(r"turn (\d+): no discard of ([ESWN])", p)
        if not m:
            continue
        i, seat = int(m.group(1)), m.group(2)
        t_a = turns[i - 1].t if 0 < i <= len(turns) else t0
        t_b = turns[i].t if i < len(turns) else t1
        if t_b - t_a > 90 or t_b <= t_a:
            continue
        corner = corner_of(entry, seat)
        try:
            reads = dense_reads(video_path, cal, work_dir, det, clf, t_a - 1.0, t_b + 1.0, [f"pond:{corner}"])
        except Exception as ex:  # noqa: BLE001
            problems.append(f"dense read for the skipped turn of {seat} failed: {ex}")
            continue
        b = next((b for b in tail_runs(logs[seat], reads[f"pond:{corner}"]) if b["tile"] not in ("X", "none")), None)
        if b is None:
            continue
        sl = _slot(8000 + len(logs[seat]), b, np.asarray(b["p"]))
        if b["gone"]:
            sl.t_removed = b["t_last"] + 0.2
        insert_slot(logs[seat], sl)
        problems.append(f"{seat}'s discard {b['tile']} at {b['t_first']:.1f}s found in a dense read (turn {i} was skipped)")
        added += 1
    return added


STILL_FRAMES = 3            # a still run: this many 5 fps frames (0.6 s) ...
STILL_FLICKER = 1           # ... each reading like the run's first, but for a single frame off in this many boxes
GAP = 0.5                   # s: frames further apart than this are not consecutive
BEFORE_MAX, AFTER_MAX = 60.0, 20.0   # s of hand row read before and after the discard of an uncertain draw


def _row(frame: dict) -> list[dict]:
    """The hand row of one reading, left to right (a meld moved beside it and slivers left out)."""
    return sorted((b for b in frame["boxes"] if b.get("role", "tile") == "tile"), key=lambda b: (b["xyxy"][0] + b["xyxy"][2]) / 2)


def still_runs(frames: list[dict], region: str) -> list[dict]:
    """Still views of a hand row found by its reading, not by the motion test: runs of at least STILL_FRAMES
    consecutive frames with the same count and the same tile in every position (a single frame may be off in
    STILL_FLICKER boxes when the next one reads as the run again). Each is an observation in the calm format (mean
    posterior per position), so hand_evidence takes it as a calm view."""
    rows = [_row(f) for f in frames]
    tops = [[int(np.argmax(b["p"])) for b in r] for r in rows]
    out: list[dict] = []

    def diff(a: int, b: int) -> int:
        return 99 if len(rows[a]) != len(rows[b]) else sum(1 for x, y in zip(tops[a], tops[b]) if x != y)

    i = 0
    while i < len(frames):
        k = i + 1
        # a frame holds the run when it reads like its first, or differs in a box or so that the next frame restores
        # (a flicker); a change that stays is a change, and a gap in the frames ends the run
        while (k < len(frames) and frames[k]["t"] - frames[k - 1]["t"] <= GAP
               and (diff(k, i) == 0 or (diff(k, i) <= STILL_FLICKER and (k + 1 == len(frames) or diff(k + 1, i) == 0)))):
            k += 1
        if k - i >= STILL_FRAMES and rows[i]:
            n = len(rows[i])
            ps = [np.mean([np.asarray(rows[q][x]["p"], np.float64) for q in range(i, k)], axis=0) for x in range(n)]
            out.append({"region": region, "t0": frames[i]["t"], "t1": frames[k - 1]["t"], "n_readings": k - i, "n_used": k - i,
                        "count": n, "quality": float(np.mean([p.max() for p in ps])), "partial": False, "dense": True,
                        "slots": [{"key": [0, x], "p": p.tolist()} for x, p in enumerate(ps)]})
        i = k
    return out


def draws(low: list[tuple[str, int]], model, turns: list[Turn], melds_before: dict[str, dict[int, int]], entry: dict,
          obs: dict, models, work_dir, t0: float, problems: list[str]) -> int:
    """For the supplied draws, read the player's hand row at 5 fps from its own previous discard
    to the next player's discard, and take every still run of it as a view of the hand (DESIGN.md 4.8 "Uncertain
    draws"): the hand before the draw, the moment after it, the hand after the discard. The draws of one seat
    whose stretches overlap are read in one window. Adds the evidence to the model; returns how many of the
    draws got a view that pins them (a full row on either side, or the moment after the draw).
    Callers select acquisition candidates by feasible alternative cost gaps;
    certified confidence bounds independently determine what needs review."""
    det, clf, video_path, cal = models
    spans: dict[str, list[tuple[float, float, int]]] = {}
    for s, j in low:
        st = next((x for x in model.turns[s] if x.j == j), None)
        if st is None or st.kind not in ("draw", "kan"):
            continue
        turn = next((t for t in turns if t.seat == s and abs(t.t - st.t_discard) < 0.6), None)
        if turn is None:
            continue
        own_prev = max((t.t for t in turns if t.seat == s and t.t < turn.t - 0.6), default=t0)
        nxt = turns[turn.i + 1].t if turn.i + 1 < len(turns) else st.t_discard + AFTER_MAX
        spans.setdefault(s, []).append((max(own_prev + 1.0, st.t_discard - BEFORE_MAX, t0), min(nxt, st.t_discard + AFTER_MAX), j))
    n_new = 0
    for s, sp in spans.items():
        corner = corner_of(entry, s)
        windows: list[list] = []
        for lo, hi, _ in sorted(sp):
            if windows and lo <= windows[-1][1]:
                windows[-1][1] = max(windows[-1][1], hi)
            else:
                windows.append([lo, hi])
        runs: list[dict] = []
        for lo, hi in windows:
            try:
                reads = dense_reads(video_path, cal, work_dir, det, clf, lo - 0.5, hi + 0.5, [f"hand:{corner}"])
            except Exception as ex:  # noqa: BLE001
                problems.append(f"dense hand read for {s} over {lo:.0f}-{hi:.0f}s failed: {ex}")
                continue
            runs += still_runs(reads[f"hand:{corner}"], f"hand:{corner}")
        hev, dev, _ = hand_evidence(s, model.turns[s], runs, melds_before[s], dealer=(s == model.dealer),
                                    wins_by_tsumo=(s == model.tsumo_winner))
        model.hand_ev += hev
        model.draw_ev += dev
        n_new += sum(1 for _, _, j in sp
                     if any(d.j == j for d in dev)
                     or any(e.j in (j - 1, j) and not e.after_draw and not e.subset and not e.hidden for e in hev))
    if low:
        problems.append(f"dense hand reads for {len(low)} uncertain draws: still views pin {n_new} of them")
    return n_new


def turned_tile(seat: str, turns: list[Turn], entry: dict, models, work_dir, t0: float,
                problems: list[str]) -> Optional[Turn]:
    """The declaring discard of a seat whose turned tile no calm reading flagged. A riichi tile lies across the
    row, so it is far wider than its neighbours: first the boxes already read, then, when the tile was called
    away before any calm frame (or the box was cut), a dense read of the pond around each discard."""
    mine = [t for t in turns if t.seat == seat and t.slot is not None]
    if not mine:
        return None
    boxes = [(t, _aspect(t.slot.xyxy)) for t in mine if t.slot.xyxy]
    if boxes:
        med = float(np.median([a for _, a in boxes]))
        wide = [(t, a) for t, a in boxes if a > 1.25 * med]
        if len(wide) == 1:
            t, a = wide[0]
            problems.append(f"riichi of {seat}: the discard at {t.t:.0f}s lies across the row (aspect {a:.2f} "
                            f"against {med:.2f}): it is the declaration")
            return t
    if models is None:
        return None
    det, clf, video_path, cal = models
    corner = corner_of(entry, seat)
    for t in mine:
        lo = max(t0, (t.slot.t_window[0] if t.slot.t_window else t.t) - 2.0)
        hi = t.t + 4.0
        if hi - lo > 30.0 or not t.slot.xyxy:
            continue
        try:
            reads = dense_pond_reads(video_path, cal, work_dir, det, clf, lo, hi, [corner])[corner]
        except Exception as ex:  # noqa: BLE001
            problems.append(f"dense read for the riichi of {seat} at {t.t:.0f}s failed: {ex}")
            return None
        cx = (t.slot.xyxy[0] + t.slot.xyxy[2]) / 2
        cy = (t.slot.xyxy[1] + t.slot.xyxy[3]) / 2
        tw = t.slot.xyxy[2] - t.slot.xyxy[0]
        # a tile at rest does not turn: the box at this position must be wide in most frames it is there, over
        # at least three of them; a single wide box is a tile in the player's fingers, not a declaration
        run = side = 0
        for r in reads:
            near = [b for b in r["boxes"] if b.get("role") == "tile"
                    and abs((b["xyxy"][0] + b["xyxy"][2]) / 2 - cx) < tw and abs((b["xyxy"][1] + b["xyxy"][3]) / 2 - cy) < tw]
            if not near:
                run = side = 0
                continue
            b = min(near, key=lambda b: abs((b["xyxy"][0] + b["xyxy"][2]) / 2 - cx))
            run += 1
            side += 1 if _aspect(b["xyxy"]) > 1.15 else 0
            if run >= 3 and side >= 0.6 * run:
                problems.append(f"riichi of {seat}: its discard at {t.t:.0f}s lies across the row in {side} of {run} "
                                f"dense frames: it is the declaration")
                return t
    return None


def _aspect(box) -> float:
    """Width over height of a pond box: an upright tile is about 0.78, a turned one about 1.3."""
    x0, y0, x1, y1 = box
    return (x1 - x0) / max(1.0, y1 - y0)
