"""Preserve unresolved pond replacements without inventing a call or discard.

Posterior alignment can mistake a changed reading of the last tile for a refill.
These helpers expose that ambiguity, plan bounded acquisition, and create the
existing discard review item. A continuous full-pond geometric track can replace
overlapping sparse evidence; repeated identity predictions alone cannot establish
whether a call really occurred.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import numpy as np

from ..train.data import CLASSES
from .hand import corner_of

MAX_REPLACEMENT_WINDOW = 30.0


def replacement_requests(turns, entry: dict, facts: dict, t0: float, t1: float) -> list[dict]:
    """Identify unresolved source slots and one bounded pond window for each.

    An independently observed anchored call (or reviewed call) explains a refill;
    an inferred hidden call does not. A reviewed discard suppresses its identity
    question but is not evidence that a call occurred. Request identities include
    the full preserved conflicting evidence, supporting one attempt per decoder
    run without changing reusable raw recognition cache identities.
    """
    requests, counts = [], {}
    for turn in turns:
        j = counts.get(turn.seat, 0)
        counts[turn.seat] = j + 1
        slot = turn.slot
        pending = getattr(slot, "pending_replacement", None)
        if not pending:
            continue
        call = turn.call
        corroborated = call is not None and (
            call.human or (call.anchor == "discard" and call.seen >= 2 and not call.partial_only))
        if corroborated:
            continue
        reviewed = any(f["seat"] == turn.seat and abs(float(f["t"]) - turn.t) <= 3
                       for f in facts.get("discard", []))
        if reviewed:
            continue
        lo = max(t0, float(pending["stable_view"]["t0"]))
        hi = min(t1, float(pending["t1"]))
        # A long blind gap remains reviewable, but never expands into an
        # unbounded rescan or silently claims coverage of the whole gap.
        window = [lo, hi] if 0 < hi - lo <= MAX_REPLACEMENT_WINDOW else None
        payload = {"seat": turn.seat, "j": j, "slot_id": slot.id,
                   "t": turn.t, "region": f"pond:{corner_of(entry, turn.seat)}",
                   "window": window, "observed": slot.tile, "pending": deepcopy(pending),
                   "call_ambiguous": call is not None}
        identity = {key: value for key, value in payload.items() if key != "observed"}
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        requests.append({**payload, "signature": signature,
                         "acquired": bool(slot.replacement_acquisition and
                                          slot.replacement_acquisition["signature"] == signature)})
    return requests


def replacement_question(request: dict, *, chosen: str | None = None) -> dict:
    """Ask for the existing discard identity while retaining the contrary times.

    Unresolved correspondence cannot be discharged by a solver certificate.
    After verified acquisition, callers use this only for an uncertified identity.
    This question does not assert which reading is true or request approval of a
    structural call; the caller must retain any separate call question as well.
    """
    views = request["pending"]["views"]
    posterior = sum((np.asarray(view["tile"]["p"]) * max(1, view["tile"]["seen"])
                     for view in views), np.zeros(len(CLASSES)))
    alternative = CLASSES[int(np.argmax(posterior))]
    return {"kind": "discard", "seat": request["seat"], "j": request["j"], "t": request["t"],
            "tile": chosen or request["observed"], "observed": request["observed"], "runner_up": alternative,
            "evidence_t": views[-1]["t0"], "tracking_uncertain": not request.get("acquired", False),
            "text": ("Continuous pond tracking links the conflicting views, but reconstruction could not "
                     "certify the discard identity. Check both views."
                     if request.get("acquired") else
                     "A later pond view conflicts with this discard, and its correspondence is unresolved. "
                     "Check the discard and the later view before accepting the log.")}


def read_replacement(request: dict, models, work_dir) -> list[dict]:
    """Read a planned window using normal dense provenance and retention policy.

    Return raw structured dense readings for a correspondence probe; do not vote
    them into an existing slot or double-count overlapping sparse observations.
    The decoder owns attempt deduplication by request signature and error/review
    reporting. A request with no bounded window returns no acquired evidence.
    """
    if request["window"] is None:
        return []
    from ..read import dense_reads
    detector, classifier, video, calibration = models
    lo, hi = request["window"]
    return dense_reads(video, calibration, work_dir, detector, classifier, lo, hi,
                       [request["region"]])[request["region"]]


def consume_replacement(slot, request: dict, readings: list[dict]) -> bool:
    """Substitute dense evidence only after a continuous full-pond correspondence.

    Require unchanged count/order and prefix identities, unique mutual geometric
    matches with IoU above .3 at every adjacent frame, and no gap over .5 seconds.
    A removed slot or a possible call cannot use this identity-only path. The
    exact last stable sparse contribution is removed before adding dense tail
    posteriors; later conflicting sparse views were never in the old slot.
    Continuity associates observations; their disagreeing identities still need
    reconstruction. Only a certified, visually supported choice can close review.
    Return false without mutation for insufficient evidence or a repeated receipt.
    """
    if (request["window"] is None or request.get("call_ambiguous") or slot.t_removed is not None
            or slot.id != request["slot_id"] or slot.pending_replacement != request["pending"]
            or slot.replacement_acquisition is not None):
        return False
    pending = request["pending"]
    lo, hi = request["window"]
    stable = pending["stable_view"]
    if (not pending["prefix_complete"] or stable["partial"]
            or lo != stable["t0"] or any(v["partial"] for v in pending["views"])):
        return False
    frames = sorted((r for r in readings if lo <= r["t"] <= hi), key=lambda r: r["t"])
    if (len(frames) < 3 or frames[0]["t"] - lo > .25 or hi - frames[-1]["t"] > .25
            or any(r.get("rejected") for r in frames)):
        return False
    rows = [sorted((b for b in r["boxes"] if b.get("role", "tile") == "tile" and "row" in b),
                   key=lambda b: (b["row"], b["col"])) for r in frames]
    n = pending["stack_size"]
    if n < 1 or any(len(row) != n for row in rows):
        return False
    prefix = [int(np.argmax(b["p"])) for b in rows[0][:-1]]
    if CLASSES[int(np.argmax(rows[0][-1]["p"]))] != slot.tile:
        return False
    if any([int(np.argmax(b["p"])) for b in row[:-1]] != prefix for row in rows):
        return False
    for k, (left, right) in enumerate(zip(rows, rows[1:])):
        if not 0 < frames[k + 1]["t"] - frames[k]["t"] <= .5:
            return False
        overlap = np.asarray([[_iou(a["xyxy"], b["xyxy"]) for b in right] for a in left])
        if not np.all(np.isfinite(overlap)):
            return False
        for i in range(n):
            if overlap[i, i] <= .3:
                return False
            if any(overlap[i, j] >= overlap[i, i] or overlap[j, i] >= overlap[i, i]
                   for j in range(n) if j != i):
                return False
    old = np.asarray(stable["tile"]["p"], float) * max(1, stable["tile"]["seen"])
    remaining = slot.p - old
    if np.min(remaining) < -1e-8:
        return False
    replacement = sum((np.asarray(row[-1]["p"], float) for row in rows), np.zeros(len(CLASSES)))
    if not np.all(np.isfinite(replacement)) or np.any(replacement < 0):
        return False
    slot.p = np.maximum(remaining, 0) + replacement
    slot.disagree += int(any(CLASSES[int(np.argmax(row[-1]["p"]))] != request["observed"] for row in rows))
    slot.sideways += float(np.mean([row[-1]["sideways"] for row in rows])) - stable["tile"]["sideways"]
    slot.t_last = max(slot.t_last, frames[-1]["t"])
    slot.replacement_acquisition = {"signature": request["signature"], "window": list(request["window"]),
                                    "frames": len(frames), "replaced_sparse_readings": stable["tile"]["seen"],
                                    "supported_tiles": sorted({CLASSES[k] for k in range(len(CLASSES))
                                        if sum(int(np.argmax(row[-1]["p"])) == k for row in rows) >= 3})}
    return True


def _iou(a, b):
    x, y, u, v = a
    X, Y, U, V = b
    intersection = max(0, min(u, U) - max(x, X)) * max(0, min(v, V) - max(y, Y))
    return intersection / max((u - x) * (v - y) + (U - X) * (V - Y) - intersection, 1e-9)
