# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve unresolved pond replacements without inventing a call or discard.

Posterior alignment can mistake a changed reading of the last tile for a refill.
These helpers expose that ambiguity, plan bounded acquisition, and create the
existing discard review item. A continuous full-pond geometric track can replace
overlapping sparse evidence; repeated identity predictions alone cannot establish
whether a call really occurred.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

from video2tenhou.read import ReadContext, dense_reads
from video2tenhou.train.data import CLASSES

from .hand import corner_of

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from video2tenhou.engine.ponds import PondSlot
    from video2tenhou.engine.turns import Turn
    from video2tenhou.read import ReadModels

MIN_ANCHORED_CALL_VIEWS = 2
DISCARD_FACT_WINDOW = 3
MIN_REPLACEMENT_FRAMES = 3
MAX_REPLACEMENT_EDGE_GAP = 0.25
MAX_REPLACEMENT_FRAME_GAP = 0.5
MIN_REPLACEMENT_OVERLAP = 0.3
MIN_REMAINING_POSTERIOR_MASS = -1e-8
MIN_REPLACEMENT_TILE_VOTES = 3


MAX_REPLACEMENT_WINDOW = 30.0


def replacement_requests(
    turns: list[Turn], entry: dict, facts: dict, t0: float, t1: float
) -> list[dict]:
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
        if slot is None:
            continue
        pending = slot.pending_replacement
        if not pending:
            continue
        call = turn.call
        corroborated = call is not None and (
            call.human
            or (
                call.anchor == "discard"
                and call.seen >= MIN_ANCHORED_CALL_VIEWS
                and not call.partial_only
            )
        )
        if corroborated:
            continue
        reviewed = any(
            f["seat"] == turn.seat
            and abs(float(f["t"]) - turn.t) <= DISCARD_FACT_WINDOW
            for f in facts.get("discard", [])
        )
        if reviewed:
            continue
        lo = max(t0, float(pending["stable_view"]["t0"]))
        hi = min(t1, float(pending["t1"]))
        # A long blind gap remains reviewable, but never expands into an
        # unbounded rescan or silently claims coverage of the whole gap.
        window = [lo, hi] if 0 < hi - lo <= MAX_REPLACEMENT_WINDOW else None
        payload = {
            "seat": turn.seat,
            "j": j,
            "slot_id": slot.id,
            "t": turn.t,
            "region": f"pond:{corner_of(entry, turn.seat)}",
            "window": window,
            "observed": slot.tile,
            "pending": deepcopy(pending),
            "call_ambiguous": call is not None,
        }
        identity = {key: value for key, value in payload.items() if key != "observed"}
        signature = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        requests.append(
            {
                **payload,
                "signature": signature,
                "acquired": bool(
                    slot.replacement_acquisition
                    and slot.replacement_acquisition["signature"] == signature
                ),
            }
        )
    return requests


def replacement_question(request: dict, *, chosen: str | None = None) -> dict:
    """Ask for the existing discard identity while retaining the contrary times.

    Unresolved correspondence cannot be discharged by a solver certificate.
    After verified acquisition, callers use this only for an uncertified identity.
    This question does not assert which reading is true or request approval of a
    structural call; the caller must retain any separate call question as well.
    """
    views = request["pending"]["views"]
    posterior = sum(
        (
            np.asarray(view["tile"]["p"]) * max(1, view["tile"]["seen"])
            for view in views
        ),
        np.zeros(len(CLASSES)),
    )
    alternative = CLASSES[int(np.argmax(posterior))]
    return {
        "kind": "discard",
        "seat": request["seat"],
        "j": request["j"],
        "t": request["t"],
        "tile": chosen or request["observed"],
        "observed": request["observed"],
        "runner_up": alternative,
        "evidence_t": views[-1]["t0"],
        "tracking_uncertain": not request.get("acquired", False),
        "text": (
            (
                "Continuous pond tracking links the conflicting views, but "
                "reconstruction could not certify the discard identity. Check both "
                "views."
            )
            if request.get("acquired")
            else (
                "A later pond view conflicts with this discard, and its "
                "correspondence is unresolved. Check the discard and the later view"
                " before accepting the log."
            )
        ),
    }


def read_replacement(request: dict, models: ReadModels, work_dir: Path) -> list[dict]:
    """Read a planned window using normal dense provenance and retention policy.

    Return raw structured dense readings for a correspondence probe; do not vote
    them into an existing slot or double-count overlapping sparse observations.
    The decoder owns attempt deduplication by request signature and error/review
    reporting. A request with no bounded window returns no acquired evidence.
    """
    if request["window"] is None:
        return []

    detector, classifier, video, calibration = models
    lo, hi = request["window"]
    return dense_reads(
        ReadContext(video, calibration, work_dir, detector, classifier),
        lo,
        hi,
        [request["region"]],
    )[request["region"]]


def _replacement_matches_slot(slot: PondSlot, request: dict) -> bool:
    return not (
        request["window"] is None
        or request.get("call_ambiguous")
        or slot.t_removed is not None
        or slot.id != request["slot_id"]
        or slot.pending_replacement != request["pending"]
        or slot.replacement_acquisition is not None
    )


def _replacement_frames(request: dict, readings: list[dict]) -> list[dict]:
    pending = request["pending"]
    lo, hi = request["window"]
    stable = pending["stable_view"]
    if (
        not pending["prefix_complete"]
        or stable["partial"]
        or lo != stable["t0"]
        or any(view["partial"] for view in pending["views"])
    ):
        return []
    frames = sorted((r for r in readings if lo <= r["t"] <= hi), key=lambda r: r["t"])
    if (
        len(frames) < MIN_REPLACEMENT_FRAMES
        or frames[0]["t"] - lo > MAX_REPLACEMENT_EDGE_GAP
        or hi - frames[-1]["t"] > MAX_REPLACEMENT_EDGE_GAP
        or any(frame.get("rejected") for frame in frames)
    ):
        return []
    return frames


def _unique_correspondence(left: list[dict], right: list[dict]) -> bool:
    overlap = np.asarray([[_iou(a["xyxy"], b["xyxy"]) for b in right] for a in left])
    if not np.all(np.isfinite(overlap)):
        return False
    for i in range(len(left)):
        if overlap[i, i] <= MIN_REPLACEMENT_OVERLAP:
            return False
        if any(
            overlap[i, j] >= overlap[i, i] or overlap[j, i] >= overlap[i, i]
            for j in range(len(left))
            if j != i
        ):
            return False
    return True


def _replacement_rows(
    frames: list[dict], stack_size: int, tile: str
) -> list[list[dict]]:
    rows = [
        sorted(
            (
                b
                for b in frame["boxes"]
                if b.get("role", "tile") == "tile" and "row" in b
            ),
            key=lambda b: (b["row"], b["col"]),
        )
        for frame in frames
    ]
    if stack_size < 1 or any(len(row) != stack_size for row in rows):
        return []
    prefix = [int(np.argmax(box["p"])) for box in rows[0][:-1]]
    if CLASSES[int(np.argmax(rows[0][-1]["p"]))] != tile:
        return []
    if any([int(np.argmax(box["p"])) for box in row[:-1]] != prefix for row in rows):
        return []
    for k, (left, right) in enumerate(itertools.pairwise(rows)):
        if not 0 < frames[k + 1]["t"] - frames[k]["t"] <= MAX_REPLACEMENT_FRAME_GAP:
            return []
        if not _unique_correspondence(left, right):
            return []
    return rows


def consume_replacement(slot: PondSlot, request: dict, readings: list[dict]) -> bool:
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
    if not _replacement_matches_slot(slot, request):
        return False
    pending = request["pending"]
    stable = pending["stable_view"]
    frames = _replacement_frames(request, readings)
    if not frames:
        return False
    rows = _replacement_rows(frames, pending["stack_size"], slot.tile)
    if not rows:
        return False
    old = np.asarray(stable["tile"]["p"], float) * max(1, stable["tile"]["seen"])
    remaining = slot.p - old
    if np.min(remaining) < MIN_REMAINING_POSTERIOR_MASS:
        return False
    replacement = sum(
        (np.asarray(row[-1]["p"], float) for row in rows), np.zeros(len(CLASSES))
    )
    if not np.all(np.isfinite(replacement)) or np.any(replacement < 0):
        return False
    slot.p = np.maximum(remaining, 0) + replacement
    slot.disagree += int(
        any(
            CLASSES[int(np.argmax(row[-1]["p"]))] != request["observed"] for row in rows
        )
    )
    slot.sideways += (
        float(np.mean([row[-1]["sideways"] for row in rows]))
        - stable["tile"]["sideways"]
    )
    slot.t_last = max(slot.t_last, frames[-1]["t"])
    slot.replacement_acquisition = {
        "signature": request["signature"],
        "window": list(request["window"]),
        "frames": len(frames),
        "replaced_sparse_readings": stable["tile"]["seen"],
        "supported_tiles": sorted(
            {
                CLASSES[k]
                for k in range(len(CLASSES))
                if sum(int(np.argmax(row[-1]["p"])) == k for row in rows)
                >= MIN_REPLACEMENT_TILE_VOTES
            }
        ),
    }
    return True


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x, y, u, v = a
    other_x0, other_y0, other_x1, other_y1 = b
    intersection = max(0, min(u, other_x1) - max(x, other_x0)) * max(
        0, min(v, other_y1) - max(y, other_y0)
    )
    return intersection / max(
        (u - x) * (v - y)
        + (other_x1 - other_x0) * (other_y1 - other_y0)
        - intersection,
        1e-9,
    )
