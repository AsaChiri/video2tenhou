# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 4: one voted observation per region per calm interval.

Within a calm interval nothing changes, so the readings of the interval are
readings of the same content. Readings whose box count differs from the
interval's mode are dropped; the surviving readings vote per slot. A slot
whose readings disagree (top class changes) is flagged: disagreement is the
misrecognition signal, the slot keeps its full distribution. The one change a
calm interval can hold is a quick discard or draw: a count that changes once
and stays changed splits the interval into two observations.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from .calm import REGIONS, Interval
from .files import atomic_write_json, atomic_write_text, json_digest, sha256_text
from .perception.evidence_policy import prepare_reading, resolve_policy
from .perception.tiles import CLASSES
from .read import load_reads, manifest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from video2tenhou.perception.evidence_policy import EvidencePolicy

MIN_CALM_READING_RUN = 2
MERGEABLE_RUN_COUNT = 2


NC = len(CLASSES)
PERSIST = 0.5
OBSERVATION_VERSION = 2


LOGGER = logging.getLogger("video2tenhou.observe")


@dataclass
class Slot:
    """A stable position's weighted tile vote and evidence disagreement."""

    # Keys encode pond (row, col), hand (index,), meld (group, index),
    # or indicator ("ind", i) positions.
    key: tuple
    p: np.ndarray  # normalised posterior
    seen: int  # readings in which the slot appeared
    sideways: float  # fraction of readings with the sideways flag
    disagree: bool  # top class changed across readings
    xyxy: tuple  # mean box in region pixels

    @property
    def tile(self) -> str:
        """Most likely tile notation; uncertainty remains in the full posterior."""
        return CLASSES[int(np.argmax(self.p))]

    @property
    def conf(self) -> float:
        """Probability of the most likely tile in the normalized vote."""
        return float(np.max(self.p))

    def to_dict(self) -> dict:
        """Serialize the vote and provenance needed by decoding and review."""
        return {
            "key": list(self.key),
            "tile": self.tile,
            "conf": round(self.conf, 4),
            "seen": self.seen,
            "sideways": round(self.sideways, 2),
            "disagree": self.disagree,
            "xyxy": [round(v, 1) for v in self.xyxy],
            "p": [round(float(v), 4) for v in self.p],
        }


@dataclass
class Observation:
    """A calm state with independent hand, indicator and adjacent-meld evidence."""

    region: str
    t0: float  # first and last reading that went into the vote (reads are clipped to
    t1: float  # the hand window, so this can be narrower than the calm interval)
    n_readings: int  # readings in the interval
    n_used: int  # readings that agreed on the box count
    count: int  # boxes per reading (mode), tiles only
    quality: float
    slots: list[Slot] = field(default_factory=list)
    indicators: list[Slot] = field(default_factory=list)
    # read-floor interval: what is seen is there, what is absent may be hidden
    partial: bool = False
    iv_t0: float = 0.0  # the calm interval's bounds
    iv_t1: float = 0.0
    # hands: groups lying beside the row (melds moved at a reveal)
    extra: list[list[Slot]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the stage-4 cache representation, including partial-view status."""
        return {
            "region": self.region,
            "t0": self.t0,
            "t1": self.t1,
            "iv_t0": self.iv_t0,
            "iv_t1": self.iv_t1,
            "n_readings": self.n_readings,
            "n_used": self.n_used,
            "count": self.count,
            "quality": round(self.quality, 3),
            "slots": [s.to_dict() for s in self.slots],
            "indicators": [s.to_dict() for s in self.indicators],
            "partial": self.partial,
            "extra": [[s.to_dict() for s in g] for g in self.extra],
        }


def _key(kind: str, b: dict, i: int) -> tuple | None:
    if kind == "pond":
        return (b["row"], b["col"]) if "row" in b else None
    if kind == "meld":
        return (b.get("group", 0), i)
    return (i,)


def _vote(entries: list[tuple[dict, float]]) -> tuple[np.ndarray, float, bool, tuple]:
    """entries: (box, weight) -> (posterior, sideways fraction, disagree, mean box)."""
    p = np.zeros(NC, np.float64)
    tops = set()
    side = 0.0
    xy = np.zeros(4)
    for b, w in entries:
        q = np.asarray(b["p"], np.float64)
        p += w * q
        tops.add(int(np.argmax(q)))
        side += b["sideways"]
        xy += np.asarray(b["xyxy"])
    p = p / max(p.sum(), 1e-9)
    n = len(entries)
    return p, side / n, len(tops) > 1, tuple(xy / n)


def _prepared(
    kind: str,
    readings: list[dict],
    iv: Interval,
    *,
    policy: EvidencePolicy | dict | None = None,
) -> list[dict]:
    """Prepare interval readings while preserving all reusable raw boxes."""
    effective = resolve_policy(policy)
    rs = sorted(
        (r for r in readings if iv.t0 - 1e-6 <= r["t"] <= iv.t1 + 1e-6),
        key=lambda r: r["t"],
    )
    return [prepare_reading(kind, r, stage="sparse", policy=effective) for r in rs]


def _count(r: dict) -> int:
    return sum(1 for b in r["boxes"] if b.get("role", "tile") == "tile")


def observe_interval(
    region: str,
    readings: list[dict],
    iv: Interval,
    *,
    policy: EvidencePolicy | dict | None = None,
) -> list[Observation]:
    """Vote one calm interval: one observation, or two when its count changed once.

    A box count that changes once and stays changed is a discard or a draw made
    quickly enough to leave the interval calm. Two runs of at least two readings each
    are two states; a single odd reading is noise and the mode decides. Readings off
    the mode count are dropped. Confidence floors follow the supplied policy; raw
    reading objects remain reusable.
    """
    kind = region.partition(":")[0]
    prepared = _prepared(kind, readings, iv, policy=policy)
    rs = [r for r in prepared if not r.get("rejected")]
    runs: list[list[dict]] = []
    for r in rs:
        if runs and _count(runs[-1][-1]) == _count(r):
            runs[-1].append(r)
        else:
            runs.append([r])
    runs = [run for run in runs if len(run) >= MIN_CALM_READING_RUN]
    merged: list[list[dict]] = []
    for run in runs:
        if merged and _count(merged[-1][0]) == _count(run[0]):
            merged[-1] += run
        else:
            merged.append(run)
    if len(merged) == MERGEABLE_RUN_COUNT:
        a, b = merged
        first, second = replace(iv, t1=a[-1]["t"]), replace(iv, t0=b[0]["t"])
        return [
            _observe_prepared(
                region, [r for r in prepared if r["t"] <= first.t1 + 1e-6], first
            ),
            _observe_prepared(
                region, [r for r in prepared if r["t"] >= second.t0 - 1e-6], second
            ),
        ]
    return [_observe_prepared(region, prepared, iv)]


def _group_reading_boxes(kind: str, used: list[dict]) -> tuple[dict, dict, dict]:
    """Group standing tiles, exposed extras and indicators independently."""
    by_key: dict[tuple, list[tuple[dict, float]]] = defaultdict(list)
    ex_key: dict[tuple, list[tuple[dict, float]]] = defaultdict(list)
    ind: dict[int, list[tuple[dict, float]]] = defaultdict(list)
    for r in used:
        j = 0
        k_ind = 0
        k_ex: dict[int, int] = defaultdict(int)
        for b in r["boxes"]:
            role = b.get("role", "tile")
            if role == "indicator":
                ind[k_ind].append((b, b["conf"]))
                k_ind += 1
                continue
            if (
                role == "extra"
            ):  # a hand's meld beside the row: kept apart from the hand
                g = int(b.get("group", 1))
                ex_key[(g, k_ex[g])].append((b, b["conf"]))
                k_ex[g] += 1
                continue
            if role != "tile":  # slivers and the overflow of a neighbouring pond
                continue
            key = _key(kind, b, j)
            j += 1
            if key is not None:
                by_key[key].append((b, b["conf"]))
    return by_key, ex_key, ind


def _observe_prepared(region: str, rs: list[dict], iv: Interval) -> Observation:
    """Vote on already structured evidence; the interval split reuses the work."""
    kind = region.partition(":")[0]
    obs = Observation(
        region,
        iv.t0,
        iv.t1,
        len(rs),
        0,
        0,
        0.0,
        partial=iv.partial,
        iv_t0=iv.t0,
        iv_t1=iv.t1,
    )
    if not rs:
        return obs
    # a pond reading with a row of more than six positions is impossible (rule of six):
    # the reader rejects
    # it as a whole and it takes no part in the vote
    rs = [r for r in rs if not r.get("rejected")]
    if not rs:
        return obs
    counts = [_count(r) for r in rs]
    mode = Counter(counts).most_common(1)[0][0]
    used = [r for r, c in zip(rs, counts, strict=False) if c == mode]
    obs.count, obs.n_used = mode, len(used)
    # the observation spans what was seen: reads are clipped to the hand window and
    # sampled sparsely
    obs.t0, obs.t1 = min(r["t"] for r in used), max(r["t"] for r in used)
    by_key, ex_key, ind = _group_reading_boxes(kind, used)
    # a tile at rest is seen in every reading of the interval; a slot present in fewer
    # than half
    # of the readings is a flickering false detection
    need = max(1, int(np.ceil(PERSIST * len(used))))
    for key in sorted(by_key):
        if len(by_key[key]) < need:
            continue
        p, side, dis, xy = _vote(by_key[key])
        obs.slots.append(Slot(key, p, len(by_key[key]), side, dis, xy))
    for k in sorted(ind):
        p, side, dis, xy = _vote(ind[k])
        obs.indicators.append(Slot(("ind", k), p, len(ind[k]), side, dis, xy))
    groups: dict[int, list[Slot]] = defaultdict(list)
    for key in sorted(ex_key):
        if len(ex_key[key]) < need:
            continue
        p, side, dis, xy = _vote(ex_key[key])
        groups[key[0]].append(Slot(("extra", *key), p, len(ex_key[key]), side, dis, xy))
    obs.extra = [groups[g] for g in sorted(groups)]
    mean_conf = (
        float(np.mean([b["conf"] for r in used for b in r["boxes"]]))
        if any(r["boxes"] for r in used)
        else 0.0
    )
    obs.quality = (len(used) / len(rs)) * mean_conf
    return obs


def _provenance_path(work: Path, hand: int) -> Path:
    return work / "obs" / "provenance" / f"{hand:02d}.json"


def provenance(work: Path, hand: int) -> dict | None:
    """Return a hand's observation completion record, or None when none is complete.

    The record holds the inputs the vote was bound to, including the digest of the
    reading manifest, and the digest of the observation text as written.
    """
    try:
        record = json.loads(_provenance_path(work, hand).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return None
    return record if isinstance(record, dict) else None


def _observation_inputs(
    work: Path,
    hand: dict,
    ivs: list[Interval],
    *,
    policy: EvidencePolicy | dict | None = None,
) -> dict:
    reads = manifest(work, hand["hand"])
    intervals = [
        [iv.region, iv.t0, iv.t1, bool(iv.partial)]
        for iv in ivs
        if iv.region in REGIONS
        and iv.calm
        and iv.t1 >= hand["t_start"]
        and iv.t0 <= hand["t_end"]
    ]
    return {
        "version": OBSERVATION_VERSION,
        "window": [hand["t_start"], hand["t_end"]],
        "reads": None if reads is None else json_digest(reads),
        "intervals": intervals,
        "classes": list(CLASSES),
        "evidence_policy": resolve_policy(policy).fingerprint_for("sparse"),
        "persist": PERSIST,
    }


def _observation_current(work: Path, hand: dict, inputs: dict) -> bool:
    record = provenance(work, hand["hand"])
    return (
        inputs["reads"] is not None
        and record is not None
        and record.get("inputs") == inputs
    )


def validate_observation_cache(
    work: Path,
    hands: list[dict],
    ivs: list[Interval] | None = None,
    *,
    policy: EvidencePolicy | dict | None = None,
) -> None:
    """Require voted evidence derived from the current reads and calm intervals.

    Review rebuilds use the saved calm intervals when ``ivs`` is omitted. Unproven
    or interrupted observation publications must be refreshed through normal
    analysis before any hand is decoded; this check does not alter files.
    The effective sparse policy must match the vote that produced the cache.
    """
    if not hands:
        return
    if ivs is None:
        try:
            ivs = [
                Interval(**json.loads(line))
                for line in (work / "calm.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            UnicodeError,
            TypeError,
        ) as error:
            raise ValueError(
                "Saved observations cannot be verified. Choose Analyze recording to"
                " refresh evidence before rebuilding logs."
            ) from error
    stale = [
        str(hand["hand"])
        for hand in hands
        if not _observation_current(
            work, hand, _observation_inputs(work, hand, ivs, policy=policy)
        )
    ]
    if stale:
        raise ValueError(
            f"Saved observations are stale for hand(s) {', '.join(stale)}. "
            "Choose Analyze recording to refresh evidence before rebuilding logs."
        )


def run_observe(
    work: Path,
    hands: list[dict],
    ivs: list[Interval],
    *,
    force: bool = False,
    touched: set[int] | None = None,
    policy: EvidencePolicy | dict | None = None,
    log: Callable[[str], None] = LOGGER.info,
) -> dict:
    """Vote current readings, atomically publishing observations and their provenance.

    Reuse requires the same reading manifest (which records the digest of every
    reading file), relevant calm intervals, effective sparse evidence policy and
    voting settings. ``touched`` forces recomputation but is not the sole
    invalidation signal: a prior process may have stopped after publishing readings
    and before updating observations. A rewrite removes the old provenance before
    replacing the observations and writes the new one last. Returned
    ``changed_hands`` identifies results whose downstream decode must be refreshed,
    including changes recovered from an earlier interrupted run.
    """
    policy = resolve_policy(policy)
    stats = defaultdict(int)
    changed = []
    odir = work / "obs"
    odir.mkdir(parents=True, exist_ok=True)
    for h in hands:
        inputs = _observation_inputs(work, h, ivs, policy=policy)
        if inputs["reads"] is None or (
            not force
            and h["hand"] not in (touched or set())
            and _observation_current(work, h, inputs)
        ):
            continue
        reads = load_reads(work, h["hand"])
        result: dict[str, list[dict]] = {}
        for region in REGIONS:
            obs_list = []
            for iv in ivs:
                if (
                    iv.region != region
                    or not iv.calm
                    or iv.t1 < h["t_start"]
                    or iv.t0 > h["t_end"]
                ):
                    continue
                obs_list.extend(
                    observation.to_dict()
                    for observation in observe_interval(
                        region, reads[region], iv, policy=policy
                    )
                    if observation.n_readings
                )
            result[region] = obs_list
            stats["observations"] += len(obs_list)
        if inputs != _observation_inputs(work, h, ivs, policy=policy):
            raise ValueError(
                "Reading inputs changed while building observations; retry analysis."
            )
        text = json.dumps(result, ensure_ascii=False)
        record = _provenance_path(work, h["hand"])
        record.unlink(missing_ok=True)
        atomic_write_text(odir / f"{h['hand']:02d}.json", text)
        atomic_write_json(
            record, {"inputs": inputs, "output_sha256": sha256_text(text)}
        )
        changed.append(h["hand"])
        stats["hands"] += 1
        log(
            f"  observe hand {h['hand']:2d}: "
            f"{sum(len(v) for v in result.values())} observations"
        )
    return {**stats, "changed_hands": changed}


def load_obs(work: Path, hand: int) -> dict[str, list[dict]]:
    """Load one hand's voted evidence; a missing cache is an explicit error."""
    return json.loads((work / "obs" / f"{hand:02d}.json").read_text(encoding="utf-8"))
