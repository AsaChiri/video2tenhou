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
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

from .calm import Interval, REGIONS
from .files import atomic_write_json, sha256_file
from .train.data import CLASSES
from .perception.evidence_policy import prepare_reading, resolve_policy

NC = len(CLASSES)
PERSIST = 0.5
OBSERVATION_VERSION = 2


@dataclass
class Slot:
    """A stable position's weighted tile vote and evidence disagreement."""
    key: tuple                      # pond: (row, col); hand: (index,); meld: (group, index); indicator: ("ind", i)
    p: np.ndarray                   # normalised posterior
    seen: int                       # readings in which the slot appeared
    sideways: float                 # fraction of readings with the sideways flag
    disagree: bool                  # top class changed across readings
    xyxy: tuple                     # mean box in region pixels

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
        return {"key": list(self.key), "tile": self.tile, "conf": round(self.conf, 4), "seen": self.seen,
                "sideways": round(self.sideways, 2), "disagree": self.disagree, "xyxy": [round(v, 1) for v in self.xyxy],
                "p": [round(float(v), 4) for v in self.p]}


@dataclass
class Observation:
    """A calm state with independent hand, indicator and adjacent-meld evidence."""
    region: str
    t0: float                       # first and last reading that went into the vote (reads are clipped to
    t1: float                       # the hand window, so this can be narrower than the calm interval)
    n_readings: int                 # readings in the interval
    n_used: int                     # readings that agreed on the box count
    count: int                      # boxes per reading (mode), tiles only
    quality: float
    slots: list[Slot] = field(default_factory=list)
    indicators: list[Slot] = field(default_factory=list)
    partial: bool = False           # read-floor interval: what is seen is there, what is absent may be hidden
    iv_t0: float = 0.0              # the calm interval's bounds
    iv_t1: float = 0.0
    extra: list[list[Slot]] = field(default_factory=list)   # hands: groups lying beside the row (melds moved at a reveal)

    def to_dict(self) -> dict:
        """Return the stage-4 cache representation, including partial-view status."""
        return {"region": self.region, "t0": self.t0, "t1": self.t1, "iv_t0": self.iv_t0, "iv_t1": self.iv_t1,
                "n_readings": self.n_readings, "n_used": self.n_used,
                "count": self.count, "quality": round(self.quality, 3), "slots": [s.to_dict() for s in self.slots],
                "indicators": [s.to_dict() for s in self.indicators], "partial": self.partial,
                "extra": [[s.to_dict() for s in g] for g in self.extra]}


def _key(kind: str, b: dict, i: int) -> Optional[tuple]:
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


def _prepared(kind: str, readings: list[dict], iv: Interval, *, policy=None) -> list[dict]:
    """Prepare fresh interval readings without removing boxes from reusable raw input."""
    effective = resolve_policy(policy)
    rs = sorted((r for r in readings if iv.t0 - 1e-6 <= r["t"] <= iv.t1 + 1e-6), key=lambda r: r["t"])
    return [prepare_reading(kind, r, stage="sparse", policy=effective, none_index=CLASSES.index("none")) for r in rs]


def _count(r: dict) -> int:
    return sum(1 for b in r["boxes"] if b.get("role", "tile") == "tile")


def observe_interval(region: str, readings: list[dict], iv: Interval, *, policy=None) -> list[Observation]:
    """The observations of one calm interval: one, or two when the box count changed once and stayed changed
    (a discard or a draw made quickly enough to leave the interval calm). Two runs of at least two readings
    each are two states; a single odd reading is noise and the mode decides (`observe`).
    Confidence floors follow the supplied policy; raw reading objects remain reusable.
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
    runs = [run for run in runs if len(run) >= 2]
    merged: list[list[dict]] = []
    for run in runs:
        if merged and _count(merged[-1][0]) == _count(run[0]):
            merged[-1] += run
        else:
            merged.append(run)
    if len(merged) == 2:
        a, b = merged
        first, second = replace(iv, t1=a[-1]["t"]), replace(iv, t0=b[0]["t"])
        return [_observe_prepared(region, [r for r in prepared if r["t"] <= first.t1 + 1e-6], first),
                _observe_prepared(region, [r for r in prepared if r["t"] >= second.t0 - 1e-6], second)]
    return [_observe_prepared(region, prepared, iv)]


def observe(region: str, readings: list[dict], iv: Interval, *, policy=None) -> Observation:
    """One voted observation of the readings inside `iv`: readings off the mode count are dropped."""
    return _observe_prepared(region, _prepared(region.partition(":")[0], readings, iv, policy=policy), iv)


def _observe_prepared(region: str, rs: list[dict], iv: Interval) -> Observation:
    """Vote on already structured evidence; the interval split reuses the work."""
    kind = region.partition(":")[0]
    obs = Observation(region, iv.t0, iv.t1, len(rs), 0, 0, 0.0, partial=iv.partial,
                      iv_t0=iv.t0, iv_t1=iv.t1)
    if not rs:
        return obs
    # a pond reading with a row of more than six positions is impossible (rule of six): the reader rejects
    # it as a whole and it takes no part in the vote
    rs = [r for r in rs if not r.get("rejected")]
    if not rs:
        return obs
    counts = [_count(r) for r in rs]
    mode = Counter(counts).most_common(1)[0][0]
    used = [r for r, c in zip(rs, counts) if c == mode]
    obs.count, obs.n_used = mode, len(used)
    # the observation spans what was seen: reads are clipped to the hand window and sampled sparsely
    obs.t0, obs.t1 = min(r["t"] for r in used), max(r["t"] for r in used)
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
            if role == "extra":                     # a hand's meld beside the row: kept apart from the hand
                g = int(b.get("group", 1))
                ex_key[(g, k_ex[g])].append((b, b["conf"]))
                k_ex[g] += 1
                continue
            if role != "tile":                      # slivers and the overflow of a neighbouring pond
                continue
            key = _key(kind, b, j)
            j += 1
            if key is not None:
                by_key[key].append((b, b["conf"]))
    # a tile at rest is seen in every reading of the interval; a slot present in fewer than half
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
        groups[key[0]].append(Slot(("extra",) + key, p, len(ex_key[key]), side, dis, xy))
    obs.extra = [groups[g] for g in sorted(groups)]
    mean_conf = float(np.mean([b["conf"] for r in used for b in r["boxes"]])) if any(r["boxes"] for r in used) else 0.0
    obs.quality = (len(used) / len(rs)) * mean_conf
    return obs


def _digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return sha256_file(path)


def _observation_inputs(work: Path, hand: dict, ivs: list[Interval], *, policy=None) -> dict:
    hdir = work / "reads" / f"{hand['hand']:02d}"
    intervals = [[iv.region, iv.t0, iv.t1, bool(iv.partial)] for iv in ivs
                 if iv.region in REGIONS and iv.calm
                 and iv.t1 >= hand["t_start"] and iv.t0 <= hand["t_end"]]
    return dict(version=OBSERVATION_VERSION, window=[hand["t_start"], hand["t_end"]],
                manifest=_digest(hdir / "done.json"),
                readings={region: _digest(hdir / f"{region.replace(':', '_')}.jsonl") for region in REGIONS},
                intervals=intervals, classes=list(CLASSES), evidence_policy=resolve_policy(policy).fingerprint_for("sparse"), persist=PERSIST)


def _observation_current(work: Path, hand: dict, inputs: dict) -> bool:
    try:
        manifest = json.loads((work / "obs" / "provenance" / f"{hand['hand']:02d}.json").read_text(encoding="utf-8"))
        return (inputs["manifest"] is not None and isinstance(manifest, dict) and manifest.get("inputs") == inputs
                and manifest.get("output_sha256") is not None
                and manifest["output_sha256"] == _digest(work / "obs" / f"{hand['hand']:02d}.json"))
    except (OSError, ValueError):
        return False


def validate_observation_cache(work: Path, hands: list[dict], ivs: list[Interval] | None = None,
                               *, policy=None) -> None:
    """Require voted evidence derived from the current reads and calm intervals.

    Review rebuilds use the saved calm intervals when ``ivs`` is omitted. Unproven,
    corrupted or interrupted observation publications must be refreshed through
    normal analysis before any hand is decoded; this check does not alter files.
    The effective sparse policy must match the vote that produced the cache.
    """
    if not hands:
        return
    try:
        if ivs is None:
            ivs = [Interval(**json.loads(line)) for line in (work / "calm.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        stale = [str(hand["hand"]) for hand in hands
                 if not _observation_current(work, hand, _observation_inputs(work, hand, ivs, policy=policy))]
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise ValueError("Saved observations cannot be verified. Choose Analyze recording to refresh evidence before rebuilding logs.") from error
    if stale:
        raise ValueError(f"Saved observations are stale for hand(s) {', '.join(stale)}. "
                         "Choose Analyze recording to refresh evidence before rebuilding logs.")


def run_observe(work: Path, hands: list[dict], ivs: list[Interval], *, force: bool = False,
                touched: Optional[set] = None, policy=None, log=print) -> dict:
    """Vote current readings, atomically publishing observations and their provenance.

    Reuse requires identical reading bytes, completion manifest, relevant calm
    intervals, effective sparse evidence policy and voting settings, plus an intact
    output. ``touched`` forces recomputation but is not the sole invalidation signal:
    a prior process may
    have stopped after publishing readings and before updating observations.
    Returned ``changed_hands`` identifies results whose downstream decode must
    be refreshed, including changes recovered from an earlier interrupted run.
    """
    from .read import load_reads
    policy = resolve_policy(policy)
    stats = defaultdict(int)
    changed = []
    odir = work / "obs"
    odir.mkdir(parents=True, exist_ok=True)
    for h in hands:
        out = odir / f"{h['hand']:02d}.json"
        if not (work / "reads" / f"{h['hand']:02d}" / "done.json").exists():
            continue
        inputs = _observation_inputs(work, h, ivs, policy=policy)
        if not force and h["hand"] not in (touched or set()) and _observation_current(work, h, inputs):
            continue
        reads = load_reads(work, h["hand"])
        result: dict[str, list[dict]] = {}
        for region in REGIONS:
            obs_list = []
            for iv in ivs:
                if iv.region != region or not iv.calm or iv.t1 < h["t_start"] or iv.t0 > h["t_end"]:
                    continue
                for o in observe_interval(region, reads[region], iv, policy=policy):
                    if o.n_readings:
                        obs_list.append(o.to_dict())
            result[region] = obs_list
            stats["observations"] += len(obs_list)
        if inputs != _observation_inputs(work, h, ivs, policy=policy):
            raise ValueError("Reading inputs changed while building observations; retry analysis.")
        atomic_write_json(out, result)
        atomic_write_json(odir / "provenance" / f"{h['hand']:02d}.json",
                     {"inputs": inputs, "output_sha256": _digest(out)})
        changed.append(h["hand"])
        stats["hands"] += 1
        log(f"  observe hand {h['hand']:2d}: {sum(len(v) for v in result.values())} observations")
    return {**stats, "changed_hands": changed}


def load_obs(work: Path, hand: int) -> dict[str, list[dict]]:
    """Load one hand's voted evidence; a missing cache is an explicit error."""
    return json.load(open(work / "obs" / f"{hand:02d}.json", encoding="utf-8"))
