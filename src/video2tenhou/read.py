# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 3: read every region on (a bounded sample of) its calm frames.

Within a calm interval the content does not change, so at most `cap` frames
spread over the interval are read; a sequential 2 fps decode of each hand
window supplies the frames (no seeking). Output:
`work/<video>/reads/<hand>/<region>.jsonl`, one Reading per line, and
`done.json` with the model ids the readings came from.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

from . import video
from .cache import source_identity
from .calm import REGIONS, Interval, geometry_key, region_key
from .files import atomic_write_text
from .perception.evidence_policy import EvidencePolicy, prepare_reading
from .perception.reader import RegionClassifier, RegionDetector, read_regions
from .train.data import region_upright

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence


if TYPE_CHECKING:
    from pathlib import Path

    from .layout import Calibration

FPS = 2.0
CAP = 6
READ_BATCH = 48  # region crops, not full 1080p frames; limits host memory
PREPROCESSING = "upright-bgr-frame1920x1080-v2"


LOGGER = logging.getLogger("video2tenhou.read")


class ReadDetector(RegionDetector, Protocol):
    """Detector operations and provenance required for reusable evidence."""

    id: str
    evidence_policy: EvidencePolicy


class ReadClassifier(RegionClassifier, Protocol):
    """Classifier operations and metadata that identify cached posteriors."""

    id: str
    classes: list[str]
    T: float


@dataclass(frozen=True)
class ReadContext:
    """Source, geometry, model pair and cache directory for one recognition run."""

    path: str | Path
    calibration: Calibration
    work: Path
    detector: ReadDetector
    classifier: ReadClassifier


@dataclass(frozen=True, kw_only=True)
class ReadOptions:
    """Sampling cap and explicit replacement policy for sparse evidence."""

    cap: int = CAP
    force: bool = False
    reread: set[str] | None = None


type ReadModels = tuple[ReadDetector, ReadClassifier, Path, Calibration]
type CropBatch = list[tuple[float, str, np.ndarray]]


def _model_metadata(clf: ReadClassifier) -> dict:
    return {"classes": clf.classes, "temperature": clf.T}


def _read_identity(path: str | Path, clf: ReadClassifier) -> dict:
    return {
        "source": source_identity(path, refresh=True),
        "fps": FPS,
        "preprocessing": PREPROCESSING,
        "classifier_metadata": _model_metadata(clf),
    }


def validate_read_cache(
    context: ReadContext,
    hands: list[dict],
) -> None:
    """Require sparse evidence compatible with the models used for dense rereads.

    Review-only rebuilds bypass ``run_read``. Check every requested manifest
    before decoding any hand so source, model/runtime or calibration changes
    cannot silently combine incompatible evidence. Missing provenance
    requires normal analysis; this function neither repairs nor rewrites caches.
    """
    path = context.path
    cal = context.calibration
    work = context.work
    det = context.detector
    clf = context.classifier
    identity = _read_identity(path, clf)
    geom = {r: region_key(cal, r) for r in REGIONS}
    stale = []
    for h in hands:
        try:
            done = json.loads(
                (work / "reads" / f"{h['hand']:02d}" / "done.json").read_text(
                    encoding="utf-8"
                )
            )
            window = [round(h["t_start"] * FPS) / FPS, h["t_end"] + 1.0 / FPS]
            valid = (
                isinstance(done, dict)
                and done.get("detector") == det.id
                and done.get("classifier") == clf.id
                and done.get("identity") == identity
                and done.get("geometry") == geom
                and done.get("window") == window
            )
        except (FileNotFoundError, json.JSONDecodeError):
            valid = False
        if not valid:
            stale.append(str(h["hand"]))
    if stale:
        msg = (
            "Saved readings no longer match the recording, recognition models "
            f"or table geometry for hand(s) {', '.join(stale)}. Choose Analyze "
            "recording to refresh evidence before rebuilding logs."
        )
        raise ValueError(msg)


def planned_times(
    ivs: list[Interval], t0: float, t1: float, cap: int = CAP, fps: float = FPS
) -> dict[str, set[float]]:
    """Per region: the sample times (multiples of 1/fps) to read inside [t0, t1]."""
    out: dict[str, set[float]] = defaultdict(set)
    for iv in ivs:
        if not iv.calm or iv.t1 < t0 or iv.t0 > t1:
            continue
        a, b = max(iv.t0, t0), min(iv.t1, t1)
        n = round((b - a) * fps) + 1
        k = min(cap, n)
        for x in np.linspace(a, b, k):
            out[iv.region].add(round(round(x * fps) / fps, 3))
    return out


def _region_file(hdir: Path, r: str) -> Path:
    return hdir / f"{r.replace(':', '_')}.jsonl"


def read_lines(p: Path) -> list[dict]:
    """Read an atomically published region file; corrupted evidence fails visibly."""
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _same_models(done: dict, det: ReadDetector, clf: ReadClassifier, cap: int) -> bool:
    return (
        isinstance(done, dict)
        and done.get("detector") == det.id
        and done.get("classifier") == clf.id
        and done.get("cap") == cap
    )


def _moved_regions(done: dict, geom: dict[str, str]) -> set[str]:
    """Regions whose crop is not the one the stored readings were made from.

    A reading is a reading *of a crop*: a region the fit moved (a new video, or a
    rectangle dragged in the tool) has readings of something else on disk, and they are
    replaced rather than added to. Regions that did not move keep theirs, so correcting
    one meld camera does not re-read the whole video.
    """
    old = done.get("geometry") or {}
    if not isinstance(old, dict):  # unproven geometry cannot authenticate any region
        return set(geom)
    return {r for r, k in geom.items() if old.get(r) != k}


def _sparse_batches(
    path: str | Path,
    cal: Calibration,
    window: Sequence[float],
    plan: dict[str, set[float]],
) -> Generator[CropBatch]:
    """Yield planned region crops with the original frame window/order and batches."""
    samples = video.sample(path, fps=FPS, start=window[0], end=window[1])
    pending = []
    try:
        for t, frame in samples:
            tk = round(t, 3)
            for region in REGIONS:
                if tk not in plan[region]:
                    continue
                kind, _, corner = region.partition(":")
                img, _ = region_upright(frame, cal, kind, corner)
                pending.append((tk, region, img))
                if len(pending) >= READ_BATCH:
                    yield pending
                    pending = []
        if pending:
            yield pending
    finally:
        samples.close()


def _publish_readings(hdir: Path, hand: int, rows: dict, metadata: dict) -> None:
    """Publish a complete reading set, invalidating its old manifest before writes."""
    schedule = metadata["plan"]
    missing = {
        region: sorted(set(schedule[region]) - rows[region].keys())
        for region in REGIONS
        if set(schedule[region]) - rows[region].keys()
    }
    if missing:
        message = f"Video ended before planned evidence for hand {hand}: {missing}"
        raise RuntimeError(message)
    done = hdir / "done.json"
    done.unlink(missing_ok=True)
    for region in REGIONS:
        atomic_write_text(
            _region_file(hdir, region),
            "".join(json.dumps(rows[region][t]) + "\n" for t in sorted(rows[region])),
        )
    total = sum(len(values) for values in rows.values())
    atomic_write_text(
        done, json.dumps({"hand": hand, "readings": total, **metadata}, indent=1)
    )


def run_read(
    context: ReadContext,
    hands: list[dict],
    ivs: list[Interval],
    *,
    options: ReadOptions | None = None,
    log: Callable[[str], None] = LOGGER.info,
) -> tuple[dict, set[int]]:
    """Fill missing planned evidence and return statistics plus changed hand ids.

    ``force`` replaces all readings; ``reread`` replaces only named kinds
    (pond/hand/meld). Model, cap and crop-geometry changes invalidate the
    corresponding caches automatically. Source content, calibrated class metadata
    and preprocessing are also authenticated. Removed plan times are discarded;
    even removal-only changes invalidate downstream observations. Retries decode
    the same hand window so nominal timestamps refer to the same sampled frames.
    Completed hands are skipped before decoding; publication is atomic per file
    with a completion manifest written last. One crop batch is prepared ahead of
    inference; sample windows, region order and classifier batches are unchanged.
    """
    path = context.path
    cal = context.calibration
    work = context.work
    det = context.detector
    clf = context.classifier
    options = options or ReadOptions()
    cap, force, reread = options.cap, options.force, options.reread
    stats = defaultdict(int)
    touched: set[int] = set()
    asked = {r for r in REGIONS if r.partition(":")[0] in (reread or set())}
    geom = {r: region_key(cal, r) for r in REGIONS}
    identity = _read_identity(path, clf)
    for h in hands:
        hdir = work / "reads" / f"{h['hand']:02d}"
        done = hdir / "done.json"
        hdir.mkdir(parents=True, exist_ok=True)
        t0, t1 = h["t_start"], h["t_end"]
        plan = planned_times(ivs, t0, t1, cap)
        try:
            prev = json.loads(done.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            prev = None
        # readings from another detector, classifier or cap are stale as a whole: the
        # hand is read again.
        # A region that merely moved is re-read on its own.
        window = [round(t0 * FPS) / FPS, t1 + 1.0 / FPS]
        append = (
            prev is not None
            and not force
            and _same_models(prev, det, clf, cap)
            and prev.get("identity") == identity
            and prev.get("window") == window
        )
        moved = _moved_regions(prev, geom) if append else set()
        fresh = asked | moved
        if prev is not None and not force and not append:
            log(f"  read hand {h['hand']:2d}: evidence inputs changed, reading again")
        elif moved and h["hand"] == hands[0]["hand"]:
            log(
                f"  read: the geometry of {', '.join(sorted(moved))} changed; "
                "those regions are read again"
            )
        schedule = {r: sorted(plan[r]) for r in REGIONS}
        rows = {r: {} for r in REGIONS}
        changed = not append or bool(fresh)
        for r in REGIONS:
            old = read_lines(_region_file(hdir, r)) if append and r not in fresh else []
            rows[r] = {round(d["t"], 3): d for d in old if round(d["t"], 3) in plan[r]}
            changed |= len(rows[r]) != len(old)
            plan[r] -= rows[r].keys()
        if (
            not changed
            and not any(plan.values())
            and prev is not None
            and prev.get("plan") == schedule
        ):
            continue
        n = 0
        if any(plan.values()):
            # The decode window is part of frame identity. Narrowing a retry to
            # missing timestamps can select different frames in ffmpeg's fps filter.
            with closing(
                _prefetch_batches(_sparse_batches(path, cal, window, plan))
            ) as batches:
                for batch in batches:
                    for rd in read_regions(batch, det, clf):
                        rows[rd.region][round(rd.t, 3)] = rd.to_dict()
                    n += len(batch)
        _publish_readings(
            hdir,
            h["hand"],
            rows,
            {
                "detector": det.id,
                "classifier": clf.id,
                "cap": cap,
                "geometry": geom,
                "identity": identity,
                "window": window,
                "plan": schedule,
            },
        )
        touched.add(h["hand"])
        stats["hands"] += 1
        stats["readings"] += n
        log(f"  read hand {h['hand']:2d}: {n} readings")
    return dict(stats), touched


def load_reads(work: Path, hand: int) -> dict[str, list[dict]]:
    """Load all region evidence for one hand, rejecting malformed JSONL."""
    hdir = work / "reads" / f"{hand:02d}"
    return {r: read_lines(_region_file(hdir, r)) for r in REGIONS}


def dense_key(det: ReadDetector, clf: ReadClassifier, cal: Calibration) -> str:
    """Bind dense evidence to its models, geometry and retention policy.

    Bind dense evidence to recognition, geometry and its stage-specific retention
    policy.
    """
    return hashlib.sha256(
        json.dumps(
            [
                det.id,
                clf.id,
                _model_metadata(clf),
                PREPROCESSING,
                geometry_key(cal),
                det.evidence_policy.fingerprint_for("dense"),
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]


def clear_dense(work: Path) -> None:
    """Forget every dense read (a forced re-read starts them over as well)."""
    with suppress(FileNotFoundError):
        shutil.rmtree(work / "dense")


def _dense_batches(
    context: ReadContext,
    t0: float,
    t1: float,
    regions: Iterable[str],
    fps: float,
) -> Generator[CropBatch]:
    """Keep the existing sample order and batch boundaries while preparing crops."""
    samples = video.sample(context.path, fps=fps, start=max(0.0, t0), end=t1)
    pending = []
    try:
        for t, frame in samples:
            for region in regions:
                kind, _, corner = region.partition(":")
                img, _ = region_upright(frame, context.calibration, kind, corner)
                pending.append((round(t, 2), region, img))
                if len(pending) == READ_BATCH:
                    yield pending
                    pending = []
        if pending:
            yield pending
    finally:
        samples.close()


def _prefetch_batches(batches: Generator[CropBatch]) -> Generator[CropBatch]:
    """Prepare one batch ahead; inference stays on the caller's original thread.

    Only decoding and cropping overlap inference. At most one current and one
    next crop batch exist. Close/join the producer before releasing its sampler,
    including when inference fails; a producer failure propagates to the caller.
    """
    iterator = iter(batches)

    def advance() -> CropBatch | None:
        return next(iterator, None)

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dense-crops")
    future = pool.submit(advance)
    try:
        while True:
            batch = future.result()
            if batch is None:
                break
            future = pool.submit(advance)
            yield batch
    finally:
        future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        iterator.close()


def dense_reads(
    context: ReadContext,
    t0: float,
    t1: float,
    regions: list[str],
    *,
    fps: float = 5.0,
) -> dict[str, list[dict]]:
    """Read selected regions on every sampled frame of a short window.

    Read the given regions on every frame of [t0, t1] at `fps` (a short window where the
    calm reads were not enough), cached under work/dense by window, models and geometry.
    Returns region -> readings (dicts), structure recomputed as in observe. One CPU crop
    batch is prepared ahead of inference; sampled pixels, order and classifier batch
    boundaries are unchanged.
    """
    path = context.path
    cal = context.calibration
    work = context.work
    det = context.detector
    clf = context.classifier
    policy = det.evidence_policy
    ddir = work / "dense"
    ddir.mkdir(parents=True, exist_ok=True)
    signature = {
        "window": [t0, t1],
        "fps": fps,
        "regions": sorted(regions),
        "models_geometry": dense_key(det, clf, cal),
        "source": source_identity(path),
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[
        :32
    ]
    # Short names also work in deeply nested Windows project workspaces.
    key = f"{t0:.3f}_{t1:.3f}_{digest}.json"
    cache = ddir / key
    if cache.exists():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if (
                isinstance(saved, dict)
                and set(saved) == set(regions)
                and all(
                    isinstance(rows, list)
                    and all(
                        isinstance(row, dict)
                        and "t" in row
                        and isinstance(row.get("boxes"), list)
                        for row in rows
                    )
                    for rows in saved.values()
                )
            ):
                return saved
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # an interrupted write is recomputed, never evidence
    out: dict[str, list[dict]] = {r: [] for r in regions}
    with closing(
        _prefetch_batches(_dense_batches(context, t0, t1, regions, fps))
    ) as batches:
        for batch in batches:
            for rd in read_regions(batch, det, clf):
                out[rd.region].append(rd.to_dict())
    for r in regions:
        kind = r.partition(":")[0]
        out[r] = [
            prepare_reading(kind, rd, stage="dense", policy=policy) for rd in out[r]
        ]
    atomic_write_text(cache, json.dumps(out))
    return out


def dense_pond_reads(
    context: ReadContext,
    t0: float,
    t1: float,
    corners: Sequence[str],
    *,
    fps: float = 5.0,
) -> dict[str, list[dict]]:
    """Read dense pond evidence, returning corner keys for engine callers."""
    out = dense_reads(context, t0, t1, [f"pond:{c}" for c in corners], fps=fps)
    return {c: out[f"pond:{c}"] for c in corners}
