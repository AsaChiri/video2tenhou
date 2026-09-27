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
import shutil
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Optional

import numpy as np

from . import video
from .cache import source_identity
from .calm import Interval, REGIONS, geometry_key, region_key
from .layout import Calibration
from .perception.classifier import Classifier
from .perception.detector import Detector
from .perception.evidence_policy import DEFAULT_POLICY, prepare_reading
from .perception.reader import read_regions
from .train.data import region_upright

FPS = 2.0
CAP = 6
READ_BATCH = 48  # region crops, not full 1080p frames; limits host memory
PREPROCESSING = "upright-bgr-frame1920x1080-v2"


def _model_metadata(clf) -> dict:
    return {"classes": getattr(clf, "classes", None), "temperature": getattr(clf, "T", None)}


def _read_identity(path: str | Path, clf: Classifier) -> dict:
    return {"source": source_identity(path, refresh=True), "fps": FPS,
            "preprocessing": PREPROCESSING, "classifier_metadata": _model_metadata(clf)}


def validate_read_cache(path: str | Path, cal: Calibration, work: Path, hands: list[dict],
                        det: Detector, clf: Classifier) -> None:
    """Require sparse evidence compatible with the models used for dense rereads.

    Review-only rebuilds bypass ``run_read``. Check every requested manifest
    before decoding any hand so source, model/runtime or calibration changes
    cannot silently combine incompatible evidence. Missing provenance
    requires normal analysis; this function neither repairs nor rewrites caches.
    """
    identity = _read_identity(path, clf)
    geom = {r: region_key(cal, r) for r in REGIONS}
    stale = []
    for h in hands:
        try:
            done = json.loads((work / "reads" / f"{h['hand']:02d}" / "done.json").read_text(encoding="utf-8"))
            window = [round(h["t_start"] * FPS) / FPS, h["t_end"] + 1.0 / FPS]
            valid = (isinstance(done, dict) and done.get("detector") == det.id
                     and done.get("classifier") == clf.id and done.get("identity") == identity
                     and done.get("geometry") == geom and done.get("window") == window)
        except (OSError, ValueError):
            valid = False
        if not valid:
            stale.append(str(h["hand"]))
    if stale:
        raise ValueError("Saved readings no longer match the recording, recognition models or table geometry "
                         f"for hand(s) {', '.join(stale)}. Choose Analyze recording to refresh evidence "
                         "before rebuilding logs.")


def _atomic_text(path: Path, content: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".read-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def planned_times(ivs: list[Interval], t0: float, t1: float, cap: int = CAP, fps: float = FPS) -> dict[str, set[float]]:
    """Per region: the sample times (multiples of 1/fps) to read inside [t0, t1]."""
    out: dict[str, set[float]] = defaultdict(set)
    for iv in ivs:
        if not iv.calm or iv.t1 < t0 or iv.t0 > t1:
            continue
        a, b = max(iv.t0, t0), min(iv.t1, t1)
        n = int(round((b - a) * fps)) + 1
        k = min(cap, n)
        for x in np.linspace(a, b, k):
            out[iv.region].add(round(round(x * fps) / fps, 3))
    return out


def _region_file(hdir: Path, r: str) -> Path:
    return hdir / f"{r.replace(':', '_')}.jsonl"


def read_lines(p: Path, log=print) -> list[dict]:
    """The readings of one jsonl file. A crash mid-write leaves a truncated last line; it is dropped from
    the file (the plan reads that time again) instead of stopping every later run."""
    if not p.exists():
        return []
    out: list[dict] = []
    bad = 0
    for line in open(p, encoding="utf-8").read().splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        log(f"  {p.parent.name}/{p.name}: {bad} malformed line(s) dropped, their times will be read again")
        with open(p, "w", encoding="utf-8") as f:
            for d in out:
                f.write(json.dumps(d) + "\n")
    return out


def _same_models(done: dict, det: Detector, clf: Classifier, cap: int) -> bool:
    return (isinstance(done, dict) and done.get("detector") == det.id
            and done.get("classifier") == clf.id and done.get("cap") == cap)


def _moved_regions(done: dict, geom: dict[str, str]) -> set[str]:
    """Regions whose crop is not the one the stored readings were made from.

    A reading is a reading *of a crop*: a region the fit moved (a new video, or a rectangle dragged in the
    tool) has readings of something else on disk, and they are replaced rather than added to. Regions that
    did not move keep theirs, so correcting one meld camera does not re-read the whole video.
    """
    old = done.get("geometry") or {}
    if not isinstance(old, dict):                 # unproven geometry cannot authenticate any region
        return set(geom)
    return {r for r, k in geom.items() if old.get(r) != k}


def _sparse_batches(path, cal, window, plan):
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
        if hasattr(samples, "close"):
            samples.close()


def run_read(path: str | Path, cal: Calibration, work: Path, hands: list[dict], ivs: list[Interval],
             det: Detector, clf: Classifier, *, cap: int = CAP, force: bool = False, reread: Optional[set] = None,
             log=print) -> tuple[dict, set[int]]:
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
        except (OSError, ValueError):
            prev = None
        # readings from another detector, classifier or cap are stale as a whole: the hand is read again.
        # A region that merely moved is re-read on its own.
        window = [round(t0 * FPS) / FPS, t1 + 1.0 / FPS]
        append = (prev is not None and not force and _same_models(prev, det, clf, cap)
                  and prev.get("identity") == identity and prev.get("window") == window)
        moved = _moved_regions(prev, geom) if append else set()
        fresh = asked | moved
        if prev is not None and not force and not append:
            log(f"  read hand {h['hand']:2d}: evidence inputs changed, reading again")
        elif moved and h["hand"] == hands[0]["hand"]:
            log(f"  read: the geometry of {', '.join(sorted(moved))} changed; those regions are read again")
        schedule = {r: sorted(plan[r]) for r in REGIONS}
        rows = {r: {} for r in REGIONS}
        changed = not append or bool(fresh)
        for r in REGIONS:
            old = read_lines(_region_file(hdir, r), log) if append and r not in fresh else []
            rows[r] = {round(d["t"], 3): d for d in old if round(d["t"], 3) in plan[r]}
            changed |= len(rows[r]) != len(old)
            plan[r] -= rows[r].keys()
        if not changed and not any(plan.values()) and prev.get("plan") == schedule:
            continue
        n = 0
        if any(plan.values()):
            # The decode window is part of frame identity. Narrowing a retry to
            # missing timestamps can select different frames in ffmpeg's fps filter.
            with closing(_prefetch_batches(_sparse_batches(path, cal, window, plan))) as batches:
                for batch in batches:
                    for rd in read_regions(batch, det, clf):
                        rows[rd.region][round(rd.t, 3)] = rd.to_dict()
                    n += len(batch)
        # the count is what the files hold (a re-read kind replaced its lines, it did not add to them);
        # done.json is written last so that a crash leaves the previous one and the missing times are planned again
        missing = {r: sorted(set(schedule[r]) - rows[r].keys()) for r in REGIONS
                   if set(schedule[r]) - rows[r].keys()}
        if missing:
            raise RuntimeError(f"Video ended before planned evidence for hand {h['hand']}: {missing}")
        # Removing the manifest before publishing prevents an interrupted
        # multi-region replacement from authenticating mixed old/new evidence.
        done.unlink(missing_ok=True)
        for r in REGIONS:
            _atomic_text(_region_file(hdir, r), "".join(json.dumps(rows[r][t]) + "\n" for t in sorted(rows[r])))
        total = sum(len(values) for values in rows.values())
        _atomic_text(done, json.dumps({"hand": h["hand"], "readings": total, "detector": det.id,
                     "classifier": clf.id, "cap": cap, "geometry": geom, "identity": identity,
                     "window": window, "plan": schedule}, indent=1))
        touched.add(h["hand"])
        stats["hands"] += 1
        stats["readings"] += n
        log(f"  read hand {h['hand']:2d}: {n} readings")
    return dict(stats), touched


def load_reads(work: Path, hand: int) -> dict[str, list[dict]]:
    """Load all region evidence for one hand, repairing interrupted JSONL tails."""
    hdir = work / "reads" / f"{hand:02d}"
    return {r: read_lines(_region_file(hdir, r)) for r in REGIONS}


def dense_key(det: Detector, clf: Classifier, cal: Calibration) -> str:
    """Bind dense evidence to recognition, geometry and its stage-specific retention policy."""
    return hashlib.sha256(json.dumps([det.id, clf.id, _model_metadata(clf), PREPROCESSING,
                                     geometry_key(cal),
                                     getattr(det, 'evidence_policy', DEFAULT_POLICY).fingerprint_for('dense')], sort_keys=True).encode()).hexdigest()[:20]


def clear_dense(work: Path) -> None:
    """Forget every dense read (a forced re-read starts them over as well)."""
    shutil.rmtree(work / "dense", ignore_errors=True)


def _dense_batches(path, cal, t0, t1, regions, fps):
    """Keep the existing sample order and batch boundaries while preparing crops."""
    samples = video.sample(path, fps=fps, start=max(0.0, t0), end=t1)
    pending = []
    try:
        for t, frame in samples:
            for region in regions:
                kind, _, corner = region.partition(":")
                img, _ = region_upright(frame, cal, kind, corner)
                pending.append((round(t, 2), region, img))
                if len(pending) == READ_BATCH:
                    yield pending
                    pending = []
        if pending:
            yield pending
    finally:
        if hasattr(samples, "close"):
            samples.close()


def _prefetch_batches(batches):
    """Prepare one batch ahead; inference stays on the caller's original thread.

    Only decoding and cropping overlap inference. At most one current and one
    next crop batch exist. Close/join the producer before releasing its sampler,
    including when inference fails; a producer failure propagates to the caller.
    """
    iterator = iter(batches)
    finished = object()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dense-crops")
    future = pool.submit(next, iterator, finished)
    try:
        while True:
            batch = future.result()
            if batch is finished:
                break
            future = pool.submit(next, iterator, finished)
            yield batch
    finally:
        future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        if hasattr(iterator, "close"):
            iterator.close()


def dense_reads(path: str | Path, cal: Calibration, work: Path, det: Detector, clf: Classifier,
                t0: float, t1: float, regions: list[str], *, fps: float = 5.0) -> dict[str, list[dict]]:
    """Read the given regions on every frame of [t0, t1] at `fps` (a short window where the calm reads were
    not enough), cached under work/dense by window, models and geometry. Returns region -> readings (dicts),
    structure recomputed as in observe. One CPU crop batch is prepared ahead of
    inference; sampled pixels, order and classifier batch boundaries are unchanged."""
    policy = getattr(det, 'evidence_policy', DEFAULT_POLICY)
    ddir = work / "dense"
    ddir.mkdir(parents=True, exist_ok=True)
    signature = {"window": [t0, t1], "fps": fps, "regions": sorted(regions),
                 "models_geometry": dense_key(det, clf, cal), "source": source_identity(path)}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:32]
    # Short names also work in deeply nested Windows project workspaces.
    key = f"{t0:.3f}_{t1:.3f}_{digest}.json"
    cache = ddir / key
    if cache.exists():
        try:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if (isinstance(saved, dict) and set(saved) == set(regions)
                    and all(isinstance(rows, list) and all(isinstance(row, dict)
                            and "t" in row and isinstance(row.get("boxes"), list)
                            for row in rows) for rows in saved.values())):
                return saved
        except (OSError, ValueError):
            pass  # an interrupted write is recomputed, never evidence
    out: dict[str, list[dict]] = {r: [] for r in regions}
    with closing(_prefetch_batches(_dense_batches(path, cal, t0, t1, regions, fps))) as batches:
        for batch in batches:
            for rd in read_regions(batch, det, clf):
                out[rd.region].append(rd.to_dict())
    for r in regions:
        kind = r.partition(":")[0]
        out[r] = [prepare_reading(kind, rd, stage='dense', policy=policy) for rd in out[r]]
    _atomic_text(cache, json.dumps(out))
    return out


def dense_pond_reads(path, cal, work, det, clf, t0, t1, corners, *, fps=5.0):
    """Read dense pond evidence, returning corner keys for engine callers."""
    out = dense_reads(path, cal, work, det, clf, t0, t1, [f"pond:{c}" for c in corners], fps=fps)
    return {c: out[f"pond:{c}"] for c in corners}
