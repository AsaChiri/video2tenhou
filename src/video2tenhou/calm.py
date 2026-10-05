# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 2: motion / occlusion per region at 2 fps -> calm and disturbed intervals.

Scores are computed once for the whole video and stored
(`work/<video>/calm_scores.npz`: t, regions, motion, skin); intervals are
derived from thresholds afterwards so the thresholds can be tuned without
another pass. Regions are rendered cheaply here (scale 1, one de-rotation
per frame), which is enough for motion and skin.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING
from zipfile import BadZipFile

import cv2
import numpy as np

from . import video
from .cache import source_identity
from .files import atomic_write_text
from .layout import CORNERS, Calibration

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


REGIONS = (
    [f"pond:{c}" for c in CORNERS]
    + [f"hand:{c}" for c in CORNERS]
    + [f"meld:{c}" for c in CORNERS]
)
MOTION_THR = 4.0  # mean abs gray difference between consecutive samples (0.5 s apart)
# skin fraction above the region's own baseline (its 20th percentile: static content)
SKIN_THR = 0.05
# score file version; 1 pads crops crossing the image edge with uniform black
SCORE_VERSION = 1
MIN_CALM_SAMPLES = 3  # >= 1 s at 2 fps
# read floor: no region goes longer than this (one turn cycle) without a read
MAX_GAP = 15.0
# the read floor's single frames must be this still (x MOTION_THR): a frame in motion
# shows tiles being moved, and at the end of a hand, pushed into the table
FLOOR_MOTION = 1.5


LOGGER = logging.getLogger("video2tenhou.calm")


def region_key(cal: Calibration, name: str) -> str:
    """Short hash of the geometry one region's pixels are cut from.

    That is its frame->image matrix, its size and, for a hand band, the roll the crop is
    turned by. Two calibrations with the same key give the same picture, so a score or a
    reading of one is a score or a reading of the other; a `scale: 1.0` written where
    nothing was written before is not a change.
    """
    transform, size = cal.transform(name)
    roll = cal.roll(name.partition(":")[2]) if name.startswith("hand:") else 0.0
    blob = json.dumps(
        [
            [round(float(v), 4) for v in transform.ravel()],
            list(size),
            round(float(roll), 3),
        ]
    )
    return hashlib.sha1(blob.encode(), usedforsecurity=False).hexdigest()[:12]


def geometry_key(cal: Calibration) -> str:
    """Geometry component of the calm cache identity, covering all regions at once."""
    blob = json.dumps({r: region_key(cal, r) for r in REGIONS}, sort_keys=True)
    return hashlib.sha1(blob.encode(), usedforsecurity=False).hexdigest()[:12]


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Return a binary HSV/YCrCb skin estimate for relative occlusion scoring."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m1 = cv2.inRange(hsv, (0, 30, 60), (25, 180, 255))
    m2 = cv2.inRange(ycc, (0, 135, 85), (255, 180, 135))
    return cv2.bitwise_and(m1, m2)


def cheap_regions(frame: np.ndarray, cal: Calibration) -> dict[str, np.ndarray]:
    """Crop every region at scale 1: ponds from one de-rotated square, others sliced."""
    out = {}
    derotation = cal.derotation()
    derot = cv2.warpAffine(
        frame, derotation[:2], (cal.side, cal.side), flags=cv2.INTER_LINEAR
    )
    for c, (rect, k, _) in cal.pond.items():
        img = rect.crop(derot)
        if k:
            img = np.rot90(img, k=-k)
        out[f"pond:{c}"] = img
    for c, (rect, _) in cal.hand.items():
        out[f"hand:{c}"] = rect.crop(frame)
    for c, (rect, _) in cal.meld.items():
        out[f"meld:{c}"] = rect.crop(frame)
    return out


@contextmanager
def _opencv_threads(count: int) -> Iterator[None]:
    """Use ``count`` OpenCV threads in the block, restoring the process setting.

    Model identities record the OpenCV thread count, so it must be restored.
    """
    previous = cv2.getNumThreads()
    cv2.setNumThreads(count)
    try:
        yield
    finally:
        cv2.setNumThreads(previous)


def scores(
    path: str | Path,
    cal: Calibration,
    *,
    fps: float = 2.0,
    log: Callable[[str], None] = LOGGER.info,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t[n], motion[n, R], skin[n, R]) over the video."""
    ts, mot, skn = [], [], []
    prev: dict[str, np.ndarray] = {}
    for n, (t, frame) in enumerate(video.sample(path, fps=fps), start=1):
        regs = cheap_regions(frame, cal)
        m_row, s_row = [], []
        with _opencv_threads(1):  # pool dispatch costs more than these small images
            for name in REGIONS:
                img = regs[name]
                small = cv2.resize(
                    img,
                    (max(1, img.shape[1] // 2), max(1, img.shape[0] // 2)),
                    interpolation=cv2.INTER_AREA,
                )
                gray = cv2.GaussianBlur(
                    cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0
                ).astype(np.float32)
                m = float(np.abs(gray - prev[name]).mean()) if name in prev else 0.0
                prev[name] = gray
                m_row.append(m)
                s_row.append(float((skin_mask(small) > 0).mean()))
        ts.append(t)
        mot.append(m_row)
        skn.append(s_row)
        if n % 1200 == 0:
            log(f"  calm: {t:.0f} s")
    return np.array(ts), np.array(mot, np.float32), np.array(skn, np.float32)


@dataclass
class Interval:
    """A motion-scored region span; partial spans cannot establish tile absence."""

    region: str
    t0: float
    t1: float
    n: int
    calm: bool
    motion: float
    skin: float
    # read floor: still frames with an arm over the region, or the stillest frames of a
    # blind span
    partial: bool = False

    def to_dict(self) -> dict:
        """Serialize stage-2 timing and quality fields for inspection and resume."""
        return {
            "region": self.region,
            "t0": round(self.t0, 3),
            "t1": round(self.t1, 3),
            "n": self.n,
            "calm": self.calm,
            "motion": round(self.motion, 3),
            "skin": round(self.skin, 4),
            "partial": self.partial,
        }


def _runs(values: Iterable[object]) -> Iterator[tuple[int, int, bool]]:
    """Yield half-open spans of equal values, retaining both true and false runs."""
    start = 0
    for value, group in groupby(values):
        end = start + sum(1 for _ in group)
        yield start, end, bool(value)
        start = end


@dataclass(frozen=True, kw_only=True)
class CalmThresholds:
    """Stillness, occlusion and coverage thresholds used to derive read intervals."""

    motion: float = MOTION_THR
    skin: float = SKIN_THR
    min_samples: int = MIN_CALM_SAMPLES
    max_gap: float = MAX_GAP


DEFAULT_THRESHOLDS = CalmThresholds()


def intervals(
    ts: np.ndarray,
    mot: np.ndarray,
    skn: np.ndarray,
    *,
    thresholds: CalmThresholds = DEFAULT_THRESHOLDS,
) -> list[Interval]:
    """Find maximal calm and disturbed runs per region.

    Calm runs shorter than ``thresholds.min_samples`` samples count as disturbed.
    """
    out: list[Interval] = []
    for r, name in enumerate(REGIONS):
        base = float(np.percentile(skn[:, r], 20))
        calm = (mot[:, r] < thresholds.motion) & (skn[:, r] < base + thresholds.skin)
        # a calm run shorter than min_calm is disturbed
        for i, j, is_calm in _runs(calm.copy()):
            if is_calm and j - i < thresholds.min_samples:
                calm[i:j] = False
        for i, j, is_calm in _runs(calm):
            out.append(
                Interval(
                    name,
                    float(ts[i]),
                    float(ts[j - 1]),
                    j - i,
                    calm=is_calm,
                    motion=float(mot[i:j, r].mean()),
                    skin=float(skn[i:j, r].mean()),
                )
            )
    out.sort(key=lambda iv: (iv.t0, iv.region))
    return out


@dataclass
class RegionMotion:
    """Motion and skin samples for one named region on the common time axis."""

    name: str
    ts: np.ndarray
    motion: np.ndarray
    skin: np.ndarray

    def floor(
        self,
        span: tuple[float, float],
        covered: list[tuple[float, float]],
        thresholds: CalmThresholds,
    ) -> list[Interval]:
        """Read the stillest available sample in long uncovered chunks."""
        a, b = span
        out: list[Interval] = []
        pts = [a] + [x for c in covered for x in c] + [b]
        for k in range(0, len(pts), 2):
            c0, c1 = pts[k], pts[k + 1]
            if c1 - c0 <= thresholds.max_gap:
                continue
            # one sample per half-floor keeps consecutive reads within the floor of
            # each other
            nchunk = int(np.ceil((c1 - c0) / (thresholds.max_gap / 2)))
            for q in range(nchunk):
                q0, q1 = (
                    c0 + q * (c1 - c0) / nchunk,
                    c0 + (q + 1) * (c1 - c0) / nchunk,
                )
                ss = np.where((self.ts > q0) & (self.ts < q1))[0]
                if len(ss) == 0:
                    continue
                best = int(ss[int(np.argmin(self.motion[ss]))])
                if self.motion[best] >= FLOOR_MOTION * thresholds.motion:
                    continue
                out.append(
                    Interval(
                        self.name,
                        float(self.ts[best]),
                        float(self.ts[best]),
                        1,
                        calm=True,
                        motion=float(self.motion[best]),
                        skin=float(self.skin[best]),
                        partial=True,
                    )
                )
        return out


def fill_gaps(
    ivs: list[Interval],
    ts: np.ndarray,
    mot: np.ndarray,
    skn: np.ndarray,
    *,
    thresholds: CalmThresholds = DEFAULT_THRESHOLDS,
) -> list[Interval]:
    """Add partial intervals wherever a region has no calm interval for max_gap.

    The partial intervals are runs of >= min_calm still samples (motion below the
    threshold whatever the skin score: an arm resting over part of the region), and
    where the span is still longer than max_gap, the stillest sample of each chunk when
    it is still enough (FLOOR_MOTION): a chunk with no such frame stays unread. A
    partial observation is positive evidence only: a tile seen is there, a tile absent
    may be hidden.
    """
    out = list(ivs)
    for r, name in enumerate(REGIONS):
        calm = sorted(
            (iv for iv in ivs if iv.region == name and iv.calm), key=lambda iv: iv.t0
        )
        edges = (
            [float(ts[0])]
            + [x for iv in calm for x in (iv.t0, iv.t1)]
            + [float(ts[-1])]
        )
        for a, b in [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]:
            if b - a <= thresholds.max_gap:
                continue
            sel = np.where((ts > a) & (ts < b))[0]
            if len(sel) == 0:
                continue
            still = mot[sel, r] < thresholds.motion
            covered: list[tuple[float, float]] = []
            for i, j, is_still in _runs(still):
                if is_still and j - i >= thresholds.min_samples:
                    out.append(
                        Interval(
                            name,
                            float(ts[sel[i]]),
                            float(ts[sel[j - 1]]),
                            j - i,
                            calm=True,
                            motion=float(mot[sel[i:j], r].mean()),
                            skin=float(skn[sel[i:j], r].mean()),
                            partial=True,
                        )
                    )
                    covered.append((float(ts[sel[i]]), float(ts[sel[j - 1]])))
            out.extend(
                RegionMotion(name, ts, mot[:, r], skn[:, r]).floor(
                    (a, b), covered, thresholds
                )
            )
    out.sort(key=lambda iv: (iv.t0, iv.region))
    return out


def run_calm(
    path: str | Path,
    cal: Calibration,
    work: Path,
    *,
    force: bool = False,
    log: Callable[[str], None] = LOGGER.info,
) -> list[Interval]:
    """Reuse source- and geometry-matched scores, derive intervals and write calm.jsonl.

    Threshold changes reuse saved scores. Changed source contents (see
    ``cache.source_identity``), changed region geometry, missing source provenance
    or ``force`` require another scoring pass.
    Incomplete caches are recomputed; failed scoring or publication preserves
    the previous complete score file.
    """
    work.mkdir(parents=True, exist_ok=True)
    npz = work / "calm_scores.npz"
    key = geometry_key(cal)
    source = source_identity(path)
    cached = None
    if npz.exists() and not force:
        # Scores belong to both the recording and its calibrated rectangles.
        # Caches without source provenance cannot establish that match.
        try:
            with np.load(npz, allow_pickle=False) as z:
                if (
                    int(z["version"]) == SCORE_VERSION
                    and str(z["calib"]) == key
                    and str(z["source_sha256"]) == source
                    and np.array_equal(z["regions"], np.array(REGIONS))
                ):
                    ts, mot, skn = z["t"], z["motion"], z["skin"]
                    shape = (len(ts), len(REGIONS))
                    if (
                        ts.ndim == 1
                        and len(ts) > 0
                        and mot.shape == skn.shape == shape
                        and all(np.isfinite(a).all() for a in (ts, mot, skn))
                        and np.all(np.diff(ts) > 0)
                    ):
                        cached = ts, mot, skn
        except (
            FileNotFoundError,
            ValueError,
            KeyError,
            EOFError,
            BadZipFile,
            TypeError,
            zlib.error,
        ):
            pass  # Interrupted or obsolete scores are a cache miss, never evidence.
        if cached is None:
            log(
                "  calm: scores are stale, incomplete or lack provenance; "
                "computing scores again"
            )
    if cached is None:
        ts, mot, skn = scores(path, cal, log=log)
        pending = None
        try:
            with NamedTemporaryFile(
                mode="wb", dir=work, prefix=".calm-scores-", suffix=".npz", delete=False
            ) as f:
                pending = Path(f.name)
                np.savez_compressed(
                    f,
                    version=np.array(SCORE_VERSION),
                    t=ts,
                    regions=np.array(REGIONS),
                    motion=mot,
                    skin=skn,
                    calib=np.array(key),
                    source_sha256=np.array(source),
                )
            pending.replace(npz)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
    else:
        ts, mot, skn = cached
    ivs = fill_gaps(intervals(ts, mot, skn), ts, mot, skn)
    atomic_write_text(
        work / "calm.jsonl", "".join(json.dumps(iv.to_dict()) + "\n" for iv in ivs)
    )
    return ivs


def summary(ivs: list[Interval]) -> dict:
    """Report per-region coverage and interval counts, separating partial reads."""
    out = {}
    for name in REGIONS:
        mine = [iv for iv in ivs if iv.region == name]
        calm = [iv for iv in mine if iv.calm and not iv.partial]
        total = sum(iv.n for iv in mine if not iv.partial) or 1
        out[name] = {
            "calm_fraction": round(sum(iv.n for iv in calm) / total, 3),
            "calm_intervals": len(calm),
            "median_calm_s": round(
                float(np.median([iv.t1 - iv.t0 for iv in calm])) if calm else 0.0, 1
            ),
            "partial_intervals": sum(1 for iv in mine if iv.partial),
        }
    return out
