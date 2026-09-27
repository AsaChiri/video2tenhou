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
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional
from zipfile import BadZipFile
from zlib import error as ZlibError

import cv2
import numpy as np

from . import video
from .cache import source_identity
from .layout import CORNERS, Calibration

REGIONS = [f"pond:{c}" for c in CORNERS] + [f"hand:{c}" for c in CORNERS] + [f"meld:{c}" for c in CORNERS]
MOTION_THR = 4.0        # mean abs gray difference between consecutive samples (0.5 s apart)
SKIN_THR = 0.05         # skin fraction above the region's own baseline (its 20th percentile: static content)
MIN_CALM_SAMPLES = 3    # >= 1 s at 2 fps
MAX_GAP = 15.0          # read floor: no region goes longer than this (one turn cycle) without a read
FLOOR_MOTION = 1.5      # the read floor's single frames must be this still (× MOTION_THR): a frame in motion shows
                        # tiles being moved, and at the end of a hand, pushed into the table


def region_key(cal: Calibration, name: str) -> str:
    """Short hash of the pixels one region is cut from: its frame->image matrix, its size, and, for a hand
    band, the roll the crop is turned by. Two calibrations with the same key give the same picture, so a
    score or a reading of one is a score or a reading of the other; a `scale: 1.0` written where nothing
    was written before is not a change."""
    M, size = cal.transform(name)
    roll = cal.roll(name.partition(":")[2]) if name.startswith("hand:") else 0.0
    blob = json.dumps([[round(float(v), 4) for v in M.ravel()], list(size), round(float(roll), 3)])
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def geometry_key(cal: Calibration) -> str:
    """Geometry component of the calm cache identity, covering all regions at once."""
    blob = json.dumps({r: region_key(cal, r) for r in REGIONS}, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Return a binary HSV/YCrCb skin estimate for relative occlusion scoring."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m1 = cv2.inRange(hsv, (0, 30, 60), (25, 180, 255))
    m2 = cv2.inRange(ycc, (0, 135, 85), (255, 180, 135))
    return cv2.bitwise_and(m1, m2)


def cheap_regions(frame: np.ndarray, cal: Calibration) -> dict[str, np.ndarray]:
    """Region crops at scale 1: ponds from one de-rotated square, hands and melds as slices."""
    out = {}
    D = cal.derotation()
    derot = cv2.warpAffine(frame, D[:2], (cal.side, cal.side), flags=cv2.INTER_LINEAR)
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


def scores(path: str | Path, cal: Calibration, *, fps: float = 2.0, start: float = 0.0, end: Optional[float] = None,
           log=print) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t[n], motion[n, R], skin[n, R]) over the video."""
    ts, mot, skn = [], [], []
    prev: dict[str, np.ndarray] = {}
    n = 0
    for t, frame in video.sample(path, fps=fps, start=start, end=end):
        regs = cheap_regions(frame, cal)
        m_row, s_row = [], []
        for name in REGIONS:
            img = regs[name]
            small = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
            gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
            m = float(np.abs(gray - prev[name]).mean()) if name in prev else 0.0
            prev[name] = gray
            m_row.append(m)
            s_row.append(float((skin_mask(small) > 0).mean()))
        ts.append(t); mot.append(m_row); skn.append(s_row)
        n += 1
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
    partial: bool = False   # read floor: still frames with an arm over the region, or the stillest frames of a blind span

    def to_dict(self) -> dict:
        """Serialize stage-2 timing and quality fields for inspection and resume."""
        return {"region": self.region, "t0": round(self.t0, 3), "t1": round(self.t1, 3), "n": self.n, "calm": self.calm,
                "motion": round(self.motion, 3), "skin": round(self.skin, 4), "partial": self.partial}


def intervals(ts: np.ndarray, mot: np.ndarray, skn: np.ndarray, *, motion_thr: float = MOTION_THR,
              skin_thr: float = SKIN_THR, min_calm: int = MIN_CALM_SAMPLES) -> list[Interval]:
    """Maximal runs of calm / disturbed samples per region; calm runs shorter than min_calm become disturbed."""
    out: list[Interval] = []
    for r, name in enumerate(REGIONS):
        base = float(np.percentile(skn[:, r], 20))
        calm = (mot[:, r] < motion_thr) & (skn[:, r] < base + skin_thr)
        # a calm run shorter than min_calm is disturbed
        i = 0
        while i < len(calm):
            j = i
            while j < len(calm) and calm[j] == calm[i]:
                j += 1
            if calm[i] and j - i < min_calm:
                calm[i:j] = False
            i = j
        i = 0
        while i < len(calm):
            j = i
            while j < len(calm) and calm[j] == calm[i]:
                j += 1
            out.append(Interval(name, float(ts[i]), float(ts[j - 1]), j - i, bool(calm[i]),
                                float(mot[i:j, r].mean()), float(skn[i:j, r].mean())))
            i = j
    out.sort(key=lambda iv: (iv.t0, iv.region))
    return out


def fill_gaps(ivs: list[Interval], ts: np.ndarray, mot: np.ndarray, skn: np.ndarray, *, max_gap: float = MAX_GAP,
              motion_thr: float = MOTION_THR, min_calm: int = MIN_CALM_SAMPLES) -> list[Interval]:
    """Read floor. Wherever a region has no calm interval for longer than max_gap, partial intervals are added:
    runs of >= min_calm still samples (motion below the threshold whatever the skin score: an arm resting over
    part of the region), and where the span is still longer than max_gap, the stillest sample of each chunk
    when it is still enough (FLOOR_MOTION): a chunk with no such frame stays unread.
    A partial observation is positive evidence only: a tile seen is there, a tile absent may be hidden."""
    out = list(ivs)
    for r, name in enumerate(REGIONS):
        calm = sorted((iv for iv in ivs if iv.region == name and iv.calm), key=lambda iv: iv.t0)
        edges = [float(ts[0])] + [x for iv in calm for x in (iv.t0, iv.t1)] + [float(ts[-1])]
        for a, b in [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]:
            if b - a <= max_gap:
                continue
            sel = np.where((ts > a) & (ts < b))[0]
            if len(sel) == 0:
                continue
            still = mot[sel, r] < motion_thr
            covered: list[tuple[float, float]] = []
            i = 0
            while i < len(sel):
                j = i
                while j < len(sel) and still[j] == still[i]:
                    j += 1
                if still[i] and j - i >= min_calm:
                    out.append(Interval(name, float(ts[sel[i]]), float(ts[sel[j - 1]]), j - i, True,
                                        float(mot[sel[i:j], r].mean()), float(skn[sel[i:j], r].mean()), partial=True))
                    covered.append((float(ts[sel[i]]), float(ts[sel[j - 1]])))
                i = j
            pts = [a] + [x for c in covered for x in c] + [b]
            for k in range(0, len(pts), 2):
                c0, c1 = pts[k], pts[k + 1]
                if c1 - c0 <= max_gap:
                    continue
                # one sample per half-floor keeps consecutive reads within the floor of each other
                nchunk = int(np.ceil((c1 - c0) / (max_gap / 2)))
                for q in range(nchunk):
                    q0, q1 = c0 + q * (c1 - c0) / nchunk, c0 + (q + 1) * (c1 - c0) / nchunk
                    ss = np.where((ts > q0) & (ts < q1))[0]
                    if len(ss) == 0:
                        continue
                    best = int(ss[int(np.argmin(mot[ss, r]))])
                    if mot[best, r] >= FLOOR_MOTION * motion_thr:
                        continue
                    out.append(Interval(name, float(ts[best]), float(ts[best]), 1, True, float(mot[best, r]), float(skn[best, r]),
                                        partial=True))
    out.sort(key=lambda iv: (iv.t0, iv.region))
    return out


def run_calm(path: str | Path, cal: Calibration, work: Path, *, force: bool = False, log=print) -> list[Interval]:
    """Reuse source- and geometry-matched scores, derive intervals and write calm.jsonl.

    Threshold changes reuse saved scores. Source contents are rehashed at this
    stage boundary, even if the path, size and timestamps are unchanged. Changed
    pixels, missing source provenance or ``force`` require another scoring pass.
    Incomplete caches are recomputed; failed scoring or publication preserves
    the previous complete score file.
    """
    work.mkdir(parents=True, exist_ok=True)
    npz = work / "calm_scores.npz"
    key = geometry_key(cal)
    source = source_identity(path, refresh=True)
    cached = None
    if npz.exists() and not force:
        # Scores belong to both the recording and its calibrated rectangles.
        # Caches without source provenance cannot establish that match.
        try:
            with np.load(npz, allow_pickle=False) as z:
                if (str(z["calib"]) == key and str(z["source_sha256"]) == source
                        and np.array_equal(z["regions"], np.array(REGIONS))):
                    ts, mot, skn = z["t"], z["motion"], z["skin"]
                    shape = (len(ts), len(REGIONS))
                    if (ts.ndim == 1 and len(ts) > 0 and mot.shape == skn.shape == shape
                            and all(np.isfinite(a).all() for a in (ts, mot, skn))
                            and np.all(np.diff(ts) > 0)):
                        cached = ts, mot, skn
        except (OSError, ValueError, KeyError, EOFError, BadZipFile, TypeError, ZlibError):
            pass  # Interrupted or obsolete scores are a cache miss, never evidence.
        if cached is None:
            log("  calm: scores are stale, incomplete or lack provenance; computing scores again")
    if cached is None:
        ts, mot, skn = scores(path, cal, log=log)
        pending = None
        try:
            with NamedTemporaryFile(mode="wb", dir=work, prefix=".calm-scores-", suffix=".npz",
                                    delete=False) as f:
                pending = Path(f.name)
                np.savez_compressed(f, t=ts, regions=np.array(REGIONS), motion=mot, skin=skn,
                                    calib=np.array(key), source_sha256=np.array(source))
            pending.replace(npz)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
    else:
        ts, mot, skn = cached
    ivs = fill_gaps(intervals(ts, mot, skn), ts, mot, skn)
    with open(work / "calm.jsonl", "w", encoding="utf-8") as f:
        for iv in ivs:
            f.write(json.dumps(iv.to_dict()) + "\n")
    return ivs


def summary(ivs: list[Interval]) -> dict:
    """Report per-region coverage and interval counts, separating partial reads."""
    out = {}
    for name in REGIONS:
        mine = [iv for iv in ivs if iv.region == name]
        calm = [iv for iv in mine if iv.calm and not iv.partial]
        total = sum(iv.n for iv in mine if not iv.partial) or 1
        out[name] = {"calm_fraction": round(sum(iv.n for iv in calm) / total, 3), "calm_intervals": len(calm),
                     "median_calm_s": round(float(np.median([iv.t1 - iv.t0 for iv in calm])) if calm else 0.0, 1),
                     "partial_intervals": sum(1 for iv in mine if iv.partial)}
    return out
