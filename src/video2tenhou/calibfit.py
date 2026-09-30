# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Stage 0: where the table and the camera panels are in *this* video (DESIGN.md 4.2a).

Cameras are re-mounted between broadcasts. The layout file says what the
composite and the table are; this module measures where they landed, writes
`labels/<video>/calib.json`, and checks the result with the one test that
decides whether a crop is usable:

    a region's border never cuts a tile.

A tile split by a border is the definition of a bad crop; it is what produces
a pond whose first row is missing, a pond that holds the neighbour's
discards, and a meld read from half its tiles. The check runs the real
detector on the real crops, so it measures the thing the pipeline will do.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from . import video as videomod
from .cache import source_identity
from .files import atomic_write_json
from .layout import CALIB_DIR, CORNERS, Calibration, Rect, apply_fit, fit_path
from .layout import apply as _apply

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from video2tenhou.perception.reader import RegionDetector


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

UNIT_MAX_SATURATION = 80
UNIT_MAX_BRIGHTNESS = 120
UNIT_MIN_AREA = 8000
TEMPLATE_BINARY_THRESHOLD = 127
MIN_FIT_STEP = 0.02
BORDER_DETECTION_CONFIDENCE = 0.4
MIN_HAND_ROW_TILES = 6
MIN_TILE_LINE_BOXES = 4
MIN_LINE_SPAN_PX = 40
MAX_LINE_ANGLE = 90
MIN_ROLL_ESTIMATES = 3
MIN_EDGE_PEAK_SEPARATION = 6
MIN_REPEATED_BORDER_CUTS = 2
SPARSE_BORDER_TILE_COUNT = 10


PLATE_FRAMES = 60  # frames of the median plate: enough to average tiles and arms away
CHECK_FRAMES = 8  # frames of the border check
CUT_FRACTION = 0.15  # a box with this much on both sides of a border is cut by it
CUT_RATE_FAIL = (
    0.08  # share of cut boxes above which a region fails (with at least two cut)
)
CUT_RATE_HARD = 0.34  # a share this bad fails on one cut alone
MIN_UNIT_IOU = (
    0.6  # an overhead fit whose unit match is below this is a failure, not a number
)
GHOST_SAT, GHOST_VAL = (
    70,
    140,
)  # a discard's ghost on the plate: white (low saturation, bright)
# px² in the de-rotated overhead: a smaller bright component is not a pond's block
GHOST_MIN_AREA = 400
UNIT_TEMPLATE = CALIB_DIR / "pml_unit.png"
GROW = {
    "pond": 30,
    "meld": 40,
    "hand": 40,
}  # px of margin the check looks at outside each region


# -- the table plate ------------------------------------------------------------------


def plate_path(work: Path) -> Path:
    """Location of the cached median table image inside one video workspace."""
    return work / "plate.png"


def table_plate(
    video: Path,
    work: Path,
    n: int = PLATE_FRAMES,
    *,
    force: bool = False,
) -> np.ndarray:
    """Estimate the stationary table from the median of sampled video frames.

    The median of `n` frames spread over the video: the table without the tiles, the
    hands or the players.

    Everything that moves averages away; the felt, the centre unit and the hard borders
    of every panel the broadcast pastes into the composite stay, and the faint white
    ghosts inside the overhead are where discards live over the whole video.
    """
    p = plate_path(work)
    source = Path(video).resolve()
    stat = source.stat()
    signature = {
        "version": 1,
        "source": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "samples": n,
        "sha256": source_identity(source),
        "frame": [videomod.FRAME_W, videomod.FRAME_H],
    }
    manifest = work / "plate.meta.json"
    try:
        matching = json.loads(manifest.read_text(encoding="utf-8")) == signature
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        matching = False
    if p.exists() and matching and not force:
        img = cv2.imread(str(p))
        if img is not None:
            return img
    info = videomod.probe(str(video))
    a = 0.05 * info.duration
    b = 0.95 * info.duration
    frames = []
    for t in np.linspace(a, b, n):
        f = videomod.frame_at(str(video), float(t))
        if f is not None:
            frames.append(f)
    if not frames:
        msg = f"no frame could be read from {video}"
        raise RuntimeError(msg)
    img = np.median(np.stack(frames), 0).astype(np.uint8)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)
    manifest.write_text(json.dumps(signature), encoding="utf-8")
    return img


# -- the overhead fit -----------------------------------------------------------------


def unit_mask(
    plate: np.ndarray,
    center: tuple[float, float] = (960.0, 540.0),
    radius: float = 260.0,
) -> np.ndarray:
    """Find the dark, unsaturated centre unit in the overhead frame.

    The centre unit of the table: the dark, unsaturated block near the middle of the
    overhead.

    The overhead shows teal felt and white tiles; the unit is the only dark thing in it,
    which is why it is the anchor the fit uses.
    """
    hsv = cv2.cvtColor(plate, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    dark = ((s < UNIT_MAX_SATURATION) & (v < UNIT_MAX_BRIGHTNESS)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((21, 21), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(dark, connectivity=8)
    best, best_area = 0, 0
    for i in range(1, n):
        if st[i, 4] < UNIT_MIN_AREA:
            continue
        if abs(ce[i][0] - center[0]) > radius or abs(ce[i][1] - center[1]) > radius:
            continue
        if st[i, 4] > best_area:
            best, best_area = i, int(st[i, 4])
    if not best:
        msg = (
            "no centre unit found near the middle of the frame: is this the right "
            "layout?"
        )
        raise RuntimeError(msg)
    return (lab == best).astype(np.uint8)


def derotation_of(
    cal: Calibration, center: Sequence[float] | np.ndarray, angle: float, scale: float
) -> np.ndarray:
    """Build a candidate frame-to-overhead transform without changing calibration."""
    rotation = np.vstack(
        [
            cv2.getRotationMatrix2D(
                (float(center[0]), float(center[1])), float(angle), float(scale)
            ),
            [0, 0, 1],
        ]
    )
    h = cal.side / 2.0
    translation = np.array(
        [[1, 0, -(center[0] - h)], [0, 1, -(center[1] - h)], [0, 0, 1]], np.float64
    )
    return translation @ rotation


def unit_template(cal: Calibration) -> np.ndarray:
    """Return the reference centre-unit mask in overhead coordinates.

    The unit as the layout knows it, in overhead coordinates: the shape a new video is
    fitted to.

    The stored mask was cut from the reference plate through the reference fit; without
    it the unit rectangle itself is the template, which is a coarser but still workable
    target.
    """
    if UNIT_TEMPLATE.exists():
        m = cv2.imread(str(UNIT_TEMPLATE), cv2.IMREAD_GRAYSCALE)
        if m is not None and m.shape == (cal.side, cal.side):
            return (m > TEMPLATE_BINARY_THRESHOLD).astype(np.uint8)
    t = np.zeros((cal.side, cal.side), np.uint8)
    u = cal.unit
    t[u.y : u.y + u.h, u.x : u.x + u.w] = 1
    return t


def _refine_overhead(
    cur: np.ndarray, val: float, iou: Callable[[np.ndarray, float, float], float]
) -> tuple[np.ndarray, float]:
    """Refine the coarse centre-unit fit by bounded coordinate hill climbing."""
    step = np.array([0.25, 0.01, 1.5, 1.5])
    for _ in range(80):  # hill climb on the four numbers
        moved = False
        for i in range(4):
            for sg in (1, -1):
                q = cur.copy()
                q[i] += sg * step[i]
                v = iou(q[2:], q[0], q[1])
                if v > val + 1e-6:
                    cur, val, moved = q, v, True
        if not moved:
            step = step / 2.0
            if step[0] < MIN_FIT_STEP:
                break
    return cur, val


def fit_overhead(plate: np.ndarray, cal: Calibration) -> dict:
    """Fit overhead centre, angle and scale by matching the centre unit.

    (centre, angle, scale) of the overhead in this video, by matching the unit to the
    template.
    """
    tmpl = unit_template(cal)
    mask = unit_mask(plate)
    side = cal.side
    tc = np.array(np.nonzero(tmpl)).mean(1)[::-1]  # template centroid, (x, y)
    m = cv2.moments(mask, binaryImage=True)
    mc = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]])

    def centre_for(angle: float, scale: float) -> np.ndarray:
        """Find the frame centre that aligns the template and observed centroids.

        The frame point that must sit at the middle of the overhead for the centroids to
        coincide.
        """
        a = np.radians(angle)
        rotation = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        return mc - np.linalg.inv(scale * rotation) @ (
            tc - np.array([side / 2.0, side / 2.0])
        )

    def iou(center: Sequence[float] | np.ndarray, angle: float, scale: float) -> float:
        w = cv2.warpAffine(
            mask,
            derotation_of(cal, center, angle, scale)[:2],
            (side, side),
            flags=cv2.INTER_NEAREST,
        )
        u = int((w | tmpl).sum())
        return float((w & tmpl).sum()) / u if u else 0.0

    best = (-1.0, cal.angle, 1.0, np.array(cal.center, float))
    for angle in np.arange(cal.angle - 5.0, cal.angle + 5.01, 0.5):
        for scale in np.arange(0.80, 1.2501, 0.02):
            c = centre_for(float(angle), float(scale))
            v = iou(c, float(angle), float(scale))
            if v > best[0]:
                best = (v, float(angle), float(scale), c)
    val, angle, scale, c = best
    cur = np.array([angle, scale, c[0], c[1]], float)
    cur, val = _refine_overhead(cur, val, iou)
    return {
        "center": [round(float(cur[2]), 2), round(float(cur[3]), 2)],
        "angle": round(float(cur[0]), 3),
        "scale": round(float(cur[1]), 4),
        "iou": round(float(val), 4),
    }


# -- the hand band --------------------------------------------------------------------


def tile_rows(
    dets: list, tol_frac: float = 0.5, max_rows: int = 4
) -> list[tuple[list, float, float]]:
    """Group face-up boxes into collinear rows and estimate row geometry.

    Face-up boxes grouped into collinear rows: [(boxes, angle in degrees, median box
    area)].

    A corner camera shows its player's hand (face up, nearest, largest), the walls (face
    down) and other players' tiles (far, small). Rows come out of a repeated best-line
    search, so the caller can pick the one the physics names: the biggest tiles are the
    nearest, and the nearest row is the player's own hand.
    """
    b = [d for d in dets if not d.back and d.conf >= BORDER_DETECTION_CONFIDENCE]
    if len(b) < MIN_TILE_LINE_BOXES:
        return []
    c = np.array(
        [[(d.xyxy[0] + d.xyxy[2]) / 2, (d.xyxy[1] + d.xyxy[3]) / 2] for d in b]
    )
    hgt = np.array([d.xyxy[3] - d.xyxy[1] for d in b])
    area = np.array([(d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1]) for d in b])
    out: list[tuple[list, float, float]] = []
    used = np.zeros(len(b), bool)
    for _ in range(max_rows):
        idx = np.nonzero(~used)[0]
        if len(idx) < MIN_TILE_LINE_BOXES:
            break
        tol = tol_frac * float(np.median(hgt[idx]))
        best = None
        for i in idx:
            for j in idx:
                if j <= i:
                    continue
                d = c[j] - c[i]
                n = float(np.hypot(*d))
                if n < MIN_LINE_SPAN_PX:
                    continue
                nrm = np.array([-d[1], d[0]]) / n
                inl = idx[np.abs((c[idx] - c[i]) @ nrm) < tol]
                if best is None or len(inl) > len(best):
                    best = inl
        if best is None or len(best) < MIN_TILE_LINE_BOXES:
            break
        used[best] = True
        vx, vy = cv2.fitLine(
            c[best].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        ).ravel()[:2]
        ang = float(np.degrees(np.arctan2(float(vy), float(vx))))
        ang -= 180 if ang > MAX_LINE_ANGLE else (-180 if ang < -MAX_LINE_ANGLE else 0)
        out.append(([b[k] for k in best], ang, float(np.median(area[best]))))
    return out


def fit_hand(
    video: Path,
    cal: Calibration,
    corner: str,
    det: RegionDetector,
    times: Iterable[float],
) -> dict | None:
    """Fit one player's hand band and roll from observed tile rows.

    One player's hand band and its roll, from the row of their own tiles in their corner
    camera.

    The player's hand is the face-up row nearest the camera, so its tiles are the
    largest in the picture; the wall is face down and everyone else's tiles are far. The
    band is the envelope of that row over the sampled frames, grown by a tile (a drawn
    tile is held apart from the row, and melds move beside it at the reveal).
    """
    rect = cal.cam[corner]
    boxes: list[tuple[float, float, float, float]] = []
    angles, heights = [], []
    for t in times:
        frame = videomod.frame_at(str(video), float(t))
        if frame is None:
            continue
        img, _ = cal.region(frame, f"cam:{corner}")
        rows = tile_rows(det.predict(img))
        if not rows:
            continue
        rows.sort(key=lambda r: -r[2])
        row, ang, _ = rows[0]
        if len(row) < MIN_HAND_ROW_TILES:
            continue
        angles.append(ang)
        for d in row:
            boxes.append(d.xyxy)
            heights.append(d.xyxy[3] - d.xyxy[1])
    if len(angles) < MIN_ROLL_ESTIMATES:
        return None
    a = np.array(boxes)
    pad = 0.7 * float(np.median(heights))
    x0 = max(0.0, a[:, 0].min() - pad) + rect.x
    y0 = max(0.0, a[:, 1].min() - pad) + rect.y
    x1 = min(float(rect.w), a[:, 2].max() + pad) + rect.x
    y1 = min(float(rect.h), a[:, 3].max() + pad) + rect.y
    roll = float(np.median(angles))
    return {
        "rect": [
            round(x0),
            round(y0),
            round(x1 - x0),
            round(y1 - y0),
        ],
        "roll": round(roll, 2),
        "n": len(angles),
        "spread": round(float(np.median(np.abs(np.array(angles) - roll))), 2),
    }


# -- the meld inset
# ---------------------------------------------------------------------


def _edge_candidates(
    e: np.ndarray, off: int, hard: tuple[int, ...], keep: int = 22
) -> list[int]:
    """Choose separated gradient peaks and retain actual frame boundaries."""
    out: list[int] = []
    for i in sorted(np.argsort(-e)[:150]):
        if out and i - out[-1] < MIN_EDGE_PEAK_SEPARATION:
            if e[i] > e[out[-1]]:
                out[-1] = int(i)
        else:
            out.append(int(i))
    got = {off + i for i in sorted(out, key=lambda i: -e[i])[:keep]}
    # the picture's own edge is a border with no gradient behind it: an inset flush
    # with the frame
    # (both bottom meld cameras are) has no line there to find
    got |= {v for v in hard if off <= v <= off + len(e) - 1}
    return sorted(got)


def _panel_rectangles(
    cx: list[int], cy: list[int], rect: Rect, size: tuple[int, int, int, int]
) -> Iterator[tuple[int, int, int, int]]:
    """Enumerate ordered border pairs with plausible size and layout overlap."""
    x, y, w, h = rect.x, rect.y, rect.w, rect.h
    wmin, wmax, hmin, hmax = size
    for i, x0 in enumerate(cx):
        for x1 in cx[i + 1 :]:
            if not wmin <= x1 - x0 <= wmax:
                continue
            for j, y0 in enumerate(cy):
                for y1 in cy[j + 1 :]:
                    if not hmin <= y1 - y0 <= hmax:
                        continue
                    ov = max(0, min(x1, x + w) - max(x0, x)) * max(
                        0, min(y1, y + h) - max(y0, y)
                    )
                    if ov < 0.25 * w * h:  # the panel has not moved to another quadrant
                        continue
                    yield x0, y0, x1, y1


def fit_panel(
    plate: np.ndarray,
    rect: Rect,
) -> tuple[int, int, int, int] | None:
    """Find the panel rectangle nearest the layout's expected position.

    The rectangle of the composite panel nearest `rect`, from the hard borders it has on
    the plate.

    A pasted panel has four straight borders that are strong along their *whole* length
    (hence the percentile, not the mean: half an edge is some object on the table) and
    an interior that is quiet compared to them. Teal-on-teal borders have no gradient at
    all, so this is a suggestion the tool shows for confirmation, never an answer
    (DESIGN.md 4.2a).
    """
    win, q, lam = 130, 25, 0.4
    size = (90, 300, 100, 330)
    x, y, w, h = rect.x, rect.y, rect.w, rect.h
    image_height, image_width = plate.shape[:2]
    search_left, search_top = max(0, x - win), max(0, y - win)
    search_right, search_bottom = (
        min(image_width, x + w + win),
        min(image_height, y + h + win),
    )
    g = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sx, sy = cv2.Scharr(g, cv2.CV_32F, 1, 0), cv2.Scharr(g, cv2.CV_32F, 0, 1)
    gx = cv2.dilate(np.abs(sx), np.ones((1, 3), np.uint8))
    gy = cv2.dilate(np.abs(sy), np.ones((3, 1), np.uint8))
    gradient_integral = cv2.integral(cv2.magnitude(sx, sy))

    def interior(a: int, b: int, c: int, d: int) -> float:
        if c <= a or d <= b:
            return 1e9
        return float(
            gradient_integral[d, c]
            - gradient_integral[b, c]
            - gradient_integral[d, a]
            + gradient_integral[b, a]
        ) / ((c - a) * (d - b))

    cx = _edge_candidates(
        gx[search_top:search_bottom, search_left:search_right].mean(0),
        search_left,
        (0, image_width - 1),
    )
    cy = _edge_candidates(
        gy[search_top:search_bottom, search_left:search_right].mean(1),
        search_top,
        (0, image_height - 1),
    )
    frame_edges = {0, image_width - 1, image_height - 1}
    best = None
    for x0, y0, x1, y1 in _panel_rectangles(cx, cy, rect, size):
        es = [
            np.inf if v in frame_edges else float(np.percentile(arr, q))
            for v, arr in (
                (x0, gx[y0:y1, x0]),
                (x1, gx[y0:y1, min(x1, image_width - 1)]),
                (y0, gy[y0, x0:x1]),
                (y1, gy[min(y1, image_height - 1), x0:x1]),
            )
        ]
        edge = min(es)
        if not np.isfinite(edge):  # a rectangle made only of frame edges is no panel
            continue
        sc = edge - lam * interior(x0 + 5, y0 + 5, x1 - 5, y1 - 5)
        if best is None or sc > best[0]:
            best = (sc, x0, y0, x1 - x0, y1 - y0)
    return None if best is None else (best[1], best[2], best[3], best[4])


# -- the check: a region's border never cuts a tile
# -------------------------------------


@dataclass
class RegionCheck:
    """Border evidence and verdict; warnings permit analysis, failures block it."""

    region: str
    level: str = "ok"  # ok | warn | fail
    held: int = 0
    cut: int = 0
    frames: int = 0
    note: str = ""
    foreign: int = 0  # ponds: boxes held that lie in another pond's block

    @property
    def ok(self) -> bool:
        """Report usability, including warnings that merit visual inspection."""
        return self.level != "fail"

    @property
    def rate(self) -> float:
        """Fraction of detected tiles intersecting a border among held and cut tiles."""
        return self.cut / max(1, self.cut + self.held)

    def line(self) -> str:
        """Format a region verdict with counts and corrective guidance.

        Human-readable verdict with counts and the corrective diagnostic, when
        available.
        """
        v = {"ok": "ok   ", "warn": "warn ", "fail": "FAIL "}[self.level]
        if self.region == "overhead":
            return f"  {v} {self.region:12s} {self.note}"
        foreign = f", {self.foreign} of another pond" if self.foreign else ""
        return (
            f"  {v} {self.region:12s} {self.held:4d} tiles held, {self.cut:3d} "
            f"cut by the border{foreign}"
        ) + (f"  ({self.note})" if self.note else "")

    def verdict(self) -> None:
        """Set the region verdict from its observed tile and clipping counts.

        Ok / warn / fail from the counts. One cut tile in a busy region is a stray box;
        a quarter of them cut is a region in the wrong place.
        """
        kind = self.region.partition(":")[0]
        if self.foreign:
            self.level = "fail"
            self.note = (
                f"{self.foreign} tile(s) inside it lie in another pond's block: it "
                "reads a neighbour's discards"
            )
        elif self.cut and (
            self.rate >= CUT_RATE_HARD
            or (self.cut >= MIN_REPEATED_BORDER_CUTS and self.rate > CUT_RATE_FAIL)
        ):
            self.level = "fail"
            self.note = (
                f"{self.rate:.0%} of the tiles at this border are cut by it: the "
                "region is misplaced"
            )
        elif self.held == 0 and kind != "meld":
            self.level = "fail"
            self.note = (
                f"no tile in {self.frames} frames of play: the region is not "
                "looking at the table"
            )
        elif self.cut >= MIN_REPEATED_BORDER_CUTS or (
            self.cut and self.held < SPARSE_BORDER_TILE_COUNT
        ):
            self.level = "warn"
            self.note = f"{self.rate:.0%} cut: look at this region in the tool"
        elif self.held == 0:
            self.level = "warn"
            self.note = (
                "no meld in the sampled frames (calls are rare): nothing to check, "
                "look at it in the tool"
            )


def grown(
    cal: Calibration, name: str, margin: int
) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[int, int]]:
    """Return a region's expanded transform, inner bounds and image size.

    The region grown by `margin`: its frame->image matrix, the true region's box in that
    image, its size.

    The check looks at a margin outside the region so a tile the border cuts is seen
    whole.
    """
    kind, _, corner = name.partition(":")
    if kind == "pond":
        rect, k, s = cal.pond[corner]
        big = Rect(
            rect.x - margin, rect.y - margin, rect.w + 2 * margin, rect.h + 2 * margin
        )
        crop, size = cal.crop_rot_scale(
            big, k, s
        )  # overhead coordinates -> the grown image
        transform = crop @ cal.derotation()
        q = _apply(crop, [[rect.x, rect.y], [rect.x + rect.w, rect.y + rect.h]])
    else:
        rect, s = (
            cal.hand[corner]
            if kind == "hand"
            else (cal.meld[corner] if kind == "meld" else (cal.cam[corner], 1.0))
        )
        big = Rect(
            rect.x - margin, rect.y - margin, rect.w + 2 * margin, rect.h + 2 * margin
        )
        transform, size = cal.crop_rot_scale(big, 0, s)
        q = _apply(transform, [[rect.x, rect.y], [rect.x + rect.w, rect.y + rect.h]])
    inner = (
        float(q[:, 0].min()),
        float(q[:, 1].min()),
        float(q[:, 0].max()),
        float(q[:, 1].max()),
    )
    return transform, inner, size


def render(
    frame: np.ndarray, transform: np.ndarray, size: tuple[int, int]
) -> np.ndarray:
    """Warp BGR pixels with a frame-to-region matrix and output dimensions."""
    return cv2.warpAffine(frame, transform[:2], size, flags=cv2.INTER_CUBIC)


def _cut(box: Sequence[float], inner: Sequence[float]) -> bool | None:
    """Classify a box as cut, inside or outside the region border.

    True when the border cuts the box, False when it is wholly inside, None when wholly
    outside.
    """
    x0, y0, x1, y1 = box
    a = max(0.0, min(x1, inner[2]) - max(x0, inner[0])) * max(
        0.0, min(y1, inner[3]) - max(y0, inner[1])
    )
    area = max(1.0, (x1 - x0) * (y1 - y0))
    f = a / area
    if f <= CUT_FRACTION:
        return None
    return f < 1.0 - CUT_FRACTION


def pond_blocks(
    plate: np.ndarray, cal: Calibration
) -> tuple[np.ndarray, dict[int, str]]:
    """Assign persistent overhead components to their owning ponds.

    Where each pond's discards lie over the whole video: the bright ghosts of the plate,
    in the de-rotated overhead, as connected components (the centre unit left out), each
    owned by the pond rectangle that holds most of it. Returns (label image, component
    -> corner).
    """
    oh = cv2.warpAffine(
        plate, cal.derotation()[:2], (cal.side, cal.side), flags=cv2.INTER_LINEAR
    )
    hsv = cv2.cvtColor(oh, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] < GHOST_SAT) & (hsv[..., 2] > GHOST_VAL)).astype(np.uint8)
    u = cal.unit
    mask[max(0, u.y) : u.y + u.h, max(0, u.x) : u.x + u.w] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    owner: dict[int, str] = {}
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < GHOST_MIN_AREA:
            continue
        comp = labels == i
        inside = {
            c: int(comp[r.y : r.y + r.h, r.x : r.x + r.w].sum())
            for c, (r, _, _) in cal.pond.items()
        }
        best = max(inside, key=inside.__getitem__)
        if inside[best]:
            owner[i] = best
    return labels, owner


def overhead_check(cal: Calibration) -> RegionCheck:
    """Reject an overhead fit whose centre-unit overlap is too low.

    The overhead fit itself: a unit match under MIN_UNIT_IOU is a failure (4.2a).
    """
    iou = ((cal.fit or {}).get("overhead") or {}).get("iou")
    rc = RegionCheck("overhead")
    if iou is not None and iou < MIN_UNIT_IOU:
        rc.level, rc.note = (
            "fail",
            (
                f"the centre unit matches the layout at IoU {iou:.2f} only: the fit"
                " is not reliable"
            ),
        )
    elif iou is not None:
        rc.note = f"centre unit matched at IoU {iou:.2f}"
    return rc


def check_all(
    video: Path, cal: Calibration, det: RegionDetector, work: Path
) -> list[RegionCheck]:
    """Check fitted geometry on table tiles and neighbouring pond blocks."""
    hands = prepare_table_samples(video, cal, det)
    blocks = pond_blocks(table_plate(video, work), cal)
    return ([overhead_check(cal)] if cal.fit else []) + check_regions(
        video, cal, det, hands, options=BorderOptions(blocks=blocks)
    )


@dataclass(frozen=True, kw_only=True)
class BorderOptions:
    """Optional region selection and independently measured pond ownership."""

    names: list[str] | None = None
    blocks: tuple[np.ndarray, dict[int, str]] | None = None


def check_regions(
    video: Path,
    cal: Calibration,
    det: RegionDetector,
    hands: list[dict] | None = None,
    *,
    options: BorderOptions | None = None,
) -> list[RegionCheck]:
    """Count clipped tiles and tiles belonging to neighbouring ponds.

    Run the detector on each region grown by a margin and count the tiles its border
    cuts; with the plate's blocks, also the tiles a pond holds that belong to another
    pond.
    """
    options = options or BorderOptions()
    names, blocks = options.names, options.blocks
    times = check_times(video, hands, CHECK_FRAMES)
    names = names or (
        [f"pond:{c}" for c in CORNERS]
        + [f"meld:{c}" for c in CORNERS]
        + [f"hand:{c}" for c in CORNERS]
    )
    out = {n: RegionCheck(n, frames=len(times)) for n in names}
    for t in times:
        frame = videomod.frame_at(str(video), float(t))
        if frame is None:
            continue
        for n in names:
            kind = n.partition(":")[0]
            transform, inner, size = grown(cal, n, GROW[kind])
            img = render(frame, transform, size)
            to_overhead = (
                cal.derotation() @ np.linalg.inv(transform)
                if kind == "pond" and blocks is not None
                else None
            )
            for d in det.predict(img):
                if d.conf < BORDER_DETECTION_CONFIDENCE:
                    continue
                c = _cut(d.xyxy, inner)
                if c is True:
                    out[n].cut += 1
                elif c is False:
                    out[n].held += 1
                    if (
                        blocks is not None
                        and to_overhead is not None
                        and _owner(blocks, to_overhead, d.xyxy)
                        not in (None, n.partition(":")[2])
                    ):
                        out[n].foreign += 1
    for rc in out.values():
        rc.verdict()
    return list(out.values())


def _owner(
    blocks: tuple[np.ndarray, dict[int, str]],
    to_overhead: np.ndarray,
    box: Sequence[float],
) -> str | None:
    """Return the pond owning a box centre, or None outside every block.

    The pond whose block holds the centre of a box (region pixels), None outside every
    block.
    """
    labels, owner = blocks
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    ox, oy = _apply(to_overhead, [[x, y]])[0]
    if not (0 <= oy < labels.shape[0] and 0 <= ox < labels.shape[1]):
        return None
    return owner.get(int(labels[int(oy), int(ox)]))


def check_times(video: Path, hands: list[dict] | None, k: int) -> list[float]:
    """Sample hand interiors, using coarse inspection when no windows are given."""
    if hands:
        picks = np.linspace(0, len(hands) - 1, min(k, len(hands)))
        return [
            float(
                hands[round(i)]["t_start"]
                + 0.55 * (hands[round(i)]["t_end"] - hands[round(i)]["t_start"])
            )
            for i in picks
        ]
    info = videomod.probe(str(video))
    return [float(t) for t in np.linspace(0.15 * info.duration, 0.9 * info.duration, k)]


# -- orchestration ---------------------------------------------------------------------


def prepare_table_samples(
    video: Path, cal: Calibration, det: RegionDetector, k: int = 24
) -> list[dict]:
    """Select tile-bearing table frames for geometry checks, independently of overlays.

    These are point samples, not inferred game boundaries. Inspect the whole
    overhead instead of requiring already-correct hand, pond or meld crops.
    Broadcast text and site records have no role in measuring crop borders.
    """
    samples = []
    for t in check_times(video, None, k):
        frame = videomod.frame_at(str(video), t)
        if frame is None:
            continue
        overhead, _ = cal.region(frame, "overhead")
        tiles = [
            d
            for d in det.predict(overhead)
            if d.conf >= BORDER_DETECTION_CONFIDENCE and not d.back
        ]
        if len(tiles) >= MIN_TILE_LINE_BOXES:
            samples.append({"t_start": t, "t_end": t})
    if not samples:
        msg = (
            "No table tiles found in the sampled frames. Open Settings -> "
            "Calibration to adjust the table position, then prepare the recording "
            "again."
        )
        raise RuntimeError(msg)
    return samples


def _is_human(fit: dict, part: str, corner: str) -> bool:
    return ((fit.get(part) or {}).get(corner) or {}).get("source") == "human"


LEVEL_RANK = {"ok": 0, "warn": 1, "fail": 2}


def _better(a: RegionCheck, b: RegionCheck) -> bool:
    """Compare crop verdicts, then visible tile counts and clipped tiles.

    Is check `a` a better crop than `b`? The verdict first, then more tiles held, then
    fewer cut: a rectangle that sees nothing cuts nothing, and must not win for that.
    """
    return (LEVEL_RANK[a.level], -a.held, a.cut) < (LEVEL_RANK[b.level], -b.held, b.cut)


def _expand_cut_meld(
    video: Path, cal: Calibration, corner: str, det: RegionDetector, hands: list[dict]
) -> list[int] | None:
    """Propose a bounded expansion from whole boxes cut by an automatic inset fit.

    Plate edges can mistake a line inside an inset for its outer border. Only
    boxes crossing the current border contribute; neighbouring tiles wholly
    outside it cannot pull the crop outward. The caller must validate the new
    crop with the ordinary border check before accepting it.
    """
    name = f"meld:{corner}"
    rect, _ = cal.meld[corner]
    bounds = np.array(rect.xyxy, dtype=float)
    transform, inner, size = grown(cal, name, GROW["meld"])
    inverse = np.linalg.inv(transform)
    changed = False
    for t in check_times(video, hands, CHECK_FRAMES):
        frame = videomod.frame_at(str(video), t)
        for detection in det.predict(render(frame, transform, size)):
            if (
                detection.conf < BORDER_DETECTION_CONFIDENCE
                or _cut(detection.xyxy, inner) is not True
            ):
                continue
            x0, y0, x1, y1 = detection.xyxy
            points = _apply(inverse, [[x0, y0], [x1, y1]])
            bounds[:2] = np.minimum(bounds[:2], points.min(0) - 4)
            bounds[2:] = np.maximum(bounds[2:], points.max(0) + 4)
            changed = True
    if not changed:
        return None
    margin = GROW["meld"]
    bounds[:2] = np.maximum(
        bounds[:2], [max(0, rect.x - margin), max(0, rect.y - margin)]
    )
    bounds[2:] = np.minimum(
        bounds[2:],
        [
            min(cal.frame[0], rect.x + rect.w + margin),
            min(cal.frame[1], rect.y + rect.h + margin),
        ],
    )
    x0, y0 = np.floor(bounds[:2]).astype(int)
    x1, y1 = np.ceil(bounds[2:]).astype(int)
    return [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]


@dataclass(frozen=True, kw_only=True)
class FitOptions:
    """Control cache refresh and retain explicitly selected geometry parts."""

    force: bool = False
    keep: Iterable[str] = ()


@dataclass
class GeometryFitter:
    """Measure camera geometry while retaining human edits and accepted fits."""

    video: Path
    cal: Calibration
    det: RegionDetector | None
    hands: list[dict]
    fit: dict

    def measure_hands(self) -> None:
        """Fit unreviewed hand bands from repeated tile rows."""
        det = self.det
        if det is None:
            return
        times = check_times(self.video, self.hands, 10)
        cal2 = Calibration(apply_fit(self.cal.data, self.fit))
        for c in CORNERS:
            if _is_human(
                self.fit, "hand", c
            ):  # a band a person drew is not overwritten by a measurement
                continue
            r = fit_hand(self.video, cal2, c, det, times)
            if r:
                self.fit.setdefault("hand", {})[c] = {
                    "rect": r["rect"],
                    "roll": r["roll"],
                }

    def measure_melds(self, plate: np.ndarray) -> None:
        """Accept proposed panels only when their border evidence improves."""
        det = self.det
        for c in CORNERS:
            if _is_human(self.fit, "meld", c):
                continue
            r = fit_panel(plate, self.cal.meld[c][0])
            if not r:
                continue
            cand = {
                **self.fit,
                "meld": {**self.fit.get("meld", {}), c: {"rect": [int(v) for v in r]}},
            }
            if det is None:
                self.fit = cand
                continue
            name = f"meld:{c}"
            now = check_regions(
                self.video,
                Calibration(apply_fit(self.cal.data, self.fit)),
                det,
                self.hands,
                options=BorderOptions(names=[name]),
            )[0]
            new_ = check_regions(
                self.video,
                Calibration(apply_fit(self.cal.data, cand)),
                det,
                self.hands,
                options=BorderOptions(names=[name]),
            )[0]
            if _better(new_, now):
                self.fit = cand
                now = new_
            if now.level == "fail" and now.cut:
                measured = Calibration(apply_fit(self.cal.data, self.fit))
                expanded = _expand_cut_meld(self.video, measured, c, det, self.hands)
                if expanded:
                    proposal = {
                        **self.fit,
                        "meld": {**self.fit.get("meld", {}), c: {"rect": expanded}},
                    }
                    check = check_regions(
                        self.video,
                        Calibration(apply_fit(self.cal.data, proposal)),
                        det,
                        self.hands,
                        options=BorderOptions(names=[name]),
                    )[0]
                    if _better(check, now):
                        self.fit = proposal


def run_fit(
    video: Path,
    cal: Calibration,
    work: Path,
    det: RegionDetector | None = None,
    *,
    options: FitOptions | None = None,
) -> dict:
    """Measure this video's geometry and write `labels/<video>/calib.json`.

    `options.keep` names parts a human already set in the tool. A suggested panel
    rectangle replaces the one in hand only when the border check
    prefers it: the plate's borders are a guess, the check is a measurement, and a guess
    that reads worse is not an improvement (DESIGN.md 4.2a).
    """
    options = options or FitOptions()
    work.mkdir(parents=True, exist_ok=True)
    plate = table_plate(video, work, force=options.force)
    old = cal.fit or {}
    fit: dict = {
        "video": Path(video).stem,
        "layout": cal.name,
        "source": "calibfit",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    fit["overhead"] = (
        old["overhead"]
        if old.get("overhead")
        and ("overhead" in options.keep or old["overhead"].get("source") == "human")
        else fit_overhead(plate, cal)
    )
    for part in ("cam", "hand", "meld"):
        if old.get(part):
            fit[part] = deepcopy(old[part])
    hands = (
        prepare_table_samples(video, Calibration(apply_fit(cal.data, fit)), det)
        if det is not None
        else []
    )
    fitter = GeometryFitter(video, cal, det, hands, fit)
    if "hand" not in options.keep:
        fitter.measure_hands()
    if "meld" not in options.keep:
        fitter.measure_melds(plate)
    fit = fitter.fit
    p = fit_path(video)
    atomic_write_json(p, fit, indent=1, retry_windows=True)
    return fit


def fit_sheet(
    video: Path, cal: Calibration, work: Path, checks: list[RegionCheck] | None = None
) -> np.ndarray:
    """Draw fitted geometry alongside its resulting overhead view.

    The plate with the fitted geometry drawn on it, next to the overhead it produces.
    """
    plate = table_plate(video, work)
    over = plate.copy()
    verdict = {c.region: c.ok for c in (checks or [])}
    for c in CORNERS:
        for name, r, default_color in (
            (f"hand:{c}", cal.hand[c][0], (0, 255, 0)),
            (f"meld:{c}", cal.meld[c][0], (255, 0, 255)),
        ):
            col = default_color if verdict.get(name, True) else (0, 0, 255)
            cv2.rectangle(over, (r.x, r.y), (r.x + r.w, r.y + r.h), col, 2)
            cv2.putText(
                over, name, (r.x + 4, r.y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2
            )
    inv = np.linalg.inv(cal.derotation())
    for c, (rect, _, _) in cal.pond.items():
        x0, y0, x1, y1 = rect.xyxy
        q = _apply(inv, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype(np.int32)
        col = (255, 128, 0) if verdict.get(f"pond:{c}", True) else (0, 0, 255)
        cv2.polylines(
            over, [q.reshape(-1, 1, 2)], isClosed=True, color=col, thickness=2
        )
        cv2.putText(
            over, f"pond {c}", tuple(q[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2
        )
    u = cal.unit
    q = _apply(
        inv, [[u.x, u.y], [u.x + u.w, u.y], [u.x + u.w, u.y + u.h], [u.x, u.y + u.h]]
    ).astype(np.int32)
    cv2.polylines(
        over, [q.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2
    )
    oh, _ = cal.region(plate, "overhead")
    for rect, _, _ in cal.pond.values():
        cv2.rectangle(
            oh, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), (255, 128, 0), 2
        )
    cv2.rectangle(oh, (u.x, u.y), (u.x + u.w, u.y + u.h), (0, 0, 255), 2)
    h = 810
    left = cv2.resize(
        over, (int(over.shape[1] * h / over.shape[0]), h), interpolation=cv2.INTER_AREA
    )
    right = cv2.resize(oh, (h, h), interpolation=cv2.INTER_AREA)
    return np.hstack([left, right])


def write_unit_template(
    video: Path, cal: Calibration, work: Path, out: Path = UNIT_TEMPLATE
) -> Path:
    """Cut the unit template from a video whose fit is trusted (the reference VOD)."""
    plate = table_plate(video, work)
    m = unit_mask(plate)
    t = cv2.warpAffine(
        m, cal.derotation()[:2], (cal.side, cal.side), flags=cv2.INTER_NEAREST
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), t * 255)
    return out
