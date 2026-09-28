"""read_region: detector + classifier on one region of one frame -> boxes with posteriors.

Hands: boxes ordered along the row (left to right in the upright band). Melds
       moved beside the row at a reveal are detected too (4% of boxes over
       the labelled states); they are separated from the row by row structure
       (role "extra") so the row's count stays the hand's count.
Ponds: boxes assigned to (row, col) in the owner's reading order; boxes that
       are not part of the consecutive discard rows are indicators (dead wall).
       A row that cannot exist (seven positions with no neighbouring pond at
       the edge) rejects the whole reading rather than being truncated.
Melds: boxes grouped into meld rows (top to bottom, left to right).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..layout import Calibration, box_iou
from ..train.data import CLASS_INDEX, crop_box, region_upright
from .classifier import Classifier
from .detector import Det, Detector

SIDEWAYS_ASPECT = 1.15
NONE_MAX = 0.5
WALL_BAND = 0.32          # bottom fraction of the upright pond region where the wall row (dead wall, indicators) can lie;
                          # a real fourth discard row in the band is kept as a row when it continues the rows
EDGE = 3                  # px: a box this close to a region edge touches it
OVERLAP_IOU = 0.3         # two boxes of one row overlapping this much are one tile seen twice
DUP_IOU = 0.5             # melds: two boxes overlapping this much are one tile boxed twice
MELD_X_OVERLAP = 0.3      # melds: boxes overlapping along x by more than this share of the narrower are in two rows


@dataclass
class Box:
    """A detected tile with class probabilities and its physical row/group role."""
    xyxy: tuple[float, float, float, float]
    conf: float
    sideways: bool
    p: np.ndarray                    # posterior over the 39 classes
    role: str = "tile"               # tile | indicator | extra (a meld beside a hand row) | other (not of this region)
    row: Optional[int] = None
    col: Optional[int] = None
    group: Optional[int] = None      # meld group index; hands: 0 = the row, 1.. = groups beside it

    @property
    def cx(self):
        """Horizontal centre in upright region pixels."""
        return (self.xyxy[0] + self.xyxy[2]) / 2

    @property
    def cy(self):
        """Vertical centre in upright region pixels."""
        return (self.xyxy[1] + self.xyxy[3]) / 2

    @property
    def w(self):
        """Width in upright region pixels."""
        return self.xyxy[2] - self.xyxy[0]

    @property
    def h(self):
        """Height in upright region pixels."""
        return self.xyxy[3] - self.xyxy[1]

    def to_dict(self) -> dict:
        """Serialize compact evidence; absent structure fields remain omitted."""
        d = {"xyxy": [round(v, 1) for v in self.xyxy], "conf": round(self.conf, 3), "sideways": self.sideways,
             "role": self.role, "p": [round(float(v), 4) for v in self.p]}
        for k in ("row", "col", "group"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d


@dataclass
class Reading:
    """One sampled region; rejected geometry must not become an observation."""
    t: float
    region: str
    size: tuple[int, int]
    boxes: list[Box] = field(default_factory=list)
    rejected: bool = False           # the boxes cannot be a state of this region (a pond row of seven): observe drops it

    def to_dict(self) -> dict:
        """Return the stage-3 JSON representation used by caches and review."""
        return {"t": self.t, "region": self.region, "size": list(self.size), "rejected": self.rejected,
                "boxes": [b.to_dict() for b in self.boxes]}


def _clipped(b: Box, region_h: Optional[float]) -> bool:
    """The box is cut by the far edge of the region (the wall row lies there)."""
    return region_h is not None and b.xyxy[3] >= region_h - EDGE


def _at_left(b: Box) -> bool:
    return b.xyxy[0] <= EDGE


def _at_right(b: Box, region_w: Optional[float]) -> bool:
    return region_w is not None and b.xyxy[2] >= region_w - EDGE


def _at_side(b: Box, region_w: Optional[float]) -> bool:
    """The box touches a side edge of the region, where a neighbouring pond can spill in."""
    return _at_left(b) or _at_right(b, region_w)


def _tile_size(boxes: list[Box], region_h: Optional[float] = None) -> tuple[float, float]:
    """Median width / height of the upright face-up tiles (sideways ones swapped); tile backs and boxes cut
    by the region edge (the wall row) do not shape the estimate."""
    good = [b for b in boxes if b.p[CLASS_INDEX["X"]] < 0.5 and not _clipped(b, region_h)] or boxes
    ws = [b.h if b.sideways else b.w for b in good]
    hs = [b.w if b.sideways else b.h for b in good]
    return (float(np.median(ws)), float(np.median(hs))) if good else (40.0, 60.0)


def _cluster_rows(boxes: list[Box], th: float) -> list[list[Box]]:
    """Boxes grouped by height, top to bottom: a box within 0.55 tile heights of the cluster's mean joins it."""
    clusters: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        if clusters and abs(b.cy - np.mean([c.cy for c in clusters[-1]])) < 0.55 * th:
            clusters[-1].append(b)
        else:
            clusters.append([b])
    return clusters


def assign_pond(boxes: list[Box], region_h: Optional[float] = None, region_w: Optional[float] = None) -> bool:
    """Rows top-down (row 0 nearest the centre), columns left to right with gaps; leftovers are indicators.
    Returns True when the reading is rejected: a row of more than six positions that no neighbouring pond
    can explain (its boxes are marked "other"; the caller drops the reading)."""
    if not boxes:
        return False
    tw, th = _tile_size(boxes, region_h)
    for b in boxes:
        # a box cut by the far edge is wide and short, not a turned tile
        if b.sideways and _clipped(b, region_h) and b.h < 0.8 * th:
            b.sideways = False
    # a sliver at the near or side edges (a tile of the centre or of a neighbouring pond cut by the region)
    # is neither a discard nor an indicator
    for b in boxes:
        at_edge = b.xyxy[1] <= EDGE or _at_side(b, region_w)
        if at_edge and (b.h < 0.5 * th or b.w < 0.5 * tw):
            b.role = "other"
    boxes = [b for b in boxes if b.role != "other"]
    if not boxes:
        return False
    clusters = _cluster_rows(boxes, th)
    # consecutive discard rows from the top; a cluster far below the previous one, a cluster that is the
    # wall (mostly face-down tiles, or cut by the region edge), or the first cluster when it already lies
    # in the wall band at the bottom of the region, is the dead wall: its face-up tiles are indicators
    rows: list[list[Box]] = []
    prev_y = None
    for cl in clusters:
        y = float(np.mean([c.cy for c in cl]))
        if prev_y is not None and y - prev_y > 1.6 * th:
            break
        backs = sum(1 for c in cl if c.p[CLASS_INDEX["X"]] >= 0.5)
        clipped = sum(1 for c in cl if _clipped(c, region_h))
        if backs >= max(2, len(cl) / 2) or clipped == len(cl):
            break
        in_band = region_h is not None and y > region_h * (1.0 - WALL_BAND)
        if in_band and prev_y is None:
            break                                   # no discard row starts in the wall band
        # in the wall band a cluster is a discard row only when it continues the rows without a gap
        # (rows lie 0.99-1.13 tile heights apart, the wall 1.18 or more) and already holds three tiles
        # (one or two face-up tiles there are dora indicators); the wall's face-down tiles are often not
        # detected at all, so the wall must not be recognised by its backs
        if in_band and (y - prev_y > 1.15 * th or len(cl) < 3):
            break
        rows.append(cl)
        prev_y = y
    in_rows = {id(b) for cl in rows for b in cl}
    for b in boxes:
        if id(b) not in in_rows:
            b.role = "indicator"
    if not rows:
        return False
    rejected = False
    # columns count from the row's own first tile: rows are not aligned with each other
    # (a player may start a new row under the third tile of the previous one)
    for r, cl in enumerate(rows):
        cl.sort(key=lambda b: b.cx)
        if len(cl) > 6 and not any(_at_side(b, region_w) for b in cl):
            # rule of six: a seventh position does not exist, and no neighbouring pond reaches this row.
            # A tile seen twice (two boxes on one tile) pushes the real sixth tile out: the less confident
            # of the overlapping boxes goes; if the row is still too long the reading is wrong as a whole
            # and is rejected, never truncated to six
            dup = [b for b in cl if any(o is not b and box_iou(b.xyxy, o.xyxy) > OVERLAP_IOU for o in cl)]
            if dup:
                worst = min(dup, key=lambda b: b.conf)
                worst.role = "other"
                cl = [b for b in cl if b is not worst]
            if len(cl) > 6:
                for b in cl:
                    b.role = "other"
                rejected = True
                continue
        # rule of six at a side edge: the boxes beyond six belong to the neighbouring pond that shares the edge
        # the row touches, whichever side that is
        while len(cl) > 6 and (_at_left(cl[0]) or _at_right(cl[-1], region_w)):
            b = cl.pop(0) if _at_left(cl[0]) else cl.pop()
            b.role = "other"
        # A called-away tile leaves no gap: columns are each row's left-to-right ranks.
        for col, b in enumerate(cl):
            if col > 5:
                b.role = "other"
                continue
            b.row, b.col = r, col
    return rejected


def _x_overlap(a: Box, b: Box) -> float:
    """Overlap along x as a fraction of the narrower box."""
    return max(0.0, min(a.xyxy[2], b.xyxy[2]) - max(a.xyxy[0], b.xyxy[0])) / max(1.0, min(a.w, b.w))


def _rows_without_overlap(cl: list[Box]) -> list[list[Box]]:
    """Split a height cluster into rows in which no two boxes overlap along x: tiles of one row lie side by
    side, so two boxes over the same x are in two rows (melds laid one above the other at an angle). The one
    exception is a kakan: its added tile lies across the pon's turned tile, both sideways, one on the other."""
    def fits(b: Box, r: list[Box]) -> bool:
        return all(_x_overlap(b, o) <= MELD_X_OVERLAP or (b.sideways and o.sideways) for o in r)

    rows: list[list[Box]] = []
    for b in sorted(cl, key=lambda b: b.cy):
        home = next((r for r in rows if fits(b, r)), None)
        if home is None:
            rows.append([b])
        else:
            home.append(b)
    return sorted(rows, key=lambda r: float(np.mean([b.cy for b in r])))


def assign_meld(boxes: list[Box]) -> None:
    """Meld rows by y (top to bottom), boxes left to right; consecutive boxes closer than a tile width share a
    group. The same tile boxed twice keeps its better box (the other becomes "other"), and boxes overlapping
    along x never share a row. Runs of more than four boxes are split into melds by the engine (call rules)."""
    if not boxes:
        return
    for b in sorted(boxes, key=lambda b: -b.conf):
        if b.role != "other":
            for o in boxes:
                if o is not b and o.role != "other" and box_iou(b.xyxy, o.xyxy) > DUP_IOU:
                    o.role = "other"
    live = [b for b in boxes if b.role != "other"]
    if not live:
        return
    tw, th = _tile_size(live)
    g = 0
    for cl in _cluster_rows(live, th):
        for r in _rows_without_overlap(cl):
            r.sort(key=lambda b: b.cx)
            prev = None
            for b in r:
                if prev is not None and b.xyxy[0] - prev.xyxy[2] > 0.8 * tw:
                    g += 1
                b.group = g
                prev = b
            g += 1


def assign_hand(boxes: list[Box]) -> None:
    """The hand row and what lies beside it. Boxes cluster by height as melds do and a cluster splits at a
    gap wider than 0.8 tile widths; the longest group is the row (role "tile", group 0). Every other group
    is a meld the player moved beside the row at a reveal (role "extra", groups 1.. left to right): its
    tiles are not in the hand, so they never enter the row's count. Ties go to the leftmost group: melds
    lie at the player's right, which the hand camera sees on the right."""
    if not boxes:
        return
    tw, th = _tile_size(boxes)
    groups: list[list[Box]] = []
    for cl in _cluster_rows(boxes, th):
        cl.sort(key=lambda b: b.cx)
        cur: list[Box] = []
        for b in cl:
            if cur and b.xyxy[0] - cur[-1].xyxy[2] > 0.8 * tw:
                groups.append(cur)
                cur = []
            cur.append(b)
        groups.append(cur)
    row = max(groups, key=lambda g: (len(g), -g[0].cx))
    for b in row:
        b.role, b.group = "tile", 0
    extras = sorted((g for g in groups if g is not row), key=lambda g: g[0].cx)
    for k, g in enumerate(extras, start=1):
        for b in g:
            b.role, b.group = "extra", k


def read_region(frame: np.ndarray, cal: Calibration, name: str, det: Detector, clf: Classifier, *,
                t: float = 0.0, img: Optional[np.ndarray] = None) -> Reading:
    """Read one frame region, optionally reusing an already upright crop."""
    kind, _, corner = name.partition(":")
    if img is None:
        img, _ = region_upright(frame, cal, kind, corner)
    boxes, crops, sideways = _prepare(img, kind, det.predict(img))
    if crops:
        _set_posteriors(boxes, clf.classify(crops, sideways))
    return _finish(img, name, t, boxes)


def read_regions(items: list[tuple[float, str, np.ndarray]], det: Detector, clf: Classifier) -> list[Reading]:
    """Read ``(time, region name, upright BGR crop)`` items in input order.

    Detector batches preserve each crop's shape; classifier crops from all
    regions share an inference call. Callers bound the buffer to avoid keeping
    an entire video's images in memory. No samples or detections are omitted.
    """
    if not items:
        return []
    detections = det.predict_batch([img for _, _, img in items])
    prepared, crops, sideways = [], [], []
    for (_, name, img), dets in zip(items, detections):
        boxes, cs, ss = _prepare(img, name.partition(":")[0], dets)
        prepared.append((boxes, len(cs)))
        crops.extend(cs)
        sideways.extend(ss)
    if crops:
        probabilities = clf.classify(crops, sideways)
        start = 0
        for boxes, count in prepared:
            _set_posteriors(boxes, probabilities[start:start + count])
            start += count
    return [_finish(img, name, t, boxes) for (t, name, img), (boxes, _) in zip(items, prepared)]


def _prepare(img: np.ndarray, kind: str, dets: list[Det]) -> tuple[list[Box], list[np.ndarray], list[bool]]:
    boxes: list[Box] = []
    crops, sideways = [], []
    for d in dets:
        x0, y0, x1, y1 = d.xyxy
        # a turned tile in a pond / meld; a box cut by the far edge of the region is wide and short but not turned
        side = kind != "hand" and (x1 - x0) > SIDEWAYS_ASPECT * (y1 - y0) and not (kind == "pond" and y1 >= img.shape[0] - EDGE)
        b = Box(d.xyxy, d.conf, side, np.zeros(len(CLASS_INDEX), np.float32))
        if d.back:
            b.p[CLASS_INDEX["X"]] = 1.0
            boxes.append(b)
            continue
        c = crop_box(img, d.xyxy)
        if c is None:
            continue
        boxes.append(b)
        crops.append(c)
        sideways.append(side)
    return boxes, crops, sideways


def _set_posteriors(boxes: list[Box], probabilities: np.ndarray) -> None:
    j = 0
    for b in boxes:
        if b.p[CLASS_INDEX["X"]] == 1.0:
            continue
        b.p = probabilities[j]
        j += 1


def _finish(img: np.ndarray, name: str, t: float, boxes: list[Box]) -> Reading:
    kind = name.partition(":")[0]
    # a detection the classifier calls "not a tile" is junk: it must not take a row / column position
    boxes = [b for b in boxes if b.p[CLASS_INDEX["none"]] < NONE_MAX]
    rejected = False
    if kind == "pond":
        rejected = assign_pond(boxes, region_h=img.shape[0], region_w=img.shape[1])
        boxes.sort(key=lambda b: (b.role != "tile", b.row if b.row is not None else 99, b.col if b.col is not None else 99, b.cx))
    elif kind == "meld":
        assign_meld(boxes)
        boxes.sort(key=lambda b: (b.group if b.group is not None else 99, b.cx))
    else:
        assign_hand(boxes)
        boxes.sort(key=lambda b: (b.group if b.group is not None else 99, b.cx))
    return Reading(t, name, (img.shape[1], img.shape[0]), boxes, rejected)
