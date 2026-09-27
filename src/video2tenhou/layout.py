"""Calibration of the broadcast composite: every pixel number of the project.

A calibration (assets/calib/<name>.json, 1080p coordinates) names the
regions of the frame and how each is rendered for reading:

  overlay        rectangles of the score / name / wind text and the round line
  cam:<corner>   the whole corner camera
  hand:<corner>  the band of the corner camera where the player's hand row lies
  meld:<corner>  the meld-camera inset of that player
  overhead       the de-rotated overhead square (side x side)
  pond:<corner>  that player's pond, upright for its owner (owner at the bottom)

`transform(name)` returns the 3x3 matrix that maps frame pixels to region
pixels (affine: translation, rotation by quarter turns, scale) and the region
size; `region(frame, name)` renders it. Labels are stored in frame
coordinates and mapped through the same matrices, so a calibration change
never invalidates a label.

Two layers (DESIGN.md 4.2). The **layout** file is the broadcast composite
and the table: overlay rectangles, corner quadrants, and — in the de-rotated
overhead square — the centre unit and the pond rectangles, which are
properties of the physical table and the same on every video of it. The
**fit** (`labels/<video>/calib.json`, written by `calibfit`) is where that
table and those panels landed in one video: the overhead centre, angle and
scale, and a rectangle per corner panel. `Calibration.load(name, video)`
applies the fit; `cal.fit` is None when the video has none, which is only
right for the video the layout was drawn on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .paths import ASSET_DIR, DATA_DIR as ROOT, LABEL_DIR

CALIB_DIR = ASSET_DIR / "calib"
CORNERS = ("TL", "TR", "BL", "BR")
FRAME_W, FRAME_H = 1920, 1080


def fit_path(video: str | Path) -> Path:
    """Where this video's fit lives. `video` is a path or a stem."""
    return LABEL_DIR / Path(video).stem / "calib.json"


def apply_fit(data: dict, fit: dict) -> dict:
    """The layout dict with one video's measured geometry substituted (DESIGN.md 4.2a).

    Only what the fit names is replaced, so a fit that measured the overhead alone keeps the
    layout's panels. The pond and unit rectangles are never in a fit: they are the table.
    """
    d = json.loads(json.dumps(data))                      # the layout is shared; never edit it in place
    oh = fit.get("overhead") or {}
    if "center" in oh:
        d["overhead"]["center"] = [float(oh["center"][0]), float(oh["center"][1])]
    for k in ("angle", "scale"):
        if k in oh:
            d["overhead"][k] = float(oh[k])
    for c, r in (fit.get("cam") or {}).items():
        d["cam"][c] = [int(round(v)) for v in r]
    for c, v in (fit.get("hand") or {}).items():
        if "rect" in v:
            d["hand"][c]["rect"] = [int(round(x)) for x in v["rect"]]
        if "roll" in v:
            d["hand"][c]["roll"] = float(v["roll"])
    for c, v in (fit.get("meld") or {}).items():
        if "rect" in v:
            d["meld"][c]["rect"] = [int(round(x)) for x in v["rect"]]
    return d


@dataclass(frozen=True)
class Rect:
    """A rectangle in 1080p frame or region pixels; coordinates are never normalized."""
    x: int
    y: int
    w: int
    h: int

    def crop(self, img: np.ndarray) -> np.ndarray:
        """Return the NumPy view inside this rectangle; callers must ensure it lies in the image."""
        return img[self.y: self.y + self.h, self.x: self.x + self.w]

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Exclusive right/bottom bounds for OpenCV and detection geometry."""
        return self.x, self.y, self.x + self.w, self.y + self.h

    @staticmethod
    def of(v: Iterable[int]) -> "Rect":
        """Build from the layout JSON sequence [x, y, width, height]."""
        x, y, w, h = (int(round(a)) for a in v)
        return Rect(x, y, w, h)


class Calibration:
    """Geometry shared by perception and labeling, with an optional per-video fit applied."""
    fit: dict | None = None          # this video's measured geometry, None when the layout's own numbers are used
    video: str | None = None

    def __init__(self, data: dict):
        self.data = data
        self.name: str = data["name"]
        self.frame: tuple[int, int] = tuple(data["frame"])  # type: ignore[assignment]
        ov = data["overlay"]
        self.score = {c: Rect.of(ov["score"][c]) for c in CORNERS}
        self.player_name = {c: Rect.of(ov["name"][c]) for c in CORNERS}
        self.wind = {c: Rect.of(ov["wind"][c]) for c in CORNERS}
        self.strip = {c: Rect.of(ov["strip"][c]) for c in CORNERS}
        self.round_wind = Rect.of(ov["round_wind"])
        self.round_num = Rect.of(ov["round_num"])
        self.honba = Rect.of(ov["honba"])
        self.sticks = Rect.of(ov["sticks"])
        self.cam = {c: Rect.of(data["cam"][c]) for c in CORNERS}
        self.hand = {c: (Rect.of(d["rect"]), float(d.get("scale", 1.0))) for c, d in data["hand"].items()}
        self.meld = {c: (Rect.of(d["rect"]), float(d.get("scale", 1.0))) for c, d in data["meld"].items()}
        oh = data["overhead"]
        self.center = (float(oh["center"][0]), float(oh["center"][1]))
        self.angle = float(oh["angle"])
        self.scale = float(oh.get("scale", 1.0))
        self.side = int(oh["side"])
        self.unit = Rect.of(oh["unit"])
        self.pond = {c: (Rect.of(d["rect"]), int(d["rot"]) % 4, float(d.get("scale", 1.0))) for c, d in data["pond"].items()}

    # -- loading -----------------------------------------------------------------
    @classmethod
    def load(cls, name_or_path: str | Path = "pml", video: str | Path | None = None) -> "Calibration":
        """The layout, with this video's fit applied when it has one (`cal.fit`)."""
        p = Path(name_or_path)
        if not p.exists():
            p = CALIB_DIR / f"{name_or_path}.json"
        data = json.load(open(p, encoding="utf-8"))
        fit = None
        if video is not None:
            fp = fit_path(video)
            if fp.exists():
                fit = json.load(open(fp, encoding="utf-8"))
                data = apply_fit(data, fit)
        cal = cls(data)
        cal.fit = fit
        cal.video = Path(video).stem if video is not None else None
        return cal

    # -- transforms ----------------------------------------------------------------
    def derotation(self) -> np.ndarray:
        """frame -> de-rotated overhead square (0..side)."""
        R = cv2.getRotationMatrix2D(self.center, self.angle, self.scale)
        R = np.vstack([R, [0.0, 0.0, 1.0]])
        h = self.side / 2.0
        T = np.array([[1, 0, -(self.center[0] - h)], [0, 1, -(self.center[1] - h)], [0, 0, 1]], np.float64)
        return T @ R

    @staticmethod
    def _crop_rot_scale(rect: Rect, k: int, scale: float) -> tuple[np.ndarray, tuple[int, int]]:
        """Matrix and size for cropping `rect`, turning it k quarter turns clockwise and scaling."""
        w, h = rect.w, rect.h
        if k == 0:
            A = np.array([[1, 0, -rect.x], [0, 1, -rect.y]], np.float64); size = (w, h)
        elif k == 1:  # clockwise 90: (x, y) -> (h - y, x)
            A = np.array([[0, -1, h + rect.y], [1, 0, -rect.x]], np.float64); size = (h, w)
        elif k == 2:
            A = np.array([[-1, 0, w + rect.x], [0, -1, h + rect.y]], np.float64); size = (w, h)
        else:  # counter-clockwise 90: (x, y) -> (y, w - x)
            A = np.array([[0, 1, -rect.y], [-1, 0, w + rect.x]], np.float64); size = (h, w)
        M = np.vstack([A * scale, [0, 0, 1]])
        return M, (int(round(size[0] * scale)), int(round(size[1] * scale)))

    def roll(self, corner: str) -> float:
        """Angle of the hand row in the corner camera; rotating a crop by -roll makes the tiles upright."""
        return float(self.data["hand"][corner].get("roll", 0.0))

    def regions(self) -> list[str]:
        """Region names accepted by transform(), including the full overhead and cameras."""
        out = ["overhead"]
        out += [f"pond:{c}" for c in self.pond] + [f"hand:{c}" for c in self.hand] + [f"meld:{c}" for c in self.meld]
        out += [f"cam:{c}" for c in self.cam]
        return out

    def transform(self, name: str) -> tuple[np.ndarray, tuple[int, int]]:
        """(3x3 frame->region matrix, (width, height) of the region image)."""
        kind, _, corner = name.partition(":")
        if kind == "overhead":
            return self.derotation(), (self.side, self.side)
        if kind == "pond":
            rect, k, s = self.pond[corner]
            M, size = self._crop_rot_scale(rect, k, s)
            return M @ self.derotation(), size
        if kind == "hand":
            rect, s = self.hand[corner]
        elif kind == "meld":
            rect, s = self.meld[corner]
        elif kind == "cam":
            rect, s = self.cam[corner], 1.0
        else:
            raise KeyError(name)
        M, size = self._crop_rot_scale(rect, 0, s)
        return M, size

    def region(self, frame: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        """(region image, 3x3 frame->region matrix). The frame must be 1080p."""
        M, (w, h) = self.transform(name)
        kind, _, corner = name.partition(":")
        if kind in ("hand", "meld", "cam"):
            rect = {"hand": self.hand[corner][0], "meld": self.meld[corner][0], "cam": self.cam[corner]}[kind] if kind != "cam" else self.cam[corner]
            x0, y0 = max(0, rect.x), max(0, rect.y)
            img = frame[y0: min(frame.shape[0], rect.y + rect.h), x0: min(frame.shape[1], rect.x + rect.w)]
            if (x0, y0) != (rect.x, rect.y) or img.shape[1] != rect.w or img.shape[0] != rect.h:
                pad = np.zeros((rect.h, rect.w, 3), frame.dtype)
                pad[y0 - rect.y: y0 - rect.y + img.shape[0], x0 - rect.x: x0 - rect.x + img.shape[1]] = img
                img = pad
            if M[0, 0] != 1.0:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
            return img, M
        img = cv2.warpAffine(frame, M[:2], (w, h), flags=cv2.INTER_CUBIC)
        return img, M


# Default calibration and the overlay rectangles the overlay reader uses.
DEFAULT = Calibration.load("pml")
WIND, ROUND_WIND, ROUND_NUM, HONBA_NUM, RIICHI_NUM = DEFAULT.wind, DEFAULT.round_wind, DEFAULT.round_num, DEFAULT.honba, DEFAULT.sticks


# -- coordinate helpers ---------------------------------------------------------------

def apply(M: np.ndarray, pts) -> np.ndarray:
    """Apply a 3x3 matrix to an (n, 2) array of points."""
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    q = (M @ np.hstack([p, np.ones((len(p), 1))]).T).T
    return q[:, :2] / q[:, 2:3]


def quad_to_box(M: np.ndarray, quad) -> tuple[float, float, float, float]:
    """Frame quad -> axis-aligned (x0, y0, x1, y1) box in region coordinates."""
    q = apply(M, quad)
    return float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())


def box_to_quad(M: np.ndarray, box) -> list[list[float]]:
    """Region (x0, y0, x1, y1) box -> frame quad (tl, tr, br, bl)."""
    x0, y0, x1, y1 = box
    q = apply(np.linalg.inv(M), [[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return [[round(float(x), 1), round(float(y), 1)] for x, y in q]


# -- contact sheet ---------------------------------------------------------------------

def contact_sheet(frame: np.ndarray, cal: Calibration) -> np.ndarray:
    """The frame with every region outlined, plus the rendered regions, in one image."""
    over = frame.copy()
    for c in CORNERS:
        for r, col in ((cal.score[c], (0, 255, 255)), (cal.player_name[c], (0, 200, 255)), (cal.wind[c], (255, 255, 0)),
                       (cal.hand[c][0], (0, 255, 0)), (cal.meld[c][0], (255, 0, 255)), (cal.cam[c], (120, 120, 120))):
            cv2.rectangle(over, (r.x, r.y), (r.x + r.w, r.y + r.h), col, 2)
    for r in (cal.round_wind, cal.round_num, cal.honba, cal.sticks):
        cv2.rectangle(over, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 255, 255), 2)
    inv = np.linalg.inv(cal.derotation())
    for c, (rect, _, _) in cal.pond.items():
        x0, y0, x1, y1 = rect.xyxy
        q = apply(inv, [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype(np.int32)
        cv2.polylines(over, [q.reshape(-1, 1, 2)], True, (255, 128, 0), 2)
        cv2.putText(over, f"pond {c}", tuple(q[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
    u = cal.unit
    q = apply(inv, [[u.x, u.y], [u.x + u.w, u.y], [u.x + u.w, u.y + u.h], [u.x, u.y + u.h]]).astype(np.int32)
    cv2.polylines(over, [q.reshape(-1, 1, 2)], True, (0, 0, 255), 2)

    def fit(img, w, h):
        s = min(w / img.shape[1], h / img.shape[0])
        out = np.zeros((h, w, 3), np.uint8)
        r = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        out[: r.shape[0], : r.shape[1]] = r
        return out

    tiles = []
    for name in [f"pond:{c}" for c in CORNERS] + [f"hand:{c}" for c in CORNERS] + [f"meld:{c}" for c in CORNERS]:
        img, _ = cal.region(frame, name)
        cell = fit(img, 480, 270)
        cv2.putText(cell, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(cell)
    rows = [np.hstack(tiles[i: i + 4]) for i in range(0, 12, 4)]
    grid = np.vstack(rows)
    top = np.hstack([fit(over, 1440, 810), fit(cal.region(frame, "overhead")[0], 480, 810)])
    return np.vstack([top, fit(grid, 1920, 810)])
