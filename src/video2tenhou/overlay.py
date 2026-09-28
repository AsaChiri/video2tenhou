"""Read the broadcast text overlay: scores, names, seat winds, round/honba/riichi.

All overlay text is white on a translucent dark box, so we isolate bright,
low-saturation pixels. Digits and wind glyphs use broadcast templates;
Tesseract reads player names and provides a fallback for unreadable numbers.
Digit comparisons reuse constant template statistics and cache exact normalized
images without skipping changed pixels or weakening confidence thresholds.

The corner boxes are *not* at fixed pixel positions: each box is sized to the
longest of its two text lines and anchored to the outer edge of the screen, so
the wind glyph and the text shift by up to ~120 px depending on the player's
name. We therefore locate the wind glyph inside a wide search strip and read
the text relative to it.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from . import layout as L

from .paths import ASSET_DIR

TEMPLATE_DIR = ASSET_DIR / "templates"
WIND_NAMES = ("E", "S", "W", "N")  # index == tenhou wind index (0..3)
WIND_KANJI = dict(zip(WIND_NAMES, "東南西北"))

# Search strip per corner (1080p). Wide enough to contain the wind glyph and
# both text lines for any name length.
CORNER_STRIP = {
    "TL": L.Rect(0, 10, 760, 130),
    "TR": L.Rect(1160, 10, 760, 130),
    "BL": L.Rect(0, 550, 760, 130),
    "BR": L.Rect(1160, 550, 760, 130),
}
GLYPH_MIN_CONF = 0.6


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def white_mask(bgr: np.ndarray, *, v_min: int = 190, s_max: int = 70) -> np.ndarray:
    """255 where the pixel is bright and unsaturated (overlay text), else 0."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, v_min), (180, s_max, 255))


def ocr_ready(bgr: np.ndarray, scale: int = 3) -> np.ndarray:
    """Black text on white, upscaled — what tesseract likes."""
    m = white_mask(bgr)
    m = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    m = cv2.GaussianBlur(m, (3, 3), 0)
    m = cv2.copyMakeBorder(m, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=0)
    return 255 - m


def _tess(img: np.ndarray, config: str) -> str:
    import pytesseract

    # Cache the exact prepared pixels, including failures to recognize text.
    # No similarity threshold can carry a previous score across a changed crop.
    return _tess_pixels(img.shape, img.dtype.str, img.tobytes(), config,
                        pytesseract.pytesseract.tesseract_cmd)


@lru_cache(maxsize=64)
def _tess_pixels(shape: tuple, dtype: str, pixels: bytes, config: str, command: str) -> str:
    import pytesseract

    image = np.frombuffer(pixels, dtype=np.dtype(dtype)).reshape(shape)
    return pytesseract.image_to_string(image, config=config).strip()


def ocr_digit(bgr: np.ndarray) -> Optional[int]:
    """Single large digit (round number, honba, riichi sticks)."""
    if _DIGITS.ready:
        v, _ = read_number(white_mask(bgr))
        if v is not None and 0 <= v <= 9:
            return v
    txt = _tess(ocr_ready(bgr, 4), "--psm 8 -c tessedit_char_whitelist=0123456789")
    m = re.search(r"\d", txt)
    return int(m.group()) if m else None


_NAME_OK = re.compile(r"[^A-Za-z0-9 .()'_-]")


def ocr_score_and_name(bgr: np.ndarray, *, names: bool = True) -> tuple[Optional[int], str]:
    """Read a two-line block '<score> / <name>': score via digit templates, name via tesseract."""
    score = None
    if _DIGITS.ready:
        score, _ = read_number(white_mask(bgr))
        if score is not None and not names:
            return score, ""
    txt = _tess(ocr_ready(bgr), "--psm 6")
    name_lines = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.search(r"-?\d{3,6}", line.replace(" ", "").replace(",", ""))
        if m:  # the score line: only used if the template reader failed
            if score is None:
                score = int(m.group())
            continue
        name_lines.append(line)
    name = _NAME_OK.sub("", " ".join(name_lines))
    name = re.sub(r"\s+", " ", name).strip(" -_.")
    return score, name


# ---------------------------------------------------------------------------
# Digit classification (score line, round/honba/riichi digits)
# ---------------------------------------------------------------------------
#
# Tesseract confuses 5/9 and 3/8 in this font even on clean binarized crops,
# so digits are classified by normalized correlation against templates cut
# from real broadcast frames (assets/templates/digit_<n>_<k>.png). Tesseract
# is the fallback while templates are incomplete.

DIGIT_SIZE = (24, 40)  # (w, h) after normalization
DIGIT_MIN_CONF = 0.75


def digit_signature(mask: np.ndarray) -> Optional[np.ndarray]:
    """Normalize a nonempty binary digit mask; return None for fewer than ten foreground pixels."""
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None
    g = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    g = cv2.resize(g, DIGIT_SIZE, interpolation=cv2.INTER_AREA)
    return g.astype(np.float32) / 255.0


class DigitMatcher:
    """Digit template matcher for PML text, used before the slower OCR fallback."""
    def __init__(self):
        self.templates: dict[str, list[np.ndarray]] = {}
        for p in sorted(TEMPLATE_DIR.glob("digit_*.png")):
            label = p.stem.split("_")[1]
            sig = digit_signature(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
            if sig is not None:
                self.templates.setdefault(label, []).append(sig)
        # The template centering and energies are constant across every frame.
        # Keep the original template order so equal correlations break ties identically.
        self._labels = [label for label, sigs in self.templates.items() for _ in sigs]
        matrix = np.array([sig.reshape(-1) for sigs in self.templates.values() for sig in sigs], dtype=np.float32)
        self._centered = matrix - matrix.mean(axis=1, keepdims=True) if len(matrix) else matrix
        self._energy = (self._centered * self._centered).sum(axis=1) if len(matrix) else np.array([])
        self._cached_signature = lru_cache(maxsize=2048)(self._match_signature)

    @property
    def ready(self) -> bool:
        """Whether templates cover all ten digits; partial template sets cannot replace OCR."""
        return all(str(d) in self.templates for d in range(10))

    def classify(self, mask: np.ndarray) -> tuple[Optional[str], float]:
        """Return the best digit label and normalized correlation, or (None, 0) for an empty mask."""
        sig = digit_signature(mask)
        if sig is None or not self._labels:
            return None, 0.0
        return self._cached_signature(sig.tobytes())

    def _match_signature(self, key: bytes) -> tuple[Optional[str], float]:
        # Exact normalized pixels are the cache key: compression changes still
        # trigger a comparison, and reloading templates creates a fresh cache.
        sig = np.frombuffer(key, dtype=np.float32)
        centered = sig - sig.mean()
        # Row-wise float32 reductions use the same NCC formula as _ncc; no
        # quantization, template pruning or lower confidence threshold is used.
        denom = np.sqrt((centered * centered).sum() * self._energy)
        denom = np.where(denom == 0, np.float32(1e-6), denom)
        scores = (self._centered * centered).sum(axis=1) / denom
        best = int(np.argmax(scores))
        return (self._labels[best], float(scores[best])) if scores[best] > -1.0 else (None, -1.0)


_DIGITS = DigitMatcher()


def line_boxes(mask: np.ndarray) -> tuple[list[tuple[int, int, int, int]], bool]:
    """Digit boxes of the *topmost* text line of a mask, left to right, plus a minus-sign flag.

    Digit-sized components (>= 16 px tall) are grouped by vertical overlap with
    the topmost one; stubs and speckles are dropped. A short, wide component
    left of the first digit at mid-height is the minus sign.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    comps = [tuple(int(v) for v in st[:4]) for st in stats[1:]]
    digits = [b for b in comps if b[3] >= 16 and b[2] >= 6]
    if not digits:
        return [], False
    top = min(digits, key=lambda b: b[1])
    line = []
    for b in digits:
        ov = min(b[1] + b[3], top[1] + top[3]) - max(b[1], top[1])
        if ov > 0.5 * min(b[3], top[3]):
            line.append(b)
    line_h = float(np.median([b[3] for b in line]))
    line = sorted((b for b in line if b[3] >= 0.75 * line_h), key=lambda b: b[0])
    # At low resolution neighbouring digits fuse into one blob; split blobs
    # wider than a digit (~0.62 x height in this font) into equal parts.
    split: list[tuple[int, int, int, int]] = []
    for x, y, w, h in line:
        n = max(1, int(round(w / (0.62 * line_h))))
        if n == 1 or w <= 0.9 * line_h:
            split.append((x, y, w, h))
        else:
            step = w / n
            split.extend((x + int(round(i * step)), y, int(round((i + 1) * step)) - int(round(i * step)), h) for i in range(n))
    line = split
    if not line:
        return [], False
    x_first, y_top = line[0][0], line[0][1]
    minus = any(
        b[3] <= 0.35 * line_h and b[2] >= 1.5 * b[3] and b[0] + b[2] <= x_first + 2
        and y_top + 0.3 * line_h <= b[1] + b[3] / 2 <= y_top + 0.7 * line_h
        for b in comps
    )
    return line, minus


def read_number(mask: np.ndarray) -> tuple[Optional[int], float]:
    """Read the (possibly negative) integer on the top text line of a mask with the digit templates.

    Returns (value, lowest per-digit confidence); (None, 0) if not readable.
    """
    boxes, minus = line_boxes(mask)
    if not boxes:
        return None, 0.0
    out, conf = ("-" if minus else ""), 1.0
    for x, y, w, h in boxes:
        d, s = _DIGITS.classify(mask[y : y + h, x : x + w])
        if d is None or s < DIGIT_MIN_CONF:
            return None, 0.0
        out += d
        conf = min(conf, s)
    return int(out), conf


def save_digit_templates(mask: np.ndarray, text: str) -> int:
    """Cut per-digit templates from a mask whose top-line text is known (e.g. "-24900")."""
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    boxes, minus = line_boxes(mask)
    digits = [ch for ch in text if ch.isdigit()]
    if len(boxes) != len(digits) or minus != text.startswith("-"):
        raise ValueError(f"found {len(boxes)} digit glyphs (minus={minus}) for {text!r}")
    for (x, y, w, h), ch in zip(boxes, digits):
        k = 1
        while (TEMPLATE_DIR / f"digit_{ch}_{k}.png").exists():
            k += 1
        cv2.imwrite(str(TEMPLATE_DIR / f"digit_{ch}_{k}.png"), mask[y : y + h, x : x + w])
    _DIGITS.__init__()
    return len(digits)


# ---------------------------------------------------------------------------
# Kanji template matching
# ---------------------------------------------------------------------------

GLYPH_SIZE = 48


def glyph_signature(mask: np.ndarray) -> Optional[np.ndarray]:
    """Tight-crop the bright pixels and resize to a fixed square.

    Makes matching independent of the glyph's on-screen size, so one template
    set serves both the corner seat winds and the (larger) round wind.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 30:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if y1 - y0 < 10 or x1 - x0 < 10:
        return None
    g = mask[y0 : y1 + 1, x0 : x1 + 1]
    g = cv2.resize(g, (GLYPH_SIZE, GLYPH_SIZE), interpolation=cv2.INTER_AREA)
    return g.astype(np.float32) / 255.0


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum())) or 1e-6
    return float((a * b).sum() / denom)


class GlyphMatcher:
    """Match a kanji crop against named templates (東/南/西/北) by normalized correlation."""

    def __init__(self, names=WIND_NAMES, prefix: str = "wind_"):
        self.templates: dict[str, np.ndarray] = {}
        for n in names:
            p = TEMPLATE_DIR / f"{prefix}{n}.png"
            if p.exists():
                sig = glyph_signature(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
                if sig is not None:
                    self.templates[n] = sig

    @property
    def ready(self) -> bool:
        """Whether all four wind glyphs have templates."""
        return len(self.templates) == len(WIND_NAMES)

    def match_mask(self, mask: np.ndarray, allowed=WIND_NAMES) -> tuple[Optional[str], float]:
        """Return the best allowed wind and correlation without applying the caller confidence threshold."""
        sig = glyph_signature(mask)
        if sig is None or not self.templates:
            return None, 0.0
        best, best_score = None, -1.0
        for name in allowed:
            tpl = self.templates.get(name)
            if tpl is None:
                continue
            s = _ncc(sig, tpl)
            if s > best_score:
                best, best_score = name, s
        return best, best_score

    def match(self, bgr: np.ndarray, allowed=WIND_NAMES) -> tuple[Optional[str], float]:
        """Match a BGR glyph crop after extracting its white text mask."""
        return self.match_mask(white_mask(bgr), allowed)


_WIND = GlyphMatcher()


def _glyph_candidates(mask: np.ndarray, *, gap: int = 20):
    """Square-ish kanji-sized blobs in a strip mask -> [(x, y, w, h)].

    Works on raw connected components and merges horizontally adjacent tall
    components (北 is drawn as two separate strokes), but never merges across
    a gap wide enough to be the space between the glyph and the text.
    """
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    tall = sorted(
        ((int(x), int(y), int(w), int(h)) for x, y, w, h, _a in stats[1:] if 50 <= h <= 100 and w >= 8),
        key=lambda c: c[0],
    )
    merged: list[list[int]] = []
    for x, y, w, h in tall:
        if merged:
            mx, my, mw, mh = merged[-1]
            v_overlap = min(y + h, my + mh) - max(y, my)
            if x - (mx + mw) <= gap and v_overlap > 0.5 * min(h, mh):
                nx, ny = min(x, mx), min(y, my)
                merged[-1] = [nx, ny, max(x + w, mx + mw) - nx, max(y + h, my + mh) - ny]
                continue
        merged.append([x, y, w, h])
    return [tuple(c) for c in merged if 55 <= c[3] <= 100 and 50 <= c[2] <= 100 and 0.7 <= c[2] / c[3] <= 1.4]


def save_wind_templates(frame: np.ndarray, corner_to_wind: dict[str, str]) -> None:
    """Cut seat-wind glyph templates out of a 1080p frame, given which wind is in which corner.

    The glyph is located as a connected component inside the corner strip, so
    the template is never clipped by a fixed rectangle.
    """
    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    for corner, wind in corner_to_wind.items():
        m = white_mask(CORNER_STRIP[corner].crop(frame))
        cands = _glyph_candidates(m)
        if not cands:
            raise ValueError(f"no glyph-sized component found in {corner}")
        # the glyph is the candidate closest to the nominal WIND rect for this corner
        nx = L.WIND[corner].x - CORNER_STRIP[corner].x
        x, y, w, h = min(cands, key=lambda c: abs(c[0] - nx))
        tpl = m[max(0, y - 2) : y + h + 2, max(0, x - 2) : x + w + 2]
        cv2.imwrite(str(TEMPLATE_DIR / f"wind_{wind}.png"), tpl)
    _WIND.__init__()


# ---------------------------------------------------------------------------
# Corner box parsing
# ---------------------------------------------------------------------------


def find_wind_glyph(mask: np.ndarray) -> Optional[tuple[str, float, tuple[int, int, int, int]]]:
    """Locate the seat-wind kanji in a corner strip mask.

    Returns (wind, confidence, (x, y, w, h)) in strip coordinates. 北 is two
    disconnected strokes, so components are bridged horizontally first.
    """
    best = None
    for x, y, w, h in _glyph_candidates(mask):
        wind, conf = _WIND.match_mask(mask[y : y + h, x : x + w])
        if wind is not None and conf >= GLYPH_MIN_CONF and (best is None or conf > best[1]):
            best = (wind, conf, (x, y, w, h))
    return best


@dataclass
class CornerInfo:
    """One overlay seat reading; unknown score/wind fields stay None rather than being guessed."""
    corner: str
    score: Optional[int]
    name: str
    wind: Optional[str]  # "E" / "S" / "W" / "N"
    wind_conf: float
    glyph_box: Optional[tuple[int, int, int, int]] = None  # frame coords


def read_corner(frame: np.ndarray, corner: str, *, names: bool = True, cal: L.Calibration | None = None) -> CornerInfo:
    """Read a PML-style glyph/text block from the selected layout's search strip."""
    strip = cal.strip[corner] if cal is not None else CORNER_STRIP[corner]
    crop = strip.crop(frame)
    found = find_wind_glyph(white_mask(crop))
    if found is None:
        return CornerInfo(corner, None, "", None, 0.0)
    wind, conf, (gx, gy, gw, gh) = found
    # Text block sits beside the glyph, on the outer side of the screen.
    y0, y1 = max(0, gy - 20), min(crop.shape[0], gy + gh + 12)
    # Keep 12 px away from the screen edge: bright things at the frame border
    # (arms, tiles) otherwise leak into the crop and OCR as an extra digit.
    if corner.endswith("L"):
        x0, x1 = 12, max(0, gx - 6)
    else:
        x0, x1 = min(crop.shape[1], gx + gw + 6), crop.shape[1] - 12
    score, name = (None, "")
    if x1 - x0 > 40:
        score, name = ocr_score_and_name(crop[y0:y1, x0:x1], names=names)
    return CornerInfo(corner, score, name, wind, conf, (strip.x + gx, strip.y + gy, gw, gh))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class OverlayState:
    """Single-frame header evidence before temporal filtering or site-record validation."""
    round_wind: Optional[str]  # "E" / "S" / "W"
    round_num: Optional[int]  # 1..4
    honba: Optional[int]
    riichi_sticks: Optional[int]
    corners: dict[str, CornerInfo]

    @property
    def kyoku_index(self) -> Optional[int]:
        """Zero-based round index, or None when the round wind or number is unknown."""
        if self.round_wind is None or self.round_num is None:
            return None
        return WIND_NAMES.index(self.round_wind) * 4 + (self.round_num - 1)

    @property
    def scores(self) -> dict[str, Optional[int]]:
        """Scores keyed by fixed camera corner; values may be unknown."""
        return {c: ci.score for c, ci in self.corners.items()}

    def seat_order(self) -> Optional[list[str]]:
        """Corners ordered by tenhou seat index (seat 0 = East of the *first* hand).

        Within one hand, seat winds rotate with the dealer: the corner showing
        East now is seat `kyoku_index % 4`. Returns None if any wind is unknown.
        """
        k = self.kyoku_index
        winds = {c: ci.wind for c, ci in self.corners.items()}
        if k is None or None in winds.values() or len(set(winds.values())) != 4:
            return None
        order: list[Optional[str]] = [None] * 4
        for corner, w in winds.items():
            seat = (WIND_NAMES.index(w) + k) % 4  # type: ignore[arg-type]
            order[seat] = corner
        return order  # type: ignore[return-value]

    def to_dict(self) -> dict:
        """Serialize readings and the derived kyoku index for diagnostics."""
        d = asdict(self)
        d["kyoku_index"] = self.kyoku_index
        return d

    def __str__(self) -> str:
        rw = WIND_KANJI.get(self.round_wind or "", "?")
        head = f"{rw}{self.round_num or '?'} 本場{self.honba if self.honba is not None else '?'} 供託{self.riichi_sticks if self.riichi_sticks is not None else '?'}"
        parts = [f"{c}:{WIND_KANJI.get(ci.wind or '', '?')} {ci.score} {ci.name}" for c, ci in self.corners.items()]
        return head + " | " + " | ".join(parts)


def normalize(frame: np.ndarray) -> np.ndarray:
    """Resize a BGR broadcast frame to the 1920x1080 coordinate system used by all layouts."""
    if frame.shape[1] != L.FRAME_W or frame.shape[0] != L.FRAME_H:
        frame = cv2.resize(frame, (L.FRAME_W, L.FRAME_H), interpolation=cv2.INTER_CUBIC)
    return frame


def read_overlay(frame: np.ndarray, *, names: bool = True, cal: L.Calibration | None = None,
                 complete_only: bool = False) -> OverlayState:
    """Read a BGR frame using the layout's rectangles; missing glyphs remain unknown.

    Geometry is configurable, while the glyph font and text arrangement follow
    PML. A different overlay style needs a reader/template adaptation as well.
    ``complete_only`` stops once a required numeric/wind field is missing or
    seat winds repeat: such a frame cannot enter timeline segmentation. Remaining
    fields stay unknown. Leave it false for diagnostic and player-name reads.
    """
    cal = cal or L.DEFAULT
    frame = normalize(frame)
    rw, rconf = _WIND.match(cal.round_wind.crop(frame), allowed=("E", "S", "W"))
    state = OverlayState(rw if rconf >= GLYPH_MIN_CONF else None, None, None, None,
                         {c: CornerInfo(c, None, "", None, 0.) for c in L.CORNERS})
    if complete_only and state.round_wind is None:
        return state
    for field, region in (("round_num", cal.round_num), ("honba", cal.honba),
                          ("riichi_sticks", cal.sticks)):
        value = ocr_digit(region.crop(frame))
        setattr(state, field, value)
        if complete_only and value is None:
            return state
    winds = set()
    for corner in L.CORNERS:
        info = read_corner(frame, corner, names=names, cal=cal)
        state.corners[corner] = info
        if complete_only and (info.wind is None or info.score is None or info.wind in winds):
            return state
        winds.add(info.wind)
    return state
