# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Upright region images and tile crops shared by recognition and training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from video2tenhou.layout import Calibration

MIN_ROLL_CORRECTION = 0.5
MIN_FACE_CROP_SIDE = 8
CROP_W, CROP_H = 64, 96


def region_upright(
    frame: np.ndarray, cal: Calibration, kind: str, corner: str
) -> tuple[np.ndarray, np.ndarray]:
    """Region image turned so tiles stand upright, and the frame->image matrix.

    Ponds and melds are upright already; hand bands are rotated by -roll about
    their centre (the canvas grows so nothing is cut off).
    """
    img, transform = cal.region(frame, f"{kind}:{corner}")
    roll = cal.roll(corner) if kind == "hand" else 0.0
    if abs(roll) < MIN_ROLL_CORRECTION:
        return img, transform
    h, w = img.shape[:2]
    # positive angle = counter-clockwise: undoes a descending row
    rotation = cv2.getRotationMatrix2D((w / 2, h / 2), roll, 1.0)
    cos, sin = abs(rotation[0, 0]), abs(rotation[0, 1])
    nw, nh = int(w * cos + h * sin), int(w * sin + h * cos)
    rotation[0, 2] += nw / 2 - w / 2
    rotation[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, rotation, (nw, nh), flags=cv2.INTER_CUBIC)
    return out, np.vstack([rotation, [0, 0, 1]]) @ transform


def crop_box(
    img: np.ndarray, box: Sequence[float], margin: float = 0.08
) -> np.ndarray | None:
    """Crop an expanded detection box clipped to the image, or None without area."""
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    x1, y1 = int(min(img.shape[1], x1 + mx)), int(min(img.shape[0], y1 + my))
    if x1 - x0 < MIN_FACE_CROP_SIDE or y1 - y0 < MIN_FACE_CROP_SIDE:
        return None
    return img[y0:y1, x0:x1]


def to_crop(img: np.ndarray) -> np.ndarray:
    """Resize a BGR face crop without changing channel order."""
    return cv2.resize(
        img,
        (CROP_W, CROP_H),
        interpolation=cv2.INTER_AREA if img.shape[0] > CROP_H else cv2.INTER_CUBIC,
    )
