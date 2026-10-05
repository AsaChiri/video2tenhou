# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Build cached frames and detector/classifier datasets from labels/<video>/boxes.

Split is by hand of the game (HELD_OUT_HANDS are never trained on), never by random
frame, so the metrics say how the models do on unseen play.

    uv run python -m video2tenhou.train.data samples/full_1080p.mp4 --out work/datasets
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from video2tenhou import video
from video2tenhou.files import atomic_write_json
from video2tenhou.layout import Calibration, quad_to_box
from video2tenhou.logging_setup import RESULT, command_logging
from video2tenhou.paths import DATA_DIR as ROOT
from video2tenhou.perception.crops import (
    CROP_H,
    CROP_W,
    crop_box,
    region_upright,
    to_crop,
)
from video2tenhou.perception.tiles import CLASS_INDEX, CLASSES

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

MIN_DETECTOR_BOX_SIDE = 4


HELD_OUT_HANDS = (4, 9, 16, 20)


LOGGER = logging.getLogger("video2tenhou.train.data")


def labels_dir(video_path: str | Path) -> Path:
    """Return the annotation directory keyed by video stem.

    Do not reuse a stem for different recordings.
    """
    return ROOT / "labels" / Path(video_path).stem


def load_labels(video_path: str | Path) -> list[dict]:
    """Load active box annotations in stable filename order, excluding .orig backups."""
    out = []
    for p in sorted(labels_dir(video_path).glob("boxes/*.json")):
        if p.name.endswith(".orig.json"):
            continue
        out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def hand_table(video_path: str | Path, work: Path) -> list[dict]:
    """Read the pipeline's hand table used to keep training and validation separate."""
    p = work / Path(video_path).stem / "hands.json"
    return json.loads(p.read_text(encoding="utf-8"))


def hand_of(t: float, hands: list[dict]) -> int | None:
    """Attribute a label to the physical hand window containing its timestamp."""
    return next(
        (i for i, hand in enumerate(hands) if hand["t_start"] <= t <= hand["t_end"]),
        None,
    )


def held_out(hand: int | None) -> bool:
    """Report whether a physical hand belongs to validation, never to training."""
    return hand in HELD_OUT_HANDS


def split_of(t: float, hands: list[dict]) -> str:
    """Split frames by whole hand to prevent training/validation leakage."""
    return "val" if held_out(hand_of(t, hands)) else "train"


def annotation_key(annotation: dict) -> str:
    """Name a region annotation as dataset files do; the time is rounded.

    Rounding can move a time across a hand boundary, so splits always use the
    annotation's exact ``t``, never this name.
    """
    return f"{annotation['kind']}_{annotation['corner']}_{round(annotation['t'])}"


def annotation_key_of(path: str | Path) -> str:
    """Recover the annotation key that prefixes a dataset file name."""
    name = re.split(r"[\\/]", str(path))[-1]  # Manifests come from either OS.
    match = re.match(r"(hand|pond|meld)_(TL|TR|BL|BR)_\d+", name)
    if not match:
        raise ValueError(f"Cannot locate original human annotation for {path}")
    return match.group()


def clipped_box(
    transform: np.ndarray, quad: list, width: int, height: int
) -> tuple[float, float, float, float]:
    """Map a reviewed frame quad into region pixels, clipped to the region."""
    x0, y0, x1, y1 = quad_to_box(transform, quad)
    return max(0.0, x0), max(0.0, y0), min(float(width), x1), min(float(height), y1)


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def ensure_frames(
    video_path: str | Path,
    times: Iterable[float],
    work: Path,
    log: Callable[[str], None] = LOGGER.info,
) -> dict[float, Path]:
    """Cache lossless frames and return timestamp-to-path mappings."""
    d = work / Path(video_path).stem / "frames"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    todo = []
    for t in sorted({float(t) for t in times}):
        p = d / f"{t:.3f}.png"
        out[t] = p
        if not p.exists():
            todo.append((t, p))
    for i, (t, p) in enumerate(todo):
        cv2.imwrite(str(p), video.frame_at(video_path, t))
        if (i + 1) % 25 == 0:
            log(f"  frames: {i + 1}/{len(todo)}")
    return out


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CropSource:
    """Stable output identity and held-out split for an annotated region."""

    split: str
    name: str
    kind: str


@dataclass
class TrainingWriter:
    """Write detector labels and classifier crops from reviewed regions."""

    det: Path
    clf: Path
    stats: Counter = field(default_factory=Counter)

    def negative(self, img: np.ndarray, source: CropSource) -> None:
        """Keep reviewed empty melds and sample their background patches."""
        cv2.imwrite(
            str(self.det / "images" / source.split / f"{source.name}.jpg"),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        (self.det / "labels" / source.split / f"{source.name}.txt").write_text("")
        self.stats[f"det_{source.split}_negatives"] += 1
        # and tile-sized patches of it (rails, cloth, shadows) are "none" crops
        # for the classifier
        h, w = img.shape[:2]
        for j in range(3):
            bw, bh = 90, 105
            x0 = random.uniform(0, max(1, w - bw))
            y0 = random.uniform(0, max(1, h - bh))
            c = crop_box(img, (x0, y0, x0 + bw, y0 + bh), margin=0.0)
            if c is not None:
                cv2.imwrite(
                    str(
                        self.clf / source.split / "none" / f"{source.name}_empty{j}.png"
                    ),
                    to_crop(c),
                )
                self.stats[f"clf_{source.split}_none"] += 1

    def labelled(self, img: np.ndarray, boxes: list[tuple], source: CropSource) -> None:
        """Write face boxes for the detector and every tile for the classifier.

        The detector localizes faces only. A region with reviewed face-down
        tiles is left out of its dataset rather than teaching them as background.
        """
        if any(b["tile"] == "X" for _, b in boxes):
            self.stats[f"det_{source.split}_back_images_excluded"] += 1
        else:
            self._detector(img, boxes, source)
        self._classifier(img, boxes, source)
        self._background(img, boxes, source)

    def _detector(
        self, img: np.ndarray, boxes: list[tuple], source: CropSource
    ) -> None:
        """Write the region image with one class-0 YOLO line per face box."""
        h, w = img.shape[:2]
        cv2.imwrite(
            str(self.det / "images" / source.split / f"{source.name}.jpg"),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        lines = [
            f"0 {(x0 + x1) / 2 / w:.5f} {(y0 + y1) / 2 / h:.5f} "
            f"{(x1 - x0) / w:.5f} {(y1 - y0) / h:.5f}"
            for (x0, y0, x1, y1), _ in boxes
        ]
        (self.det / "labels" / source.split / f"{source.name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        self.stats[f"det_{source.split}_images"] += 1
        self.stats[f"det_{source.split}_boxes"] += len(lines)

    def _classifier(
        self, img: np.ndarray, boxes: list[tuple], source: CropSource
    ) -> None:
        """Keep tile identities and both possible upright sideways rotations."""
        # classifier: labelled crops, upright; sideways tiles in both 90-degree turns
        for i, ((x0, y0, x1, y1), b) in enumerate(boxes):
            tile = b["tile"]
            if tile not in CLASS_INDEX or tile == "none":
                continue
            c = crop_box(img, (x0, y0, x1, y1))
            if c is None:
                continue
            variants = [c]
            if b.get("sideways"):
                variants = [
                    cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE),
                    cv2.rotate(c, cv2.ROTATE_90_COUNTERCLOCKWISE),
                ]
            for j, v in enumerate(variants):
                cv2.imwrite(
                    str(
                        self.clf
                        / source.split
                        / tile
                        / f"{source.name}_{i:02d}_{j}.png"
                    ),
                    to_crop(v),
                )
                self.stats[f"clf_{source.split}_{source.kind}"] += 1

    def _background(
        self, img: np.ndarray, boxes: list[tuple], source: CropSource
    ) -> None:
        """Sample classifier negatives that overlap no reviewed tile box."""
        h, w = img.shape[:2]
        # classifier negatives: background patches that overlap no labelled box
        bw = np.median([x1 - x0 for (x0, _, x1, _), _ in boxes]) if boxes else 40
        bh = np.median([y1 - y0 for (_, y0, _, y1), _ in boxes]) if boxes else 60
        for j in range(2):
            for _ in range(20):
                x0 = random.uniform(0, max(1, w - bw))
                y0 = random.uniform(0, max(1, h - bh))
                x1, y1 = x0 + bw, y0 + bh
                if all(
                    min(x1, bx1) - max(x0, bx0) <= 0 or min(y1, by1) - max(y0, by0) <= 0
                    for (bx0, by0, bx1, by1), _ in boxes
                ):
                    c = crop_box(img, (x0, y0, x1, y1), margin=0.0)
                    if c is not None:
                        cv2.imwrite(
                            str(
                                self.clf
                                / source.split
                                / "none"
                                / f"{source.name}_bg{j}.png"
                            ),
                            to_crop(c),
                        )
                        self.stats[f"clf_{source.split}_none"] += 1
                    break


def _training_boxes(d: dict, img: np.ndarray, transform: np.ndarray) -> list[tuple]:
    """Clip reviewed quads to the region and exclude degenerate boxes."""
    h, w = img.shape[:2]
    boxes = []
    for b in d["boxes"]:
        x0, y0, x1, y1 = clipped_box(transform, b["quad"], w, h)
        if x1 - x0 < MIN_DETECTOR_BOX_SIDE or y1 - y0 < MIN_DETECTOR_BOX_SIDE:
            continue
        boxes.append(((x0, y0, x1, y1), b))
    return boxes


def build(
    video_path: str | Path,
    cal: Calibration,
    work: Path,
    out: Path,
    *,
    seed: int = 0,
) -> dict:
    """Write detector and classifier datasets from human labels, split by hand."""
    random.seed(seed)
    labels = load_labels(video_path)
    hands = hand_table(video_path, work)
    frames = ensure_frames(video_path, [d["t"] for d in labels], work, LOGGER.info)
    det = out / "detector"
    clf = out / "classifier"
    for split in ("train", "val"):
        (det / "images" / split).mkdir(parents=True, exist_ok=True)
        (det / "labels" / split).mkdir(parents=True, exist_ok=True)
        for c in CLASSES:
            (clf / split / c).mkdir(parents=True, exist_ok=True)
    writer = TrainingWriter(det, clf)
    for d in labels:
        kind, corner, t = d["kind"], d["corner"], float(d["t"])
        if not d["boxes"] and not (kind == "meld" and d.get("melds") == []):
            continue
        frame = cv2.imread(str(frames[t]))
        if frame is None:
            raise OSError(f"Cannot read cached training frame: {frames[t]}")
        img, transform = region_upright(frame, cal, kind, corner)
        source = CropSource(split_of(t, hands), annotation_key(d), kind)
        if d["boxes"]:
            writer.labelled(img, _training_boxes(d, img, transform), source)
        else:
            writer.negative(img, source)
    stats = writer.stats
    (det / "data.yaml").write_text(
        f"path: {det.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        "nc: 1\nnames:\n  0: face\n"
    )
    atomic_write_json(
        out / "meta.json",
        {
            "classes": CLASSES,
            "crop": [CROP_W, CROP_H],
            "held_out_hands": HELD_OUT_HANDS,
            "stats": dict(stats),
        },
        indent=1,
    )
    return dict(stats)


@command_logging
def main(argv: list[str] | None = None) -> None:
    """Build training data from measured geometry and local annotations."""
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--calib", default="pml")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="work/datasets")
    a = ap.parse_args(argv)
    stats = build(
        a.video, Calibration.load(a.calib, a.video), Path(a.work), Path(a.out)
    )
    RESULT.info("%s", json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
