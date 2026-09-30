"""Training data from labels/<video>/boxes: cached frames, detector and classifier datasets.

Split is by hand of the game (HELD_OUT_HANDS are never trained on), never by
random frame, so the metrics say how the models do on unseen play.

    uv run python -m video2tenhou.train.data samples/full_1080p.mp4 --out work/datasets
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .. import video
from ..layout import Calibration, quad_to_box
from ..paths import DATA_DIR as ROOT

HELD_OUT_HANDS = (4, 9, 16, 20)
KINDS = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + [
    f"{n}z" for n in range(1, 8)
]
CLASSES = KINDS + ["0m", "0p", "0s", "X", "none"]  # 39
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
CROP_W, CROP_H = 64, 96
DET_CLASSES = ["face", "back"]


def labels_dir(video_path: str | Path) -> Path:
    """Local annotation directory keyed by video stem; do not reuse a stem for different recordings."""
    return ROOT / "labels" / Path(video_path).stem


def load_labels(video_path: str | Path) -> list[dict]:
    """Load active box annotations in stable filename order, excluding .orig backups."""
    out = []
    for p in sorted(labels_dir(video_path).glob("boxes/*.json")):
        if p.name.endswith(".orig.json"):
            continue
        out.append(json.load(open(p, encoding="utf-8")))
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


def split_of(t: float, hands: list[dict]) -> str:
    """Assign a frame to train/validation by whole hand to avoid adjacent-frame leakage."""
    return "val" if hand_of(t, hands) in HELD_OUT_HANDS else "train"


# ---------------------------------------------------------------------------
# frames
# ---------------------------------------------------------------------------


def ensure_frames(
    video_path: str | Path, times, work: Path, log=print
) -> dict[float, Path]:
    """Cache lossless frames for unique timestamps and return timestamp-to-path mappings."""
    d = work / Path(video_path).stem / "frames"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    todo = []
    for t in sorted(set(float(t) for t in times)):
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
# region rendering with roll
# ---------------------------------------------------------------------------


def region_upright(
    frame: np.ndarray, cal: Calibration, kind: str, corner: str
) -> tuple[np.ndarray, np.ndarray]:
    """Region image turned so tiles stand upright, and the frame->image matrix.

    Ponds and melds are upright already; hand bands are rotated by -roll about
    their centre (the canvas grows so nothing is cut off).
    """
    img, M = cal.region(frame, f"{kind}:{corner}")
    roll = cal.roll(corner) if kind == "hand" else 0.0
    if abs(roll) < 0.5:
        return img, M
    h, w = img.shape[:2]
    R = cv2.getRotationMatrix2D(
        (w / 2, h / 2), roll, 1.0
    )  # positive angle = counter-clockwise: undoes a descending row
    cos, sin = abs(R[0, 0]), abs(R[0, 1])
    nw, nh = int(w * cos + h * sin), int(w * sin + h * cos)
    R[0, 2] += nw / 2 - w / 2
    R[1, 2] += nh / 2 - h / 2
    out = cv2.warpAffine(img, R, (nw, nh), flags=cv2.INTER_CUBIC)
    return out, np.vstack([R, [0, 0, 1]]) @ M


def crop_box(img: np.ndarray, box, margin: float = 0.08) -> np.ndarray | None:
    """Crop an expanded detection box, clipped to the image; return None when it has no area."""
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    x1, y1 = int(min(img.shape[1], x1 + mx)), int(min(img.shape[0], y1 + my))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return img[y0:y1, x0:x1]


def to_crop(img: np.ndarray) -> np.ndarray:
    """Resize a BGR face crop to the classifier input size without changing channel order."""
    return cv2.resize(
        img,
        (CROP_W, CROP_H),
        interpolation=cv2.INTER_AREA if img.shape[0] > CROP_H else cv2.INTER_CUBIC,
    )


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------


def build(
    video_path: str | Path,
    cal: Calibration,
    work: Path,
    out: Path,
    *,
    seed: int = 0,
    log=print,
) -> dict:
    """Write detector and classifier datasets from human labels, using a hand-level validation split."""
    random.seed(seed)
    labels = load_labels(video_path)
    hands = hand_table(video_path, work)
    frames = ensure_frames(video_path, [d["t"] for d in labels], work, log)
    det = out / "detector"
    clf = out / "classifier"
    for split in ("train", "val"):
        (det / "images" / split).mkdir(parents=True, exist_ok=True)
        (det / "labels" / split).mkdir(parents=True, exist_ok=True)
        for c in CLASSES:
            (clf / split / c).mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    for d in labels:
        kind, corner, t = d["kind"], d["corner"], float(d["t"])
        if not d["boxes"]:
            # an empty label (a meld camera with no meld) is a negative image for the detector
            if kind == "meld" and d.get("melds") == []:
                split = split_of(t, hands)
                frame = cv2.imread(str(frames[t]))
                img, _ = region_upright(frame, cal, kind, corner)
                name = f"{kind}_{corner}_{int(round(t))}"
                cv2.imwrite(
                    str(det / "images" / split / f"{name}.jpg"),
                    img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )
                (det / "labels" / split / f"{name}.txt").write_text("")
                stats[f"det_{split}_negatives"] += 1
                # and tile-sized patches of it (rails, cloth, shadows) are "none" crops for the classifier
                h, w = img.shape[:2]
                for j in range(3):
                    bw, bh = 90, 105
                    x0 = random.uniform(0, max(1, w - bw))
                    y0 = random.uniform(0, max(1, h - bh))
                    c = crop_box(img, (x0, y0, x0 + bw, y0 + bh), margin=0.0)
                    if c is not None:
                        cv2.imwrite(
                            str(clf / split / "none" / f"{name}_empty{j}.png"),
                            to_crop(c),
                        )
                        stats[f"clf_{split}_none"] += 1
            continue
        split = split_of(t, hands)
        frame = cv2.imread(str(frames[t]))
        img, M = region_upright(frame, cal, kind, corner)
        h, w = img.shape[:2]
        name = f"{kind}_{corner}_{int(round(t))}"
        boxes = []
        for b in d["boxes"]:
            x0, y0, x1, y1 = quad_to_box(M, b["quad"])
            x0, y0, x1, y1 = (
                max(0.0, x0),
                max(0.0, y0),
                min(float(w), x1),
                min(float(h), y1),
            )
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            boxes.append(((x0, y0, x1, y1), b))
        # detector: every box, class face (back when the tile is X)
        cv2.imwrite(
            str(det / "images" / split / f"{name}.jpg"),
            img,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        lines = []
        for (x0, y0, x1, y1), b in boxes:
            cls = 1 if b["tile"] == "X" else 0
            lines.append(
                f"{cls} {(x0 + x1) / 2 / w:.5f} {(y0 + y1) / 2 / h:.5f} {(x1 - x0) / w:.5f} {(y1 - y0) / h:.5f}"
            )
        (det / "labels" / split / f"{name}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else "")
        )
        stats[f"det_{split}_images"] += 1
        stats[f"det_{split}_boxes"] += len(lines)
        # classifier: labelled crops, upright; sideways tiles in both 90-degree turns
        for i, ((x0, y0, x1, y1), b) in enumerate(boxes):
            tile = b["tile"]
            if tile not in CLASS_INDEX or tile in ("none",):
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
                    str(clf / split / tile / f"{name}_{i:02d}_{j}.png"), to_crop(v)
                )
                stats[f"clf_{split}_{kind}"] += 1
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
                            str(clf / split / "none" / f"{name}_bg{j}.png"), to_crop(c)
                        )
                        stats[f"clf_{split}_none"] += 1
                    break
    nl = "\n"
    (det / "data.yaml").write_text(
        f"path: {det.resolve().as_posix()}{nl}train: images/train{nl}val: images/val{nl}names:{nl}"
        + "".join(f"  {i}: {c}{nl}" for i, c in enumerate(DET_CLASSES))
    )
    json.dump(
        {
            "classes": CLASSES,
            "crop": [CROP_W, CROP_H],
            "held_out_hands": HELD_OUT_HANDS,
            "stats": dict(stats),
        },
        open(out / "meta.json", "w"),
        indent=1,
    )
    return dict(stats)


def main(argv=None):
    """Build training datasets for a video using its measured layout and local annotations."""
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--calib", default="pml")
    ap.add_argument("--work", default="work")
    ap.add_argument("--out", default="work/datasets")
    a = ap.parse_args(argv)
    stats = build(
        a.video, Calibration.load(a.calib, a.video), Path(a.work), Path(a.out)
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
