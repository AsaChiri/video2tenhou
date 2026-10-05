# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Export unreviewed face-box drafts for the images in a ZIP archive.

Images keep their original bytes under generated names, so archive paths are
never used as output paths. Predictions, normalized YOLO drafts and a manifest
stay separate from accepted annotations; no training split or data.yaml is made.
A failed run keeps its partial output with ``complete: false`` in summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from video2tenhou.files import sha256_file
from video2tenhou.perception import detector

if TYPE_CHECKING:
    from collections.abc import Sequence

    from video2tenhou.perception.detector import Det

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
POLICY = """# Unreviewed detector drafts

Every box is a machine prediction, not ground truth. `draft_labels/` holds
normalized YOLO boxes for class 0 (face); `predictions/` keeps raw and clipped
pixel coordinates with confidences; `manifest.jsonl` links each file to its
archive member and image hash. Zero detections do not show an empty scene, and
empty draft files are not reviewed negatives. Review missed objects, extra boxes
and box edges before accepting an image. Use only complete exports
(`summary.json`: complete=true), keep human-reviewed evaluation independent of
these drafts and never merge them into human labels automatically.
"""

LOGGER = logging.getLogger("tools.pseudolabel_images")


class FaceDetector(Protocol):
    """The face detector's recognition identity, threshold and prediction call."""

    id: str
    conf: float

    def predict(self, image: np.ndarray, /) -> Sequence[Det]:
        """Return face boxes in the image's pixel coordinates."""
        ...


def image_members(archive: zipfile.ZipFile) -> list[str]:
    """Name the archive's image files; other members are ignored."""
    members = [
        info.filename
        for info in archive.infolist()
        if not info.is_dir()
        and PurePosixPath(info.filename).suffix.lower() in IMAGE_SUFFIXES
    ]
    if not members:
        raise ValueError("Archive contains no supported image files")
    return members


def draft_box(detection: Det, width: int, height: int) -> dict:
    """Clip a prediction to the image and add its normalized YOLO box."""
    raw = [float(value) for value in detection.xyxy]
    x0, x1 = (min(max(value, 0.0), width) for value in (raw[0], raw[2]))
    y0, y1 = (min(max(value, 0.0), height) for value in (raw[1], raw[3]))
    if not (x1 > x0 and y1 > y0 and 0 <= detection.conf <= 1):
        raise ValueError(f"Invalid detection {raw} with confidence {detection.conf}")
    return {
        "raw_xyxy": raw,
        "xyxy": [x0, y0, x1, y1],
        "confidence": float(detection.conf),
        "class_id": 0,
        "yolo": [
            (x0 + x1) / (2 * width),
            (y0 + y1) / (2 * height),
            (x1 - x0) / width,
            (y1 - y0) / height,
        ],
    }


def write_drafts(
    output: Path, member: str, content: bytes, model: FaceDetector
) -> dict:
    """Write one image, its predictions and its YOLO drafts; return its manifest row."""
    digest = hashlib.sha256(content).hexdigest()
    source = PurePosixPath(member)
    stem = re.sub(r"[^A-Za-z0-9_-]", "_", source.stem)[:60] or "image"
    name = f"{stem}_{hashlib.sha256(member.encode()).hexdigest()[:16]}"
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {member}")
    height, width = image.shape[:2]
    boxes = [draft_box(found, width, height) for found in model.predict(image)]
    row = {
        "member": member,
        "sha256": digest,
        "width": width,
        "height": height,
        "image": f"images/{name}{source.suffix.lower()}",
        "predictions": f"predictions/{name}.json",
        "draft_labels": f"draft_labels/{name}.txt",
        "detections": len(boxes),
        "status": "unreviewed_predictions" if boxes else "unreviewed_zero_detections",
        "reviewed": False,
    }
    (output / row["image"]).write_bytes(content)
    (output / row["predictions"]).write_text(
        json.dumps({**row, "boxes": boxes}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output / row["draft_labels"]).write_text(
        "".join(
            "0 " + " ".join(f"{value:.9f}" for value in box["yolo"]) + "\n"
            for box in boxes
        ),
        encoding="utf-8",
    )
    return row


def export_drafts(
    archive_path: Path,
    weights: Path,
    output: Path,
    *,
    device: str | None = None,
    model: FaceDetector | None = None,
) -> dict:
    """Export drafts once into a new directory, leaving the archive unchanged.

    The production detector applies its checkpoint metadata's inference
    defaults; ``model`` substitutes a detector for tests. Errors propagate after
    an incomplete summary and the failing image's manifest row are saved.
    """
    output.mkdir(parents=True, exist_ok=False)
    for name in ("images", "predictions", "draft_labels"):
        (output / name).mkdir()
    (output / "README.md").write_text(POLICY, encoding="utf-8")
    summary = {
        "complete": False,
        "reviewed": False,
        "archive": str(archive_path.resolve()),
        "archive_sha256": sha256_file(archive_path),
        "weights": str(weights.resolve()),
        "images": 0,
        "images_without_detections": 0,
        "detections": 0,
        "error": None,
    }
    try:
        if model is None:
            model = detector.Detector(weights, device=device)
        summary.update(recognition_id=model.id, confidence=model.conf)
        with (
            zipfile.ZipFile(archive_path) as archive,
            (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest,
        ):
            for member in image_members(archive):
                row: dict = {"member": member, "status": "error", "reviewed": False}
                try:
                    row = write_drafts(output, member, archive.read(member), model)
                except Exception as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    raise
                finally:
                    manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                summary["images"] += 1
                summary["images_without_detections"] += not row["detections"]
                summary["detections"] += row["detections"]
        summary["complete"] = True
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return summary


def main(argv: list[str] | None = None) -> None:
    """Export unreviewed predictions, returning a nonzero exit on failure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--device",
        help="Detector device, e.g. cpu or 0; default chooses CUDA when available",
    )
    args = parser.parse_args(argv)
    summary = export_drafts(args.archive, args.weights, args.out, device=args.device)
    LOGGER.info(
        "%s images, %s detections: %s",
        summary["images"],
        summary["detections"],
        args.out,
    )


if __name__ == "__main__":
    main()
