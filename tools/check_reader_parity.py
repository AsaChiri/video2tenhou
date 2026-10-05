# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Check that batched region reading matches one-region-at-a-time reading.

    uv run python tools/check_reader_parity.py --images work/datasets/detector/images

Images use the detector dataset's ``kind_corner_*.jpg`` names. Both paths run
the local detector and classifier; only classifier batching differs, because
detector batches are processed one image at a time. Box counts, tile
identities, roles, rows, groups, orientation and rejection must match; the
largest posterior and coordinate differences are reported. This checks
equivalence only; it measures no speed.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from video2tenhou.perception.classifier import Classifier
from video2tenhou.perception.detector import Detector
from video2tenhou.perception.reader import (
    Reading,
    RegionClassifier,
    RegionDetector,
    read_region,
    read_regions,
)
from video2tenhou.read import READ_BATCH

LOGGER = logging.getLogger("tools.check_reader_parity")


def region(path: Path) -> tuple[float, str, np.ndarray]:
    """Load one dataset image as a ``(time, region name, BGR image)`` item."""
    kind, corner, *_ = path.stem.split("_")
    image = cv2.imread(str(path))
    if kind not in {"hand", "pond", "meld"} or image is None:
        raise ValueError(f"Expected a readable kind_corner_*.jpg region image: {path}")
    return 0.0, f"{kind}:{corner}", image


def differs(serial: Reading, batched: Reading) -> bool:
    """Report any change in what the evidence stages would consume."""
    return serial.rejected != batched.rejected or [
        (int(box.p.argmax()), box.role, box.row, box.col, box.group, box.sideways)
        for box in serial.boxes
    ] != [
        (int(box.p.argmax()), box.role, box.row, box.col, box.group, box.sideways)
        for box in batched.boxes
    ]


def compare(
    images: Path, detector: RegionDetector, classifier: RegionClassifier
) -> dict:
    """Read every image both ways and list those whose readings differ."""
    paths = sorted(images.rglob("*.jpg"))
    if not paths:
        raise ValueError(f"No JPEG region images found in {images}")
    changed, posterior, coordinate = [], 0.0, 0.0
    for start in range(0, len(paths), READ_BATCH):
        chunk = paths[start : start + READ_BATCH]
        items = [region(path) for path in chunk]
        batched = read_regions(items, detector, classifier)
        for path, (t, name, image), after in zip(chunk, items, batched, strict=True):
            before = read_region(image, name, detector, classifier, t=t)
            if differs(before, after):
                changed.append(str(path))
                continue
            for x, y in zip(before.boxes, after.boxes, strict=True):
                posterior = max(posterior, float(np.abs(x.p - y.p).max()))
                coordinate = max(
                    coordinate, float(np.abs(np.subtract(x.xyxy, y.xyxy)).max())
                )
    return {
        "images": len(paths),
        "changed_images": changed,
        "max_posterior_delta": posterior,
        "max_box_delta": coordinate,
    }


def main(argv: list[str] | None = None) -> int:
    """Print the comparison and return nonzero when any reading changed."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args(argv)
    result = compare(args.images, Detector(), Classifier())
    LOGGER.info("%s", json.dumps(result, indent=2))
    return int(bool(result["changed_images"]))


if __name__ == "__main__":
    raise SystemExit(main())
