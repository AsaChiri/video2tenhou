"""Reproducible tile perception timing with recognition equivalence checks.

Run ``uv run python -m video2tenhou.benchmark --images work/datasets/detector/images``.
Images must use the training dataset's ``kind_corner_*.jpg`` naming. This
benchmark compares the former per-region path with the production buffer;
it does not estimate whole-video runtime or replace held-out accuracy tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .perception.classifier import Classifier
    from .perception.detector import Detector


def compare_images(directory: Path, detector: Detector, classifier: Classifier) -> dict:
    """Compare all labeled region JPEGs, excluding image I/O and warmup timing.

    Report count, tile identity and structural disagreements separately from
    posterior drift. CUDA reductions can change probabilities slightly while
    preserving every label; this is reported rather than claimed bit-exact.
    """
    from .perception.reader import read_region, read_regions
    from .read import READ_BATCH

    paths = sorted(directory.rglob("*.jpg"))
    if not paths:
        raise ValueError(f"No JPEG region images found in {directory}")
    stats = dict(
        images=0,
        boxes=0,
        count_changes=0,
        tile_changes=0,
        structure_changes=0,
        rejected_changes=0,
        max_posterior_delta=0.0,
        max_box_delta=0.0,
        serial_seconds=0.0,
        buffered_seconds=0.0,
    )
    changed_images = []
    for start in range(0, len(paths), READ_BATCH):
        chunk = paths[start : start + READ_BATCH]
        items = []
        for path in chunk:
            parts = path.stem.split("_")
            if len(parts) < 2 or parts[0] not in {"pond", "hand", "meld"}:
                raise ValueError(f"Expected kind_corner_timestamp.jpg: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Cannot read {path}")
            items.append((0.0, ":".join(parts[:2]), img))

        def serial():
            return [
                read_region(None, None, name, detector, classifier, t=t, img=img)
                for t, name, img in items
            ]

        if start == 0:
            serial()
            read_regions(items, detector, classifier)
        began = perf_counter()
        before = serial()
        stats["serial_seconds"] += perf_counter() - began
        began = perf_counter()
        after = read_regions(items, detector, classifier)
        stats["buffered_seconds"] += perf_counter() - began
        for path, a, b in zip(chunk, before, after, strict=False):
            stats["images"] += 1
            changed = a.rejected != b.rejected
            stats["rejected_changes"] += int(changed)
            if len(a.boxes) != len(b.boxes):
                stats["count_changes"] += 1
                changed_images.append(str(path))
                continue
            for x, y in zip(a.boxes, b.boxes, strict=False):
                stats["boxes"] += 1
                tile_changed = int(x.p.argmax() != y.p.argmax())
                structure_changed = int(
                    (x.role, x.row, x.col, x.group, x.sideways)
                    != (y.role, y.row, y.col, y.group, y.sideways)
                )
                stats["tile_changes"] += tile_changed
                stats["structure_changes"] += structure_changed
                stats["max_posterior_delta"] = max(
                    stats["max_posterior_delta"], float(np.max(np.abs(x.p - y.p)))
                )
                stats["max_box_delta"] = max(
                    stats["max_box_delta"],
                    float(np.max(np.abs(np.asarray(x.xyxy) - y.xyxy))),
                )
                changed |= bool(tile_changed or structure_changed)
            if changed:
                changed_images.append(str(path))
    stats["speedup"] = stats["serial_seconds"] / max(stats["buffered_seconds"], 1e-9)
    return {"metrics": stats, "changed_images": changed_images}


def main(argv=None) -> int:
    """Print benchmark JSON and return nonzero when recognition changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images", type=Path, required=True, help="Region JPEG dataset directory"
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    args = parser.parse_args(argv)
    from .perception.classifier import Classifier
    from .perception.detector import Detector

    result = compare_images(args.images, Detector(), Classifier())
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    return int(bool(result["changed_images"]))


if __name__ == "__main__":
    raise SystemExit(main())
