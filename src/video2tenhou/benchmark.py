"""Reproducible perception and header timing with recognition equivalence checks.

Run ``uv run python -m video2tenhou.benchmark --images work/datasets/detector/images``.
Images must use the training dataset's ``kind_corner_*.jpg`` naming. This
benchmark compares the former per-region path with the production buffer;
it does not estimate whole-video runtime or replace held-out accuracy tests.
Use ``--video recording.mp4 --window 450 750`` for a streaming header comparison
requiring packaged glyph templates and ffmpeg, but no detector/classifier weights.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .layout import Calibration
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
    stats = dict(images=0, boxes=0, count_changes=0, tile_changes=0,
                 structure_changes=0, rejected_changes=0, max_posterior_delta=0.,
                 max_box_delta=0., serial_seconds=0., buffered_seconds=0.)
    changed_images = []
    for start in range(0, len(paths), READ_BATCH):
        chunk = paths[start:start + READ_BATCH]
        items = []
        for path in chunk:
            parts = path.stem.split("_")
            if len(parts) < 2 or parts[0] not in {"pond", "hand", "meld"}:
                raise ValueError(f"Expected kind_corner_timestamp.jpg: {path}")
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Cannot read {path}")
            items.append((0., ":".join(parts[:2]), img))

        def serial():
            return [read_region(None, None, name, detector, classifier, t=t, img=img)
                    for t, name, img in items]

        if start == 0:
            serial()
            read_regions(items, detector, classifier)
        began = perf_counter()
        before = serial()
        stats["serial_seconds"] += perf_counter() - began
        began = perf_counter()
        after = read_regions(items, detector, classifier)
        stats["buffered_seconds"] += perf_counter() - began
        for path, a, b in zip(chunk, before, after):
            stats["images"] += 1
            changed = a.rejected != b.rejected
            stats["rejected_changes"] += int(changed)
            if len(a.boxes) != len(b.boxes):
                stats["count_changes"] += 1
                changed_images.append(str(path))
                continue
            for x, y in zip(a.boxes, b.boxes):
                stats["boxes"] += 1
                tile_changed = int(x.p.argmax() != y.p.argmax())
                structure_changed = int((x.role, x.row, x.col, x.group, x.sideways) !=
                                        (y.role, y.row, y.col, y.group, y.sideways))
                stats["tile_changes"] += tile_changed
                stats["structure_changes"] += structure_changed
                stats["max_posterior_delta"] = max(stats["max_posterior_delta"], float(np.max(np.abs(x.p - y.p))))
                stats["max_box_delta"] = max(stats["max_box_delta"], float(np.max(np.abs(np.asarray(x.xyxy) - y.xyxy))))
                changed |= bool(tile_changed or structure_changed)
            if changed:
                changed_images.append(str(path))
    stats["speedup"] = stats["serial_seconds"] / max(stats["buffered_seconds"], 1e-9)
    return {"metrics": stats, "changed_images": changed_images}


def compare_header(path: Path, windows: list[tuple[float, float]], cal: Calibration, fps: float = 1.) -> dict:
    """Stream each window twice and compare original scalar versus production digits.

    Report the complete sample/read pass, time waiting for sampled frames and
    recognition time separately. Template loading is excluded; production starts
    with a fresh matcher and retains its bounded cache across the given windows.
    No frames are retained. Small serialized header rows are retained for exact
    comparison, including missing values and checksum validity. Names are disabled
    as in the header stage; unreadable digits can still invoke Tesseract.

    This standalone benchmark temporarily replaces overlay's global matcher and
    must not run alongside other overlay readers. It restores the original object
    even when sampling or recognition fails. Timing order is scalar then production;
    repeat runs to assess disk-cache and scheduling effects.
    """
    from . import overlay, video
    from .timeline import reading_from_state, valid

    if not windows or not math.isfinite(fps) or fps <= 0 or any(
            not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start
            for start, end in windows):
        raise ValueError("Use positive fps and nonempty windows with 0 <= START < END.")

    class ScalarDigits(overlay.DigitMatcher):
        def classify(self, mask):
            sig = overlay.digit_signature(mask)
            if sig is None:
                return None, 0.
            best, score = None, -1.
            for label, templates in self.templates.items():
                for template in templates:
                    candidate = overlay._ncc(sig, template)
                    if candidate > score:
                        best, score = label, candidate
            return best, score

    original = overlay._DIGITS
    passes = {}
    rows = {}
    try:
        for name, matcher in (("scalar", ScalarDigits()), ("production", overlay.DigitMatcher())):
            overlay._DIGITS = matcher
            overlay._tess_pixels.cache_clear()
            metrics = dict(frames=0, total_seconds=0., sampling_seconds=0., recognition_seconds=0.)
            readings = []
            for window_index, (start, end) in enumerate(windows):
                began = perf_counter()
                samples = video.sample(path, fps=fps, start=start, end=end)
                try:
                    while True:
                        sampled = perf_counter()
                        try:
                            t, frame = next(samples)
                        except StopIteration:
                            metrics["sampling_seconds"] += perf_counter() - sampled
                            break
                        metrics["sampling_seconds"] += perf_counter() - sampled
                        recognized = perf_counter()
                        state = overlay.read_overlay(frame, names=False, cal=cal)
                        metrics["recognition_seconds"] += perf_counter() - recognized
                        row = reading_from_state(t, state)
                        row["ok"] = valid(row)
                        readings.append({"window": window_index, **row})
                        metrics["frames"] += 1
                finally:
                    samples.close()
                metrics["total_seconds"] += perf_counter() - began
            passes[name], rows[name] = metrics, readings
    finally:
        overlay._DIGITS = original
    if not rows["scalar"] and not rows["production"]:
        raise ValueError("No frames were sampled; check the video and window bounds.")
    changed = []
    for i in range(max(len(rows["scalar"]), len(rows["production"]))):
        before = rows["scalar"][i] if i < len(rows["scalar"]) else None
        after = rows["production"][i] if i < len(rows["production"]) else None
        if before != after:
            changed.append({"index": i, "scalar": before, "production": after})
    return {"video": str(path), "windows": windows, "fps": fps, "metrics": passes,
            "rows_equal": not changed, "changed_rows": changed,
            "total_speedup": passes["scalar"]["total_seconds"] / max(passes["production"]["total_seconds"], 1e-9),
            "recognition_speedup": passes["scalar"]["recognition_seconds"] / max(passes["production"]["recognition_seconds"], 1e-9)}


def main(argv=None) -> int:
    """Print benchmark JSON and return nonzero when recognition changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="Region JPEG dataset directory (requires model weights)")
    source.add_argument("--video", type=Path, help="Header benchmark video (no model weights required)")
    parser.add_argument("--window", type=float, nargs=2, action="append", metavar=("START", "END"), help="Header interval in seconds; repeat for multiple windows")
    parser.add_argument("--fps", type=float, default=1., help="Header sampling rate (default: 1)")
    parser.add_argument("--calib", default="pml", help="Header layout name or JSON path")
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    args = parser.parse_args(argv)
    if args.video:
        if not args.window:
            parser.error("--video requires at least one --window START END")
        from .layout import Calibration
        result = compare_header(args.video, args.window, Calibration.load(args.calib, args.video), fps=args.fps)
    else:
        if args.window or args.calib != "pml" or args.fps != 1.:
            parser.error("--window, --calib and --fps apply only to --video")
        from .perception.classifier import Classifier
        from .perception.detector import Detector
        result = compare_images(args.images, Detector(), Classifier())
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    return int(bool(result.get("changed_images", result.get("changed_rows"))))


if __name__ == "__main__":
    raise SystemExit(main())
