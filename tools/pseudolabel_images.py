"""Generate unreviewed face-localization box drafts from a ZIP of images.

Images retain their original bytes. Predictions, normalized YOLO drafts and a
provenance manifest are separate from accepted annotations. No training split or
data.yaml is created. A failed run retains partial output marked incomplete.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import time
import zipfile


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = {0: "face"}
POLICY = """# Unreviewed detector drafts

Every box is a machine prediction, not ground truth. `draft_labels/` contains
normalized YOLO boxes using the class mapping in summary.json. The supported detector emits class 0=face. `predictions/` retains raw
coordinates, clipped coordinates and confidence. `manifest.jsonl` links these to
the exact archive member and image hash; `summary.json` records model provenance.
Images preserve their original bytes. Coordinates use OpenCV's decoded image.

Zero detections do not establish an empty scene. Empty draft files are NOT
approved negative labels. Inspect missed objects, box edges and localization classes
before accepting images. Low-confidence predictions are retained as candidates.
No data.yaml or train/validation split is supplied. Split by source/session and
duplicate group before training, and keep human-reviewed evaluation independent
of teacher-generated labels. Exact duplicate groups do not detect near duplicates.

Use only completed runs (`summary.json`: complete=true). On failure, keep the
partial manifest for diagnosis and choose a new destination to retry. Do not
merge these files into human labels automatically. Retain image and teacher
provenance when deciding whether the eventual dataset/model can be distributed.
"""


def sha256_file(path):
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_classes(names):
    """Require the face-only localization schema; identities belong to the classifier."""
    if names != CLASS_NAMES:
        raise ValueError(f"Expected detector classes {CLASS_NAMES!r}; checkpoint declares {names!r}")


def image_members(archive):
    """List safe, unique image members without extracting any archive paths.

    Reject traversal, drive/absolute names, symlinks and duplicate normalized
    image paths. Nonimage files are ignored and never written.
    """
    members = []
    seen = set()
    for member in archive.infolist():
        name = member.filename.replace("\\", "/")
        if member.is_dir() or PurePosixPath(name).suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = name.split("/")
        if (name.startswith("/") or any(part in ("", ".", "..") for part in parts)
                or any(":" in part or "\x00" in part for part in parts)
                or stat.S_ISLNK(member.external_attr >> 16)):
            raise ValueError(f"Unsafe image member: {member.filename!r}")
        key = name.casefold()
        if key in seen:
            raise ValueError(f"Duplicate image member path: {member.filename!r}")
        seen.add(key)
        members.append((member, name))
    if not members:
        raise ValueError("Archive contains no supported image files")
    return members


def normalized_box(detection, width, height):
    """Validate a prediction and return raw/clipped pixels plus normalized YOLO values.

    Out-of-image edges are clipped. Nonfinite, reversed, empty or wholly outside
    boxes fail the run instead of silently becoming training annotations.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    raw = [float(value) for value in detection.xyxy]
    confidence = float(detection.conf)
    if len(raw) != 4 or not all(math.isfinite(value) for value in [*raw, confidence]):
        raise ValueError("Detection coordinates and confidence must be finite")
    if not 0 <= confidence <= 1 or raw[2] <= raw[0] or raw[3] <= raw[1]:
        raise ValueError("Invalid detection area or confidence")
    x0, y0, x1, y1 = [max(0., min(value, limit)) for value, limit in zip(raw, [width, height, width, height])]
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Detection has no area inside the image")
    if detection.back is not False:
        raise ValueError("Face-only detector cannot export a back prediction")
    class_id, names = 0, CLASS_NAMES
    result = dict(raw_xyxy=raw, xyxy=[x0, y0, x1, y1], confidence=confidence,
                  class_id=class_id, class_name=names[class_id],
                  yolo=[(x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
                        (x1 - x0) / width, (y1 - y0) / height])
    result["back"] = False
    return result


def export_drafts(archive_path, weights, output, *, device=None, predictor=None):
    """Export drafts once into a new directory, leaving sources unchanged.

    Optional predictor injection supports offline tests; it must implement
    predict(BGR_image) and expose model.names. Production constructs the project's
    detector with the checkpoint metadata's inference defaults.
    Failure propagates after an incomplete
    summary and, for image failures, a manifest error row are saved.
    """
    archive_path, weights, output = (Path(value).resolve() for value in (archive_path, weights, output))
    if not archive_path.is_file() or not weights.is_file():
        raise FileNotFoundError("Archive and checkpoint must both be existing files")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("images", "predictions", "draft_labels"):
        (output / name).mkdir()
    (output / "README.md").write_text(POLICY, encoding="utf-8")
    started = time.perf_counter()
    summary = dict(schema_version=1, complete=False, reviewed=False, archive=str(archive_path), weights=str(weights),
                   settings={"device_requested": device},
                   class_mode="face", classes=None,
                   images_completed=0, images_zero_detections=0, detections=0,
                   confidence_bins={"below_0.5": 0, "0.5_to_below_0.8": 0, "0.8_to_1": 0},
                   class_counts={}, image_dimensions=None, versions={}, error=None)
    duplicates = defaultdict(list)
    try:
        summary["archive_sha256"] = sha256_file(archive_path)
        summary["weights_sha256"] = sha256_file(weights)
        summary["exporter_sha256"] = sha256_file(__file__)
        for package in ("video2tenhou", "libreyolo", "torch", "numpy", "opencv-python", "opencv-python-headless"):
            try:
                summary["versions"][package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                summary["versions"][package] = None
        with zipfile.ZipFile(archive_path) as archive:
            members = image_members(archive)
            summary["images_expected"] = len(members)
            if predictor is None:
                from video2tenhou.perception.detector import Detector
                predictor = Detector(weights, device=device)
            validate_classes(predictor.model.names)
            summary["recognition_id"] = getattr(predictor, "id", None)
            for setting in ("imgsz", "conf", "iou", "cuda_graph"):
                summary["settings"][setting] = getattr(predictor, setting, None)
            metadata_path = weights.with_name("meta.json")
            summary["metadata_sha256"] = sha256_file(metadata_path) if metadata_path.exists() else None
            summary["classes"] = dict(predictor.model.names)
            summary["class_counts"] = {class_id: 0 for class_id in summary["classes"]}
            summary["settings"]["device_actual"] = str(getattr(predictor, "device", device))
            import cv2
            import numpy as np
            emitted = set()
            with (output / "manifest.jsonl").open("x", encoding="utf-8") as manifest:
                for member, source_name in members:
                    row = dict(source_archive=str(archive_path), member=member.filename, normalized_member=source_name,
                               reviewed=False, status="error")
                    try:
                        content = archive.read(member)
                        digest = hashlib.sha256(content).hexdigest()
                        row["sha256"] = digest
                        source = PurePosixPath(source_name)
                        stem = re.sub(r"[^A-Za-z0-9_-]", "_", source.stem)[:60] or "image"
                        member_id = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:16]
                        name = f"{stem}_{member_id}_{digest[:16]}"
                        if name.casefold() in emitted:
                            raise ValueError(f"Output filename collision: {source_name!r}")
                        emitted.add(name.casefold())
                        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                        if image is None:
                            raise ValueError(f"Cannot decode image: {source_name}")
                        height, width = image.shape[:2]
                        row.update(width=width, height=height, image=f"images/{name}{source.suffix.lower()}",
                                   predictions=f"predictions/{name}.json", draft_labels=f"draft_labels/{name}.txt")
                        # Exclusive writes prevent accidental reuse even if a caller
                        # changes output contents during this single-process export.
                        with (output / row["image"]).open("xb") as stream:
                            stream.write(content)
                        boxes = [normalized_box(detection, width, height)
                                 for detection in predictor.predict(image)]
                        runtime_device = getattr(getattr(predictor.model, "predictor", None), "device",
                                                 getattr(predictor, "device", device))
                        summary["settings"]["device_actual"] = str(runtime_device)
                        row.update(detections=len(boxes), status="unreviewed_predictions" if boxes else "unreviewed_zero_detections")
                        with (output / row["predictions"]).open("x", encoding="utf-8") as stream:
                            json.dump(dict(member=source_name, sha256=digest, width=width, height=height,
                                           reviewed=False, status=row["status"], boxes=boxes), stream, indent=2, allow_nan=False)
                        with (output / row["draft_labels"]).open("x", encoding="utf-8") as stream:
                            for box in boxes:
                                stream.write(str(box["class_id"]) + " " + " ".join(f"{value:.9f}" for value in box["yolo"]) + "\n")
                        duplicates[digest].append(source_name)
                        summary["images_completed"] += 1
                        summary["images_zero_detections"] += int(not boxes)
                        summary["detections"] += len(boxes)
                        dimensions = summary["image_dimensions"]
                        if dimensions is None:
                            summary["image_dimensions"] = dict(min_width=width, max_width=width, min_height=height, max_height=height)
                        else:
                            for axis, value in (("width", width), ("height", height)):
                                dimensions[f"min_{axis}"] = min(dimensions[f"min_{axis}"], value)
                                dimensions[f"max_{axis}"] = max(dimensions[f"max_{axis}"], value)
                        for box in boxes:
                            summary["class_counts"][box["class_id"]] += 1
                            confidence = box["confidence"]
                            label = "below_0.5" if confidence < .5 else "0.5_to_below_0.8" if confidence < .8 else "0.8_to_1"
                            summary["confidence_bins"][label] += 1
                    except BaseException as error:
                        row.update(status="error", error=f"{type(error).__name__}: {error}")
                        raise
                    finally:
                        manifest.write(json.dumps(row, ensure_ascii=False) + "\n")
                        manifest.flush()
            summary["complete"] = True
    except BaseException as error:
        summary["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["exact_duplicate_groups"] = [dict(sha256=digest, members=names) for digest, names in duplicates.items() if len(names) > 1]
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main(argv=None):
    """Export explicitly unreviewed predictions; failures return a nonzero process exit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", help="Detector device, e.g. cpu or 0; default chooses CUDA when available")
    args = parser.parse_args(argv)
    result = export_drafts(args.archive, args.weights, args.out, device=args.device)
    print(json.dumps(dict(complete=result["complete"], images=result["images_completed"], detections=result["detections"], output=str(args.out)), indent=2))


if __name__ == "__main__":
    main()
