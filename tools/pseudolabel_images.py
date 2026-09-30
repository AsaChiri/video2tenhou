# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Generate unreviewed face-localization box drafts from a ZIP of images.

Images retain their original bytes. Predictions, normalized YOLO drafts and a
provenance manifest are separate from accepted annotations. No training split or
data.yaml is created. A failed run retains partial output marked incomplete.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import stat
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict

import cv2
import numpy as np

from video2tenhou.files import sha256_file
from video2tenhou.perception import detector

if TYPE_CHECKING:
    from collections.abc import Sequence

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CLASS_NAMES = {0: "face"}
BOX_COORDINATE_COUNT = 4
LOW_CONFIDENCE_BOUNDARY = 0.5
HIGH_CONFIDENCE_BOUNDARY = 0.8


LOGGER = logging.getLogger("tools.pseudolabel_images")


class FaceDetection(Protocol):
    """Pixel coordinates and confidence emitted by a face detector."""

    xyxy: tuple[float, float, float, float]
    conf: float
    back: bool


class FaceModel(Protocol):
    """Class names required to interpret localization predictions."""

    names: dict[int, str]


class FacePredictor(Protocol):
    """Minimal detector contract used by the unreviewed draft exporter."""

    @property
    def model(self) -> FaceModel:
        """Expose the checkpoint's localization classes."""
        ...

    def predict(self, image: np.ndarray, /) -> Sequence[FaceDetection]:
        """Return face boxes in the input image's pixel coordinates."""
        ...


class DraftBox(TypedDict):
    """Validated coordinates and normalized training values for one draft."""

    raw_xyxy: list[float]
    xyxy: list[float]
    confidence: float
    class_id: int
    class_name: str
    yolo: list[float]
    back: bool


class DuplicateGroup(TypedDict):
    """Archive members sharing exactly the same original image bytes."""

    sha256: str
    members: list[str]


class ExportSummary(TypedDict):
    """Durable progress and provenance, including interrupted exports."""

    schema_version: int
    complete: bool
    reviewed: bool
    archive: str
    weights: str
    settings: dict[str, object]
    class_mode: str
    classes: dict[int, str] | None
    images_completed: int
    images_zero_detections: int
    detections: int
    confidence_bins: dict[str, int]
    class_counts: dict[int, int]
    image_dimensions: dict[str, int] | None
    versions: dict[str, str | None]
    error: str | None
    archive_sha256: NotRequired[str]
    weights_sha256: NotRequired[str]
    exporter_sha256: NotRequired[str]
    metadata_sha256: NotRequired[str | None]
    recognition_id: NotRequired[str | None]
    images_expected: NotRequired[int]
    elapsed_seconds: NotRequired[float]
    exact_duplicate_groups: NotRequired[list[DuplicateGroup]]


class ManifestRow(TypedDict):
    """Per-image provenance with paths present only after decoding succeeds."""

    source_archive: str
    member: str
    normalized_member: str
    reviewed: bool
    status: str
    sha256: NotRequired[str]
    width: NotRequired[int]
    height: NotRequired[int]
    image: NotRequired[str]
    predictions: NotRequired[str]
    draft_labels: NotRequired[str]
    detections: NotRequired[int]
    error: NotRequired[str]


POLICY = (
    "# Unreviewed detector drafts\n\nEvery box is a machine prediction, not ground "
    "truth. `draft_labels/` contains\nnormalized YOLO boxes using the class mapping"
    " in summary.json. The supported detector emits class 0=face. `predictions/` "
    "retains raw\ncoordinates, clipped coordinates and confidence. `manifest.jsonl`"
    " links these to\nthe exact archive member and image hash; `summary.json` "
    "records model provenance.\nImages preserve their original bytes. Coordinates "
    "use OpenCV's decoded image.\n\nZero detections do not establish an empty "
    "scene. Empty draft files are NOT\napproved negative labels. Inspect missed "
    "objects, box edges and localization classes\nbefore accepting images. "
    "Low-confidence predictions are retained as candidates.\nNo data.yaml or "
    "train/validation split is supplied. Split by source/session and\nduplicate "
    "group before training, and keep human-reviewed evaluation independent\nof "
    "teacher-generated labels. Exact duplicate groups do not detect near "
    "duplicates.\n\nUse only completed runs (`summary.json`: complete=true). On "
    "failure, keep the\npartial manifest for diagnosis and choose a new destination"
    " to retry. Do not\nmerge these files into human labels automatically. Retain "
    "image and teacher\nprovenance when deciding whether the eventual dataset/model"
    " can be distributed.\n"
)


def validate_classes(names: dict[int, str]) -> None:
    """Require face-only localization, leaving identities to the classifier."""
    if names != CLASS_NAMES:
        msg = (
            f"Expected detector classes {CLASS_NAMES!r}; checkpoint declares {names!r}"
        )
        raise ValueError(msg)


def image_members(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str]]:
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
        if (
            name.startswith("/")
            or any(part in ("", ".", "..") for part in parts)
            or any(":" in part or "\x00" in part for part in parts)
            or stat.S_ISLNK(member.external_attr >> 16)
        ):
            msg = f"Unsafe image member: {member.filename!r}"
            raise ValueError(msg)
        key = name.casefold()
        if key in seen:
            msg = f"Duplicate image member path: {member.filename!r}"
            raise ValueError(msg)
        seen.add(key)
        members.append((member, name))
    if not members:
        msg = "Archive contains no supported image files"
        raise ValueError(msg)
    return members


def normalized_box(detection: FaceDetection, width: int, height: int) -> DraftBox:
    """Validate a prediction and return raw/clipped pixels plus normalized YOLO values.

    Out-of-image edges are clipped. Nonfinite, reversed, empty or wholly outside
    boxes fail the run instead of silently becoming training annotations.
    """
    if width <= 0 or height <= 0:
        msg = "Image dimensions must be positive"
        raise ValueError(msg)
    raw = [float(value) for value in detection.xyxy]
    confidence = float(detection.conf)
    if len(raw) != BOX_COORDINATE_COUNT or not all(
        math.isfinite(value) for value in [*raw, confidence]
    ):
        msg = "Detection coordinates and confidence must be finite"
        raise ValueError(msg)
    if not 0 <= confidence <= 1 or raw[2] <= raw[0] or raw[3] <= raw[1]:
        msg = "Invalid detection area or confidence"
        raise ValueError(msg)
    x0, y0, x1, y1 = [
        max(0.0, min(value, limit))
        for value, limit in zip(raw, [width, height, width, height], strict=False)
    ]
    if x1 <= x0 or y1 <= y0:
        msg = "Detection has no area inside the image"
        raise ValueError(msg)
    if detection.back is not False:
        msg = "Face-only detector cannot export a back prediction"
        raise ValueError(msg)
    class_id, names = 0, CLASS_NAMES
    return {
        "raw_xyxy": raw,
        "xyxy": [x0, y0, x1, y1],
        "confidence": confidence,
        "class_id": class_id,
        "class_name": names[class_id],
        "yolo": [
            (x0 + x1) / (2 * width),
            (y0 + y1) / (2 * height),
            (x1 - x0) / width,
            (y1 - y0) / height,
        ],
        "back": False,
    }


@dataclass
class DraftWriter:
    """Write image artifacts and account for completed drafts within one export."""

    output: Path
    predictor: FacePredictor
    summary: ExportSummary
    duplicates: defaultdict[str, list[str]]
    emitted: set[str] = field(default_factory=set)

    def _read_image(
        self, archive: zipfile.ZipFile, member: zipfile.ZipInfo, row: ManifestRow
    ) -> tuple[bytes, np.ndarray]:
        """Decode a member and reserve its unique output paths."""
        source_name = row["normalized_member"]
        content = archive.read(member)
        digest = hashlib.sha256(content).hexdigest()
        row["sha256"] = digest
        source = PurePosixPath(source_name)
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", source.stem)[:60] or "image"
        member_id = hashlib.sha256(source_name.encode("utf-8")).hexdigest()[:16]
        name = f"{stem}_{member_id}_{digest[:16]}"
        if name.casefold() in self.emitted:
            msg = f"Output filename collision: {source_name!r}"
            raise ValueError(msg)
        self.emitted.add(name.casefold())
        image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            msg = f"Cannot decode image: {source_name}"
            raise ValueError(msg)
        height, width = image.shape[:2]
        row.update(
            width=width,
            height=height,
            image=f"images/{name}{source.suffix.lower()}",
            predictions=f"predictions/{name}.json",
            draft_labels=f"draft_labels/{name}.txt",
        )
        return content, image

    def write(
        self, archive: zipfile.ZipFile, member: zipfile.ZipInfo, row: ManifestRow
    ) -> None:
        """Publish one image and its unreviewed predictions before counting it."""
        content, image = self._read_image(archive, member, row)
        width, height = row["width"], row["height"]
        # Exclusive writes prevent accidental reuse even if a caller
        # changes output contents during this single-process export.
        with (self.output / row["image"]).open("xb") as stream:
            stream.write(content)
        boxes = [
            normalized_box(detection, width, height)
            for detection in self.predictor.predict(image)
        ]
        row.update(
            detections=len(boxes),
            status="unreviewed_predictions" if boxes else "unreviewed_zero_detections",
        )
        self._write_predictions(row, boxes)
        self._summarize(row, boxes)

    def _write_predictions(self, row: ManifestRow, boxes: list[DraftBox]) -> None:
        """Create JSON and YOLO drafts without replacing existing output."""
        source_name, digest = row["normalized_member"], row["sha256"]
        width, height = row["width"], row["height"]
        with (self.output / row["predictions"]).open("x", encoding="utf-8") as stream:
            json.dump(
                {
                    "member": source_name,
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "reviewed": False,
                    "status": row["status"],
                    "boxes": boxes,
                },
                stream,
                indent=2,
                allow_nan=False,
            )
        with (self.output / row["draft_labels"]).open("x", encoding="utf-8") as stream:
            for box in boxes:
                stream.write(
                    str(box["class_id"])
                    + " "
                    + " ".join(f"{value:.9f}" for value in box["yolo"])
                    + "\n"
                )

    def _summarize(self, row: ManifestRow, boxes: list[DraftBox]) -> None:
        """Account only for an image whose artifact writes completed."""
        source_name, digest = row["normalized_member"], row["sha256"]
        width, height = row["width"], row["height"]
        self.duplicates[digest].append(source_name)
        self.summary["images_completed"] += 1
        self.summary["images_zero_detections"] += int(not boxes)
        self.summary["detections"] += len(boxes)
        dimensions = self.summary["image_dimensions"]
        if dimensions is None:
            self.summary["image_dimensions"] = {
                "min_width": width,
                "max_width": width,
                "min_height": height,
                "max_height": height,
            }
        else:
            for axis, value in (("width", width), ("height", height)):
                dimensions[f"min_{axis}"] = min(dimensions[f"min_{axis}"], value)
                dimensions[f"max_{axis}"] = max(dimensions[f"max_{axis}"], value)
        for box in boxes:
            self.summary["class_counts"][box["class_id"]] += 1
            confidence = box["confidence"]
            label = (
                "below_0.5"
                if confidence < LOW_CONFIDENCE_BOUNDARY
                else "0.5_to_below_0.8"
                if confidence < HIGH_CONFIDENCE_BOUNDARY
                else "0.8_to_1"
            )
            self.summary["confidence_bins"][label] += 1


def _versions() -> dict[str, str | None]:
    versions = {}
    for package in (
        "video2tenhou",
        "libreyolo",
        "torch",
        "numpy",
        "opencv-python",
        "opencv-python-headless",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def export_drafts(
    archive_path: str | Path,
    weights: str | Path,
    output: str | Path,
    *,
    device: str | None = None,
    predictor: FacePredictor | None = None,
) -> ExportSummary:
    """Export drafts once into a new directory, leaving sources unchanged.

    Optional predictor injection supports offline tests; it must implement
    predict(BGR_image) and expose model.names. Production constructs the project's
    detector with the checkpoint metadata's inference defaults.
    Failure propagates after an incomplete
    summary and, for image failures, a manifest error row are saved.
    """
    archive_path, weights, output = (
        Path(value).resolve() for value in (archive_path, weights, output)
    )
    if not archive_path.is_file() or not weights.is_file():
        msg = "Archive and checkpoint must both be existing files"
        raise FileNotFoundError(msg)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("images", "predictions", "draft_labels"):
        (output / name).mkdir()
    (output / "README.md").write_text(POLICY, encoding="utf-8")
    started = time.perf_counter()
    summary: ExportSummary = {
        "schema_version": 1,
        "complete": False,
        "reviewed": False,
        "archive": str(archive_path),
        "weights": str(weights),
        "settings": {"device_requested": device},
        "class_mode": "face",
        "classes": None,
        "images_completed": 0,
        "images_zero_detections": 0,
        "detections": 0,
        "confidence_bins": {"below_0.5": 0, "0.5_to_below_0.8": 0, "0.8_to_1": 0},
        "class_counts": {},
        "image_dimensions": None,
        "versions": {},
        "error": None,
    }
    duplicates: defaultdict[str, list[str]] = defaultdict(list)
    try:
        summary["archive_sha256"] = sha256_file(archive_path)
        summary["weights_sha256"] = sha256_file(weights)
        summary["exporter_sha256"] = sha256_file(__file__)
        summary["versions"] = _versions()
        with zipfile.ZipFile(archive_path) as archive:
            members = image_members(archive)
            summary["images_expected"] = len(members)
            if predictor is None:
                predictor = detector.Detector(weights, device=device)
            validate_classes(predictor.model.names)
            summary["recognition_id"] = getattr(predictor, "id", None)
            for setting in ("imgsz", "conf", "iou", "cuda_graph"):
                summary["settings"][setting] = getattr(predictor, setting, None)
            metadata_path = weights.with_name("meta.json")
            summary["metadata_sha256"] = (
                sha256_file(metadata_path) if metadata_path.exists() else None
            )
            summary["classes"] = dict(predictor.model.names)
            summary["class_counts"] = dict.fromkeys(summary["classes"], 0)
            summary["settings"]["device_actual"] = str(
                getattr(predictor, "device", device)
            )
            writer = DraftWriter(output, predictor, summary, duplicates)
            with (output / "manifest.jsonl").open("x", encoding="utf-8") as manifest:
                for member, source_name in members:
                    row: ManifestRow = {
                        "source_archive": str(archive_path),
                        "member": member.filename,
                        "normalized_member": source_name,
                        "reviewed": False,
                        "status": "error",
                    }
                    try:
                        writer.write(archive, member, row)
                    except BaseException as error:
                        row.update(
                            status="error", error=f"{type(error).__name__}: {error}"
                        )
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
        summary["exact_duplicate_groups"] = [
            {"sha256": digest, "members": names}
            for digest, names in duplicates.items()
            if len(names) > 1
        ]
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
    result = export_drafts(args.archive, args.weights, args.out, device=args.device)
    LOGGER.info(
        "%s",
        json.dumps(
            {
                "complete": result["complete"],
                "images": result["images_completed"],
                "detections": result["detections"],
                "output": str(args.out),
            },
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
