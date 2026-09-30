# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Build a face-localization dataset without mixing pseudo-labels into validation.

Human PML images keep their original held-out hand split. Archive drafts are
training-only and explicitly unreviewed; unknown-class images are excluded.
Exact duplicate images share a split or fail, never silently leak into validation.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from video2tenhou.files import sha256_file as file_hash
from video2tenhou.logging_setup import RESULT, command_logging

from .data import HELD_OUT_HANDS, hand_of, hand_table, load_labels


@dataclass(frozen=True, kw_only=True)
class FaceSource:
    """Label origin and grouping retained in the dataset manifest."""

    split: str
    origin: str
    reviewed: bool
    group: str
    view: str
    source_row: str | dict | None = None


@dataclass
class FaceDataset:
    """Write dataset images and enforce duplicate and provenance invariants."""

    output: Path
    rows: list[dict] = field(default_factory=list)
    seen: dict[str, dict] = field(default_factory=dict)
    stats: Counter = field(default_factory=Counter)

    def save(self, image: Path, lines: list[str], source: FaceSource) -> None:
        """Copy one unique image and record its label provenance."""
        digest = file_hash(image)
        if digest in self.seen:
            previous = self.seen[digest]
            if previous["split"] != source.split:
                msg = f"Image duplicated across train/validation: {image}"
                raise ValueError(msg)
            self.stats[f"{source.origin}_exact_duplicates_skipped"] += 1
            return
        name = f"{source.origin}_{image.stem}_{digest[:12]}{image.suffix.lower()}"
        image_out = self.output / "images" / source.split / name
        shutil.copy2(image, image_out)
        label_out = self.output / "labels" / source.split / (Path(name).stem + ".txt")
        label_out.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        row = {
            "image": str(image_out.relative_to(self.output)),
            "labels": str(label_out.relative_to(self.output)),
            "sha256": digest,
            "labels_sha256": file_hash(label_out),
            "source": str(image),
            "source_label": source.source_row,
            "origin": source.origin,
            "reviewed": source.reviewed,
            "group": source.group,
            "view": source.view,
            "split": source.split,
            "boxes": len(lines),
        }
        self.rows.append(row)
        self.seen[digest] = row
        self.stats[f"{source.origin}_{source.split}_images"] += 1
        self.stats[f"{source.origin}_{source.split}_boxes"] += len(lines)

    def add_human(self, human: Path, video: Path, hands: list[dict]) -> None:
        """Preserve the reviewed hand split and reject nonface annotations."""
        source_annotations = {
            f"{row['kind']}_{row['corner']}_{round(row['t'])}": row
            for row in load_labels(video)
        }
        for split in ("val", "train"):
            for image in sorted((human / "images" / split).glob("*")):
                if not image.is_file():
                    continue
                match = re.fullmatch(
                    r"(pond|hand|meld)_(TL|TR|BL|BR)_(\d+)", image.stem
                )
                if not match:
                    msg = f"Cannot establish PML hand identity for {image}"
                    raise ValueError(msg)
                view, _, _timestamp = match.groups()
                annotation = source_annotations.get(image.stem)
                if annotation is None:
                    msg = f"Human source annotation is missing: {image}"
                    raise ValueError(msg)
                # Filenames round seconds and can cross a hand boundary. Use the
                # original human annotation timestamp, as the original builder did.
                hand = hand_of(float(annotation["t"]), hands)
                expected = "val" if hand in HELD_OUT_HANDS else "train"
                if split != expected:
                    msg = f"Human image split disagrees with held-out hand: {image}"
                    raise ValueError(msg)
                label = human / "labels" / split / (image.stem + ".txt")
                lines = [
                    line
                    for line in label.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if any(line.split()[0] != "0" for line in lines):
                    self.stats["human_nonface_images_excluded"] += 1
                    continue
                self.save(
                    image,
                    lines,
                    FaceSource(
                        split=split,
                        origin="human",
                        reviewed=True,
                        group=f"{video.stem}:hand:{hand}"
                        if hand is not None
                        else f"{video.stem}:outside_hand_windows",
                        view=view,
                        source_row=str(label),
                    ),
                )

    def add_drafts(self, drafts: Path | None, summary: dict | None) -> None:
        """Admit complete face predictions as training-only unreviewed data."""
        draft_lines = (
            (drafts / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            if drafts is not None
            else []
        )
        for line in draft_lines:
            if summary is None:
                msg = "Draft images require their completed provenance summary"
                raise ValueError(msg)
            row = json.loads(line)
            if row.get("status") not in (
                "unreviewed_predictions",
                "unreviewed_zero_detections",
            ):
                msg = "Draft manifest contains an incomplete/error image"
                raise ValueError(msg)
            image = (drafts / row["image"]).resolve()
            prediction = (drafts / row["predictions"]).resolve()
            if not image.is_relative_to(drafts) or not prediction.is_relative_to(
                drafts
            ):
                msg = "Draft path leaves its input directory"
                raise ValueError(msg)
            if file_hash(image) != row["sha256"]:
                msg = f"Draft image changed after prediction: {image}"
                raise ValueError(msg)
            boxes = json.loads(prediction.read_text(encoding="utf-8"))["boxes"]
            if not boxes or any(box["class_id"] != 0 for box in boxes):
                self.stats["pseudo_empty_or_unknown_images_excluded"] += 1
                continue
            lines = [
                "0 " + " ".join(f"{value:.9f}" for value in box["yolo"])
                for box in boxes
            ]
            self.save(
                image,
                lines,
                FaceSource(
                    split="train",
                    origin="pseudo",
                    reviewed=False,
                    group=f"archive:{summary['archive_sha256']}",
                    view="archive_photo",
                    source_row=row,
                ),
            )


def build_face_dataset(
    human: Path, drafts: Path | None, output: Path, *, video: Path, hands: list[dict]
) -> dict:
    """Copy the verified human split and optional unique drafts into a new dataset.

    This uses all retained teacher confidences; it does not claim they are human
    labels. Any image containing nonface annotations is excluded rather than
    silently teaching those omitted objects as background. Human empty labels are
    retained as reviewed negatives; empty pseudo-labels are never negatives.
    """
    human, output = (Path(path).resolve() for path in (human, output))
    drafts = Path(drafts).resolve() if drafts is not None else None
    summary = None
    if drafts is not None:
        summary = json.loads((drafts / "summary.json").read_text(encoding="utf-8"))
        if not summary.get("complete") or summary.get("class_mode") != "face":
            msg = "Use a completed face-localization draft export"
            raise ValueError(msg)
        names = {int(key): value for key, value in summary["classes"].items()}
        if names != {0: "face"}:
            msg = "Expected the face-localization class mapping {0: face}"
            raise ValueError(msg)
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (output / kind / split).mkdir(parents=True)
    dataset = FaceDataset(output)
    dataset.add_human(human, video, hands)
    dataset.add_drafts(drafts, summary)
    report = {
        "complete": True,
        "classes": {0: "face"},
        "stats": dict(dataset.stats),
        "held_out_hands": HELD_OUT_HANDS,
        "archive_teacher": summary,
        "human_source": str(human),
        "video": str(video),
        "grouping": (
            "All archive photos are training-only, including unknown sessions; "
            "human hand groups remain held out."
        ),
        "limitations": (
            "Pseudo labels are unreviewed. Exact hashes do not exclude visually "
            "similar train/validation images. No back class is trained."
        ),
    }
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in dataset.rows),
        encoding="utf-8",
    )
    (output / "data.yaml").write_text(
        (
            f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\nnc: "
            "1\nnames:\n  0: face\n"
        ),
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


@command_logging
def main(argv: list[str] | None = None) -> None:
    """Prepare an isolated face dataset while preserving all source labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", required=True, type=Path)
    parser.add_argument(
        "--drafts",
        type=Path,
        help="Optional unreviewed training archive; omit for human-only refinement",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--work", default=Path("work"), type=Path)
    args = parser.parse_args(argv)
    result = build_face_dataset(
        args.human,
        args.drafts,
        args.out,
        video=args.video,
        hands=hand_table(args.video, args.work),
    )
    RESULT.info("%s", json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
