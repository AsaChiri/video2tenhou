"""Build a face-localization dataset without mixing pseudo-labels into validation.

Human PML images keep their original held-out hand split. Archive drafts are
training-only and explicitly unreviewed; unknown-class images are excluded.
Exact duplicate images share a split or fail, never silently leak into validation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

from .data import HELD_OUT_HANDS, hand_of, hand_table, load_labels


def file_hash(path: Path) -> str:
    """Return a streaming SHA-256 for dataset and checkpoint provenance."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_face_dataset(human: Path, drafts: Path | None, output: Path, *, video: Path, hands: list[dict]) -> dict:
    """Copy the verified human split and optional unique drafts into a new dataset.

    This uses all retained teacher confidences; it does not claim they are human
    labels. Any image containing nonface annotations is excluded rather than
    silently teaching those omitted objects as background. Human empty labels are
    retained as reviewed negatives; empty pseudo-labels are never negatives.
    """
    human, output = (Path(path).resolve() for path in (human, output))
    drafts = Path(drafts).resolve() if drafts is not None else None
    summary, face_ids = None, set()
    if drafts is not None:
        summary = json.loads((drafts / "summary.json").read_text(encoding="utf-8"))
        if not summary.get("complete") or summary.get("class_mode") != "face":
            raise ValueError("Use a completed face-localization draft export")
        names = {int(key): value for key, value in summary["classes"].items()}
        if names != {0: "face"}:
            raise ValueError("Expected the face-localization class mapping {0: face}")
        face_ids = {0}
    output.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (output / kind / split).mkdir(parents=True)
    rows = []
    seen = {}
    stats = Counter()
    source_annotations = {f"{row['kind']}_{row['corner']}_{int(round(row['t']))}": row for row in load_labels(video)}

    def save(image, lines, *, split, origin, reviewed, group, view, source_row=None):
        digest = file_hash(image)
        if digest in seen:
            previous = seen[digest]
            if previous["split"] != split:
                raise ValueError(f"Image duplicated across train/validation: {image}")
            stats[f"{origin}_exact_duplicates_skipped"] += 1
            return
        name = f"{origin}_{image.stem}_{digest[:12]}{image.suffix.lower()}"
        image_out = output / "images" / split / name
        shutil.copy2(image, image_out)
        label_out = output / "labels" / split / (Path(name).stem + ".txt")
        label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        row = dict(image=str(image_out.relative_to(output)), labels=str(label_out.relative_to(output)),
                   sha256=digest, labels_sha256=file_hash(label_out), source=str(image), source_label=source_row, origin=origin,
                   reviewed=reviewed, group=group, view=view, split=split, boxes=len(lines))
        rows.append(row)
        seen[digest] = row
        stats[f"{origin}_{split}_images"] += 1
        stats[f"{origin}_{split}_boxes"] += len(lines)

    for split in ("val", "train"):
        for image in sorted((human / "images" / split).glob("*")):
            if not image.is_file():
                continue
            match = re.fullmatch(r"(pond|hand|meld)_(TL|TR|BL|BR)_(\d+)", image.stem)
            if not match:
                raise ValueError(f"Cannot establish PML hand identity for {image}")
            view, _, timestamp = match.groups()
            annotation = source_annotations.get(image.stem)
            if annotation is None:
                raise ValueError(f"Human source annotation is missing: {image}")
            # Filenames round seconds and can cross a hand boundary. Use the
            # original human annotation timestamp, as the original builder did.
            hand = hand_of(float(annotation["t"]), hands)
            expected = "val" if hand in HELD_OUT_HANDS else "train"
            if split != expected:
                raise ValueError(f"Human image split disagrees with held-out hand: {image}")
            label = human / "labels" / split / (image.stem + ".txt")
            lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if any(line.split()[0] != "0" for line in lines):
                stats["human_nonface_images_excluded"] += 1
                continue
            save(image, lines, split=split, origin="human", reviewed=True,
                 group=f"{video.stem}:hand:{hand}" if hand is not None else f"{video.stem}:outside_hand_windows",
                 view=view, source_row=str(label))

    draft_lines = (drafts / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if drafts is not None else []
    for line in draft_lines:
        row = json.loads(line)
        if row.get("status") not in ("unreviewed_predictions", "unreviewed_zero_detections"):
            raise ValueError("Draft manifest contains an incomplete/error image")
        image = (drafts / row["image"]).resolve()
        prediction = (drafts / row["predictions"]).resolve()
        if not image.is_relative_to(drafts) or not prediction.is_relative_to(drafts):
            raise ValueError("Draft path leaves its input directory")
        if file_hash(image) != row["sha256"]:
            raise ValueError(f"Draft image changed after prediction: {image}")
        boxes = json.loads(prediction.read_text(encoding="utf-8"))["boxes"]
        if not boxes or any(box["class_id"] not in face_ids for box in boxes):
            stats["pseudo_empty_or_unknown_images_excluded"] += 1
            continue
        lines = ["0 " + " ".join(f"{value:.9f}" for value in box["yolo"]) for box in boxes]
        save(image, lines, split="train", origin="pseudo", reviewed=False,
             group=f"archive:{summary['archive_sha256']}", view="archive_photo", source_row=row)
    report = dict(complete=True, classes={0: "face"}, stats=dict(stats), held_out_hands=HELD_OUT_HANDS,
                  archive_teacher=summary, human_source=str(human), video=str(video),
                  grouping="All archive photos are training-only, including unknown sessions; human hand groups remain held out.",
                  limitations="Pseudo labels are unreviewed. Exact hashes do not exclude visually similar train/validation images. No back class is trained.")
    (output / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "data.yaml").write_text(f"path: {output.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: face\n", encoding="utf-8")
    (output / "provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv=None):
    """Prepare an isolated face dataset while preserving all source labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", required=True, type=Path)
    parser.add_argument("--drafts", type=Path, help="Optional unreviewed training archive; omit for human-only refinement")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--work", default=Path("work"), type=Path)
    args = parser.parse_args(argv)
    result = build_face_dataset(args.human, args.drafts, args.out, video=args.video, hands=hand_table(args.video, args.work))
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
