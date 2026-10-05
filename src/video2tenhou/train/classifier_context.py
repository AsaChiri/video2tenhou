# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce human context refinement without changing classifier calibration.

The dataset keeps existing human anchor pixels and adds larger patches around
reviewed TRAIN boxes. Only those patches receive crop-boundary augmentation;
distillation is confined to unchanged anchors. Whole-hand validation uses the
original annotation timestamp, never its rounded filename. Existing supervised
``none`` anchors and outside-hand-window annotations are retained; new contexts
must name one of the 37 face identities. This module never infers tile labels.

Use ``python -m video2tenhou.train.classifier_context build --help`` or ``train
--help``. Destinations must be new; source checkpoints and datasets are immutable.
The fixed three-epoch recipe preserves the base inference temperature and freezes
batch-normalization running statistics, while its affine parameters still learn.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from video2tenhou.cache import source_identity
from video2tenhou.files import atomic_write_json, atomic_write_text
from video2tenhou.files import sha256_file as file_hash
from video2tenhou.layout import Calibration, fit_path
from video2tenhou.logging_setup import RESULT, command_logging
from video2tenhou.perception.classifier import make_model, to_tensor
from video2tenhou.perception.crops import region_upright, to_crop
from video2tenhou.perception.tiles import CLASSES

from .data import (
    HELD_OUT_HANDS,
    annotation_key,
    annotation_key_of,
    clipped_box,
    hand_of,
    hand_table,
    held_out,
    load_labels,
)
from .train_classifier import Crops, predict_logits

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

MIN_CONTEXT_SIDE = 8


LOGGER = logging.getLogger("video2tenhou.train.classifier_context")


def _json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected a metadata object: {path}")
    return data


def _annotation_index(labels: list[dict], hands: list[dict]) -> dict:
    """Bind rounded source keys to exact annotation times and hand splits."""
    annotations = {}
    for annotation in labels:
        key = annotation_key(annotation)
        if key in annotations:
            raise ValueError(f"Ambiguous rounded annotation key: {key}")
        annotations[key] = annotation
    return {
        key: {"t": d["t"], "hand": hand_of(d["t"], hands)}
        for key, d in annotations.items()
    }


def _eligible_contexts(human_manifest: Path, index: dict) -> set[str]:
    """Require reviewed human training rows with unambiguous annotations."""
    eligible = set()
    for line in human_manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") != "train":
            continue
        if row.get("origin") != "human" or row.get("reviewed") is not True:
            raise ValueError(
                "Context input must contain only reviewed human TRAIN rows"
            )
        key = annotation_key_of(row["source"])
        if key not in index or held_out(index[key]["hand"]):
            raise ValueError(
                f"Human TRAIN row has missing or held-out annotation: {key}"
            )
        eligible.add(key)
    return eligible


def _anchor_rows(anchors: Path, index: dict, video: Path) -> tuple[list[dict], Counter]:
    """Retain unchanged anchors and reject missing or held-out identities."""
    rows: list[dict] = []
    counts: Counter = Counter()
    for folder in sorted(anchors.iterdir()):
        if not folder.is_dir() or folder.name not in CLASSES:
            continue
        for image in sorted(folder.glob("*.png")):
            key = annotation_key_of(image)
            if key not in index or held_out(index[key]["hand"]):
                raise ValueError(
                    f"Anchor belongs to missing or held-out annotation: {image}"
                )
            hand = index[key]["hand"]
            rows.append(
                {
                    "kind": "anchor",
                    "image": str(image),
                    "sha256": file_hash(image),
                    "tile": folder.name,
                    "group": f"{video.stem}:hand:{hand}",
                    "annotation": key,
                    "t": index[key]["t"],
                    "reviewed": True,
                }
            )
            counts["anchors"] += 1
    if not rows:
        raise ValueError("The original human anchor corpus is empty")
    return rows, counts


def _context_patches(
    d: dict, image: np.ndarray, transform: np.ndarray, contexts: Path, counts: Counter
) -> list[dict]:
    """Save a padded patch around each known-face box, in the annotation's order."""
    height, width = image.shape[:2]
    rows = []
    for i, box in enumerate(d["boxes"]):
        if box["tile"] not in CLASSES or box["tile"] in ("X", "none"):
            continue
        x0, y0, x1, y1 = clipped_box(transform, box["quad"], width, height)
        w, h = x1 - x0, y1 - y0
        if min(w, h) < MIN_CONTEXT_SIDE:
            counts["too_small_excluded"] += 1
            continue
        a, b = int(max(0, x0 - 0.35 * w)), int(max(0, y0 - 0.35 * h))
        c, e = int(min(width, x1 + 0.35 * w)), int(min(height, y1 + 0.35 * h))
        path = contexts / f"{annotation_key(d)}_{i:02d}.png"
        if not cv2.imwrite(str(path), image[b:e, a:c]):
            raise OSError(f"Could not save context patch: {path}")
        rows.append(
            {
                "kind": "context",
                "image": str(path),
                "sha256": file_hash(path),
                "tile": box["tile"],
                "sideways": bool(box.get("sideways", False)),
                "box": [x0 - a, y0 - b, x1 - a, y1 - b],
                "annotation": annotation_key(d),
                "t": d["t"],
                "box_index": i,
                "quad": box["quad"],
                "transform": transform.tolist(),
                "reviewed": True,
            }
        )
        counts["contexts"] += 1
        counts[f"context_{d['kind']}"] += 1
    return rows


def build_context_dataset(
    anchors: Path,
    human_manifest: Path,
    output: Path,
    *,
    video: Path,
    work: Path,
    calib: str | Path = "pml",
) -> dict:
    """Build new context patches from cached lossless frames and reviewed quads.

    ``anchors`` is the reference classifier train/ directory; ``human_manifest``
    is a face-data manifest whose human TRAIN rows determine eligible contexts.
    Annotation time and the supplied workdir's hand table establish the split.
    Frames must already exist under work/<video>/frames/<time-to-3dp>.png.
    Anchor files are referenced with content hashes, never recropped or changed.
    The output records annotation, frame, geometry, video and source identities.
    """
    anchors, human_manifest, output, video, work = (
        Path(p).resolve() for p in (anchors, human_manifest, output, video, work)
    )
    if output.exists():
        raise FileExistsError(output)
    cv2.setNumThreads(1)
    hands, labels = hand_table(video, work), load_labels(video)
    index = _annotation_index(labels, hands)
    eligible = _eligible_contexts(human_manifest, index)
    cal = Calibration.load(calib, video)
    rows, counts = _anchor_rows(anchors, index, video)
    frame_hashes: dict[str, str] = {}
    output.mkdir(parents=True, exist_ok=False)
    (output / "contexts").mkdir()
    report = {
        "complete": False,
        "classes": CLASSES,
        "held_out_hands": HELD_OUT_HANDS,
        "video": str(video),
        "video_sha256": source_identity(video),
        "annotations": index,
        "human_manifest_sha256": file_hash(human_manifest),
        "anchors": str(anchors),
        "annotation_rows": labels,
        "hands": hands,
        "calibration": cal.data,
        "calibration_sha256": file_hash(fit_path(video))
        if fit_path(video).is_file()
        else None,
        "policy": (
            "Existing human anchors plus reviewed training-hand contexts. No pseudo"
            " labels. Rounded filenames never determine the hand split."
        ),
    }
    atomic_write_json(output / "provenance.json", report, indent=2)
    for d in labels:  # Preserve the recorded recipe's annotation order.
        key = annotation_key(d)
        if key not in eligible or not d["boxes"]:
            continue
        frame_path = work / video.stem / "frames" / f"{d['t']:.3f}.png"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Missing cached lossless frame: {frame_path}")
        image, transform = region_upright(frame, cal, d["kind"], d["corner"])
        patches = _context_patches(d, image, transform, output / "contexts", counts)
        if patches and str(frame_path) not in frame_hashes:
            frame_hashes[str(frame_path)] = file_hash(frame_path)
        rows += [
            row
            | {
                "group": f"{video.stem}:hand:{index[key]['hand']}",
                "frame_sha256": frame_hashes[str(frame_path)],
            }
            for row in patches
        ]
    atomic_write_text(
        output / "manifest.jsonl", "".join(json.dumps(row) + "\n" for row in rows)
    )
    report.update(
        complete=True,
        counts=dict(counts),
        frames=frame_hashes,
        manifest_sha256=file_hash(output / "manifest.jsonl"),
    )
    atomic_write_json(output / "provenance.json", report, indent=2)
    return report


class CropRandom(Protocol):
    """Random draws used by the recorded crop augmentation recipe."""

    def uniform(self, a: float, b: float) -> float:
        """Draw a real value from the specified interval."""
        ...

    def choice[Item](self, seq: Sequence[Item]) -> Item:
        """Choose one of the available rotations."""
        ...


def context_crop(image: np.ndarray, row: dict, rng: CropRandom = random) -> np.ndarray:
    """Apply the recorded 0-20% margin and ±6% center shift to a context patch.

    Four random draws occur in margin-X, margin-Y, shift-X, shift-Y order. A
    human sideways flag adds a random clockwise/counterclockwise quarter-turn.
    Anchors return the original array untouched and consume no random draws.
    """
    if row["kind"] == "anchor":
        return image
    x0, y0, x1, y1 = row["box"]
    w, h = x1 - x0, y1 - y0
    mx, my = rng.uniform(0, 0.20), rng.uniform(0, 0.20)
    dx, dy = rng.uniform(-0.06, 0.06) * w, rng.uniform(-0.06, 0.06) * h
    a, b = int(max(0, x0 - mx * w + dx)), int(max(0, y0 - my * h + dy))
    c, d = (
        int(min(image.shape[1], x1 + mx * w + dx)),
        int(min(image.shape[0], y1 + my * h + dy)),
    )
    image = image[b:d, a:c]
    if min(image.shape[:2]) < MIN_CONTEXT_SIDE:
        raise ValueError("Augmented context is too small")
    if row["sideways"]:
        image = cv2.rotate(
            image, rng.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE])
        )
    return image


class ContextInputs(Dataset):
    """Hashed human training inputs returning normalized image, class, anchor mask."""

    def __init__(self, rows: list[dict]) -> None:
        """Verify immutable inputs once before sampling; preserve manifest order."""
        self.rows = rows
        for row in rows:
            if (
                row.get("reviewed") is not True
                or row.get("kind") not in ("anchor", "context")
                or row.get("tile") not in CLASSES
                or (row["kind"] == "context" and row["tile"] in ("X", "none"))
            ):
                raise ValueError("Expected reviewed anchors or known-face contexts")
            if file_hash(Path(row["image"])) != row["sha256"]:
                raise ValueError(f"Training image changed: {row['image']}")

    def __len__(self) -> int:
        """Return the number of anchor/context rows sampled per training epoch."""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, bool]:
        """Read one image without augmenting anchors or relabeling any pixels."""
        row = self.rows[index]
        image = cv2.imread(row["image"])
        if image is None:
            raise ValueError(f"Unreadable training image: {row['image']}")
        image = context_crop(image, row)
        return (
            to_tensor(to_crop(image)),
            CLASSES.index(row["tile"]),
            row["kind"] == "anchor",
        )


def freeze_batchnorm_statistics(model: nn.Module) -> None:
    """Enter training mode except BN statistics; do not freeze affine parameters."""
    model.train()
    for module in model.modules():
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.SyncBatchNorm,
                nn.LazyBatchNorm1d,
                nn.LazyBatchNorm2d,
                nn.LazyBatchNorm3d,
            ),
        ):
            module.eval()


def refinement_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    anchors: torch.Tensor,
    images: torch.Tensor,
    teacher: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """CE(smoothing=.05) plus 4*T² KL at T=2 on unchanged anchors only.

    The frozen teacher never receives augmented contexts or gradients. Empty
    anchor batches use CE alone, with the same reduction as the recorded recipe.
    """
    loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.05)
    if anchors.any():
        with torch.no_grad():
            target = teacher(images[anchors])
        penalty = nn.functional.kl_div(
            nn.functional.log_softmax(logits[anchors] / 2.0, dim=1),
            nn.functional.softmax(target / 2.0, dim=1),
            reduction="batchmean",
        )
        loss = loss + 4.0 * 4.0 * penalty
    return loss


@dataclass
class RefinementInputs:
    """Verified training corpus, held-out loader and reference metadata."""

    rows: list[dict]
    loader: DataLoader
    val: Crops
    val_loader: DataLoader
    meta: dict
    validation_files: dict[str, str]


def _refinement_inputs(data: Path, validation: Path, base: Path) -> RefinementInputs:
    """Validate hashes and hand boundaries before any checkpoint is written."""
    rows = [
        json.loads(line)
        for line in (data / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    provenance = _json(data / "provenance.json")
    if not provenance.get("complete") or provenance["manifest_sha256"] != file_hash(
        data / "manifest.jsonl"
    ):
        raise ValueError("Use a complete, unmodified context dataset")
    annotation_index = provenance["annotations"]
    for row in rows:
        identity = annotation_index[row["annotation"]]
        if held_out(identity["hand"]) or row["t"] != identity["t"]:
            raise ValueError(
                "Training input violates the original annotation hand split"
            )
    dataset = ContextInputs(rows)
    counts = Counter(row["tile"] for row in rows)
    weights = [1 / np.sqrt(counts[row["tile"]]) for row in rows]
    loader = DataLoader(
        dataset,
        batch_size=128,
        sampler=WeightedRandomSampler(weights, len(rows), replacement=True),
        num_workers=0,
    )
    val = Crops(validation, augment=False)
    if not len(val):
        raise ValueError("Validation crops are empty")
    validation_files = {}
    for path, _, _ in val.items:
        key = annotation_key_of(path)
        if key not in annotation_index or not held_out(annotation_index[key]["hand"]):
            raise ValueError(f"Validation crop is not in a held-out hand: {path}")
        validation_files[str(path)] = file_hash(path)
    val_loader = DataLoader(val, batch_size=256, num_workers=0)
    meta = _json(base / "meta.json")
    if (
        meta["classes"] != CLASSES
        or not np.isfinite(meta["temperature"])
        or meta["temperature"] <= 0
    ):
        raise ValueError(
            "Base classifier metadata has incompatible classes or temperature"
        )
    return RefinementInputs(rows, loader, val, val_loader, meta, validation_files)


def _save_epoch(
    model: nn.Module,
    inputs: RefinementInputs,
    output: Path,
    report: dict,
    *,
    epoch: int,
    mean_loss: float,
    before: torch.Tensor,
    device: str,
) -> dict:
    """Publish an epoch's weights, unchanged temperature and paired validation."""
    model.eval()
    val = inputs.val
    logits, targets, views = predict_logits(model, inputs.val_loader, device)
    after = logits.argmax(1) == targets
    losses, gains = (
        (before & ~after).nonzero().flatten().tolist(),
        (~before & after).nonzero().flatten().tolist(),
    )
    target = output / f"epoch_{epoch}"
    target.mkdir()
    torch.save(model.state_dict(), target / "weights.pt")
    new_meta = dict(inputs.meta)
    new_meta.update(
        val_acc=float(after.float().mean()),
        train_crops=len(inputs.rows),
        val_crops=len(val),
        per_view={
            kind: {
                "acc": sum(bool(after[i]) for i, k in enumerate(views) if k == kind)
                / views.count(kind),
                "n": views.count(kind),
            }
            for kind in set(views)
        },
    )
    new_meta.pop("top_confusions", None)
    new_meta["refinement"] = {
        "base_weights_sha256": report["base_weights_sha256"],
        "dataset_manifest_sha256": report["data_manifest_sha256"],
        "epoch": epoch,
        "temperature_policy": "Unchanged original temperature to isolate logit changes",
    }
    atomic_write_json(target / "meta.json", new_meta, indent=2)
    row = {
        "epoch": epoch,
        "loss": mean_loss,
        "validation_correct": int(after.sum()),
        "lost_original_correct": [str(val.items[i][0]) for i in losses],
        "gained": [str(val.items[i][0]) for i in gains],
        "weights_sha256": file_hash(target / "weights.pt"),
    }
    report["epochs"].append(row)
    atomic_write_json(
        target / "provenance.json",
        {**report, "checkpoint_epoch": epoch, "checkpoint_complete": True},
        indent=2,
    )
    atomic_write_json(output / "provenance.json", report, indent=2)
    return row


def train_refinement(
    data: Path,
    validation: Path,
    base: Path,
    output: Path,
    *,
    device: str = "cuda",
) -> dict:
    """Run the fixed three-epoch distilled recipe in a new directory; never promote it.

    ``base`` supplies both student initialization and the immutable teacher.
    Validation retains whole held-out hands 4/9/16/20. Checkpoints, unchanged
    temperature metadata and provenance are saved per epoch; performance and
    downstream reconstruction qualification remain separate acceptance gates.
    """
    data, validation, base, output = (
        Path(p).resolve() for p in (data, validation, base, output)
    )
    if output.exists():
        raise FileExistsError(output)
    random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(8)
    cv2.setNumThreads(1)
    inputs = _refinement_inputs(data, validation, base)
    model = make_model(len(CLASSES))
    model.load_state_dict(
        torch.load(base / "weights.pt", map_location="cpu", weights_only=True)
    )
    model.to(device).eval()
    teacher = make_model(len(CLASSES))
    teacher.load_state_dict(
        torch.load(base / "weights.pt", map_location="cpu", weights_only=True)
    )
    teacher.to(device).eval()
    teacher.requires_grad_(requires_grad=False)
    initial, labels, _ = predict_logits(model, inputs.val_loader, device)
    before = initial.argmax(1) == labels
    report = {
        "complete": False,
        "base_weights_sha256": file_hash(base / "weights.pt"),
        "base_metadata_sha256": file_hash(base / "meta.json"),
        "data_manifest_sha256": file_hash(data / "manifest.jsonl"),
        "validation": inputs.validation_files,
        "teacher": (
            "Immutable reference classifier; identical initialization to student"
        ),
        "configuration": {
            "seed": 0,
            "epochs": 3,
            "lr": 1e-5,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "label_smoothing": 0.05,
            "batch": 128,
            "workers": 0,
            "batchnorm": "frozen_running_statistics",
            "margin_fraction": [0, 0.20],
            "center_shift_fraction": [-0.06, 0.06],
            "temperature": inputs.meta["temperature"],
            "distillation_temperature": 2.0,
            "distillation_weight": 4.0,
            "distillation_scope": "unchanged human anchors only",
            "device": str(device),
        },
        "baseline_validation_correct": int(before.sum()),
        "validation_count": len(inputs.val),
        "epochs": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output / "recipe.py")
    shutil.copy2(data / "provenance.json", output / "dataset-provenance.json")
    atomic_write_json(output / "provenance.json", report, indent=2)
    started = time.perf_counter()
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        for epoch in range(1, 4):
            freeze_batchnorm_statistics(model)
            total, n = 0.0, 0
            for batch_images, batch_labels, batch_anchors in inputs.loader:
                images, labels, anchors = (
                    batch_images.to(device),
                    batch_labels.to(device),
                    batch_anchors.to(device),
                )
                logits = model(images)
                loss = refinement_loss(logits, labels, anchors, images, teacher)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(labels)
                n += len(labels)
            row = _save_epoch(
                model,
                inputs,
                output,
                report,
                epoch=epoch,
                mean_loss=total / n,
                before=before,
                device=device,
            )
            LOGGER.info("%s", json.dumps(row))
        if (
            file_hash(base / "weights.pt") != report["base_weights_sha256"]
            or file_hash(base / "meta.json") != report["base_metadata_sha256"]
        ):
            raise ValueError("Reference teacher checkpoint changed during refinement")
        report["complete"] = True
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        atomic_write_json(output / "provenance.json", report, indent=2)
    return report


@command_logging
def main(argv: list[str] | None = None) -> None:
    """Build a contextual corpus or reproduce the fixed distilled refinement."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    for name in ("anchors", "human-manifest", "video", "work", "out"):
        build.add_argument("--" + name, required=True, type=Path)
    build.add_argument("--calib", default="pml")
    train = commands.add_parser("train")
    for name in ("data", "validation", "base", "out"):
        train.add_argument("--" + name, required=True, type=Path)
    train.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.command == "build":
        report = build_context_dataset(
            args.anchors,
            args.human_manifest,
            args.out,
            video=args.video,
            work=args.work,
            calib=args.calib,
        )
        RESULT.info("%s", json.dumps(report["counts"], indent=2))
    else:
        train_refinement(
            args.data, args.validation, args.base, args.out, device=args.device
        )


if __name__ == "__main__":
    main()
