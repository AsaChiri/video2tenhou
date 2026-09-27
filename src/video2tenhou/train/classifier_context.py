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
from collections import Counter
import json
from pathlib import Path
import random
import re
import shutil
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..layout import Calibration, fit_path, quad_to_box
from .data import CLASSES, HELD_OUT_HANDS, hand_of, hand_table, load_labels, region_upright, to_crop
from .face_data import file_hash
from .train_classifier import Crops, make_model, predict_logits, to_tensor


def _key(annotation):
    return f"{annotation['kind']}_{annotation['corner']}_{int(round(annotation['t']))}"


def _image_key(path):
    # Source manifests can have been written on Windows or POSIX.
    name = re.split(r"[\\/]", str(path))[-1]
    match = re.match(r"(hand|pond|meld)_(TL|TR|BL|BR)_\d+", name)
    if not match:
        raise ValueError(f"Cannot locate original human annotation for {path}")
    return match.group()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build_context_dataset(anchors: Path, human_manifest: Path, output: Path, *, video: Path,
                          work: Path, calib: str | Path = "pml") -> dict:
    """Build new context patches from cached lossless frames and reviewed quads.

    ``anchors`` is the reference classifier train/ directory; ``human_manifest``
    is a face-data manifest whose human TRAIN rows determine eligible contexts.
    Annotation time and the supplied workdir's hand table establish the split.
    Frames must already exist under work/<video>/frames/<time-to-3dp>.png.
    Anchor files are referenced with content hashes, never recropped or changed.
    The output records annotation, frame, geometry, video and source identities.
    """
    anchors, human_manifest, output, video, work = (Path(p).resolve() for p in (anchors, human_manifest, output, video, work))
    if output.exists():
        raise FileExistsError(output)
    cv2.setNumThreads(1)
    hands, labels = hand_table(video, work), load_labels(video)
    annotations = {}
    for annotation in labels:
        key = _key(annotation)
        if key in annotations:
            raise ValueError(f"Ambiguous rounded annotation key: {key}")
        annotations[key] = annotation
    index = {key: dict(t=d["t"], hand=hand_of(d["t"], hands)) for key, d in annotations.items()}
    eligible = set()
    for line in human_manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") != "train":
            continue
        if row.get("origin") != "human" or row.get("reviewed") is not True:
            raise ValueError("Context input must contain only reviewed human TRAIN rows")
        key = _image_key(row["source"])
        if key not in index or index[key]["hand"] in HELD_OUT_HANDS:
            raise ValueError(f"Human TRAIN row has missing or held-out annotation: {key}")
        eligible.add(key)
    cal = Calibration.load(calib, video)
    rows, counts, frame_hashes = [], Counter(), {}
    for folder in sorted(anchors.iterdir()):
        if not folder.is_dir() or folder.name not in CLASSES:
            continue
        for image in sorted(folder.glob("*.png")):
            key = _image_key(image)
            if key not in index or index[key]["hand"] in HELD_OUT_HANDS:
                raise ValueError(f"Anchor belongs to missing or held-out annotation: {image}")
            hand = index[key]["hand"]
            rows.append(dict(kind="anchor", image=str(image), sha256=file_hash(image), tile=folder.name,
                             group=f"{video.stem}:hand:{hand}", annotation=key, t=index[key]["t"], reviewed=True))
            counts["anchors"] += 1
    if not rows:
        raise ValueError("The original human anchor corpus is empty")
    output.mkdir(parents=True, exist_ok=False)
    (output / "contexts").mkdir()
    report = dict(complete=False, classes=CLASSES, held_out_hands=HELD_OUT_HANDS,
                  video=str(video), video_sha256=file_hash(video), annotations=index,
                  human_manifest_sha256=file_hash(human_manifest), anchors=str(anchors),
                  annotation_rows=labels, hands=hands, calibration=cal.data,
                  calibration_sha256=file_hash(fit_path(video)) if fit_path(video).is_file() else None,
                  policy="Existing human anchors plus reviewed training-hand contexts. No pseudo labels. Rounded filenames never determine the hand split.")
    _write_json(output / "provenance.json", report)
    for d in labels:  # Preserve the recorded recipe's annotation order.
        key = _key(d)
        if key not in eligible or not d["boxes"]:
            continue
        frame_path = work / video.stem / "frames" / f"{d['t']:.3f}.png"
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Missing cached lossless frame: {frame_path}")
        image, transform = region_upright(frame, cal, d["kind"], d["corner"])
        height, width = image.shape[:2]
        for i, box in enumerate(d["boxes"]):
            if box["tile"] not in CLASSES or box["tile"] in ("X", "none"):
                continue
            x0, y0, x1, y1 = quad_to_box(transform, box["quad"])
            x0, y0, x1, y1 = max(0., x0), max(0., y0), min(float(width), x1), min(float(height), y1)
            w, h = x1 - x0, y1 - y0
            if min(w, h) < 8:
                counts["too_small_excluded"] += 1
                continue
            a, b = int(max(0, x0 - .35 * w)), int(max(0, y0 - .35 * h))
            c, e = int(min(width, x1 + .35 * w)), int(min(height, y1 + .35 * h))
            path = output / "contexts" / f"{key}_{i:02d}.png"
            if not cv2.imwrite(str(path), image[b:e, a:c]):
                raise OSError(f"Could not save context patch: {path}")
            if str(frame_path) not in frame_hashes:
                frame_hashes[str(frame_path)] = file_hash(frame_path)
            rows.append(dict(kind="context", image=str(path), sha256=file_hash(path), tile=box["tile"],
                             sideways=bool(box.get("sideways", False)), box=[x0-a, y0-b, x1-a, y1-b],
                             group=f"{video.stem}:hand:{index[key]['hand']}", annotation=key, t=d["t"],
                             box_index=i, quad=box["quad"], frame_sha256=frame_hashes[str(frame_path)],
                             transform=transform.tolist(), reviewed=True))
            counts["contexts"] += 1
            counts[f"context_{d['kind']}"] += 1
    (output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report.update(complete=True, counts=dict(counts), frames=frame_hashes,
                  manifest_sha256=file_hash(output / "manifest.jsonl"))
    _write_json(output / "provenance.json", report)
    return report


def context_crop(image: np.ndarray, row: dict, rng=random) -> np.ndarray:
    """Apply the recorded 0–20% margin and ±6% center shift to a context patch.

    Four random draws occur in margin-X, margin-Y, shift-X, shift-Y order. A
    human sideways flag adds a random clockwise/counterclockwise quarter-turn.
    Anchors return the original array untouched and consume no random draws.
    """
    if row["kind"] == "anchor":
        return image
    x0, y0, x1, y1 = row["box"]
    w, h = x1 - x0, y1 - y0
    mx, my = rng.uniform(0, .20), rng.uniform(0, .20)
    dx, dy = rng.uniform(-.06, .06) * w, rng.uniform(-.06, .06) * h
    a, b = int(max(0, x0-mx*w+dx)), int(max(0, y0-my*h+dy))
    c, d = int(min(image.shape[1], x1+mx*w+dx)), int(min(image.shape[0], y1+my*h+dy))
    image = image[b:d, a:c]
    if min(image.shape[:2]) < 8:
        raise ValueError("Augmented context is too small")
    if row["sideways"]:
        image = cv2.rotate(image, rng.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]))
    return image


class ContextInputs(Dataset):
    """Hashed human training inputs returning normalized image, class, anchor mask."""
    def __init__(self, rows: list[dict]):
        """Verify immutable inputs once before sampling; preserve manifest order."""
        self.rows = rows
        for row in rows:
            if (row.get("reviewed") is not True or row.get("kind") not in ("anchor", "context") or
                    row.get("tile") not in CLASSES or row["kind"] == "context" and row["tile"] in ("X", "none")):
                raise ValueError("Expected reviewed anchors or known-face contexts")
            if file_hash(Path(row["image"])) != row["sha256"]:
                raise ValueError(f"Training image changed: {row['image']}")

    def __len__(self):
        """Number of anchor/context rows sampled per training epoch."""
        return len(self.rows)

    def __getitem__(self, index):
        """Read one image without augmenting anchors or relabeling any pixels."""
        row = self.rows[index]
        image = cv2.imread(row["image"])
        if image is None:
            raise ValueError(f"Unreadable training image: {row['image']}")
        image = context_crop(image, row)
        return to_tensor(to_crop(image)), CLASSES.index(row["tile"]), row["kind"] == "anchor"


def freeze_batchnorm_statistics(model: nn.Module) -> None:
    """Enter training mode except BN statistics; do not freeze affine parameters."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def refinement_loss(logits: torch.Tensor, labels: torch.Tensor, anchors: torch.Tensor,
                    images: torch.Tensor, teacher: nn.Module) -> torch.Tensor:
    """CE(smoothing=.05) plus 4*T² KL at T=2 on unchanged anchors only.

    The frozen teacher never receives augmented contexts or gradients. Empty
    anchor batches use CE alone, with the same reduction as the recorded recipe.
    """
    loss = nn.functional.cross_entropy(logits, labels, label_smoothing=.05)
    if anchors.any():
        with torch.no_grad():
            target = teacher(images[anchors])
        penalty = nn.functional.kl_div(nn.functional.log_softmax(logits[anchors] / 2., dim=1),
                                      nn.functional.softmax(target / 2., dim=1), reduction="batchmean")
        loss = loss + 4. * 4. * penalty
    return loss


def train_refinement(data: Path, validation: Path, base: Path, output: Path, *, device="cuda", log=print) -> dict:
    """Run the fixed three-epoch distilled recipe in a new directory; never promote it.

    ``base`` supplies both student initialization and the immutable teacher.
    Validation retains whole held-out hands 4/9/16/20. Checkpoints, unchanged
    temperature metadata and provenance are saved per epoch; performance and
    downstream reconstruction qualification remain separate acceptance gates.
    """
    data, validation, base, output = (Path(p).resolve() for p in (data, validation, base, output))
    if output.exists():
        raise FileExistsError(output)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    torch.set_num_threads(8); cv2.setNumThreads(1)
    rows = [json.loads(line) for line in (data / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    provenance = _json(data / "provenance.json")
    if not provenance.get("complete") or provenance["manifest_sha256"] != file_hash(data / "manifest.jsonl"):
        raise ValueError("Use a complete, unmodified context dataset")
    annotation_index = provenance["annotations"]
    for row in rows:
        identity = annotation_index[row["annotation"]]
        if identity["hand"] in HELD_OUT_HANDS or row["t"] != identity["t"]:
            raise ValueError("Training input violates the original annotation hand split")
    dataset = ContextInputs(rows)
    counts = Counter(row["tile"] for row in rows)
    weights = [1 / np.sqrt(counts[row["tile"]]) for row in rows]
    loader = DataLoader(dataset, batch_size=128, sampler=WeightedRandomSampler(weights, len(rows), replacement=True), num_workers=0)
    val = Crops(validation, augment=False)
    if not len(val):
        raise ValueError("Validation crops are empty")
    validation_files = {}
    for path, _, _ in val.items:
        key = _image_key(path)
        if key not in annotation_index or annotation_index[key]["hand"] not in HELD_OUT_HANDS:
            raise ValueError(f"Validation crop is not in a held-out hand: {path}")
        validation_files[str(path)] = file_hash(path)
    val_loader = DataLoader(val, batch_size=256, num_workers=0)
    meta = _json(base / "meta.json")
    if meta["classes"] != CLASSES or not np.isfinite(meta["temperature"]) or meta["temperature"] <= 0:
        raise ValueError("Base classifier metadata has incompatible classes or temperature")
    model = make_model(len(CLASSES))
    model.load_state_dict(torch.load(base / "weights.pt", map_location="cpu", weights_only=True))
    model.to(device).eval()
    teacher = make_model(len(CLASSES))
    teacher.load_state_dict(torch.load(base / "weights.pt", map_location="cpu", weights_only=True))
    teacher.to(device).eval(); teacher.requires_grad_(False)
    initial, labels, _ = predict_logits(model, val_loader, device)
    before = initial.argmax(1) == labels
    report = dict(complete=False, base_weights_sha256=file_hash(base / "weights.pt"),
                  base_metadata_sha256=file_hash(base / "meta.json"), data_manifest_sha256=file_hash(data / "manifest.jsonl"),
                  validation=validation_files, teacher="Immutable reference classifier; identical initialization to student",
                  configuration=dict(seed=0, epochs=3, lr=1e-5, optimizer="AdamW", weight_decay=1e-4,
                                     label_smoothing=.05, batch=128, workers=0, batchnorm="frozen_running_statistics",
                                     margin_fraction=[0, .20], center_shift_fraction=[-.06, .06],
                                     temperature=meta["temperature"], distillation_temperature=2., distillation_weight=4.,
                                     distillation_scope="unchanged human anchors only", device=str(device)),
                  baseline_validation_correct=int(before.sum()), validation_count=len(val), epochs=[])
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(__file__), output / "recipe.py")
    shutil.copy2(data / "provenance.json", output / "dataset-provenance.json")
    _write_json(output / "provenance.json", report)
    started = time.perf_counter()
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        for epoch in range(1, 4):
            freeze_batchnorm_statistics(model)
            total, n = 0., 0
            for images, labels, anchors in loader:
                images, labels, anchors = images.to(device), labels.to(device), anchors.to(device)
                logits = model(images)
                loss = refinement_loss(logits, labels, anchors, images, teacher)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += loss.item() * len(labels); n += len(labels)
            model.eval()
            logits, targets, views = predict_logits(model, val_loader, device)
            after = logits.argmax(1) == targets
            losses, gains = (before & ~after).nonzero().flatten().tolist(), (~before & after).nonzero().flatten().tolist()
            target = output / f"epoch_{epoch}"
            target.mkdir()
            torch.save(model.state_dict(), target / "weights.pt")
            new_meta = dict(meta)
            new_meta.update(val_acc=float(after.float().mean()), train_crops=len(rows), val_crops=len(val),
                            per_view={kind: dict(acc=sum(bool(after[i]) for i, k in enumerate(views) if k == kind) / views.count(kind),
                                                 n=views.count(kind)) for kind in set(views)})
            new_meta.pop("top_confusions", None)
            new_meta["refinement"] = dict(base_weights_sha256=report["base_weights_sha256"],
                                           dataset_manifest_sha256=report["data_manifest_sha256"], epoch=epoch,
                                           temperature_policy="Unchanged original temperature to isolate logit changes")
            _write_json(target / "meta.json", new_meta)
            row = dict(epoch=epoch, loss=total/n, validation_correct=int(after.sum()),
                       lost_original_correct=[str(val.items[i][0]) for i in losses], gained=[str(val.items[i][0]) for i in gains],
                       weights_sha256=file_hash(target / "weights.pt"))
            report["epochs"].append(row)
            _write_json(target / "provenance.json", {**report, "checkpoint_epoch": epoch, "checkpoint_complete": True})
            _write_json(output / "provenance.json", report)
            log(json.dumps(row))
        if (file_hash(base / "weights.pt") != report["base_weights_sha256"] or
                file_hash(base / "meta.json") != report["base_metadata_sha256"]):
            raise ValueError("Reference teacher checkpoint changed during refinement")
        report["complete"] = True
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        _write_json(output / "provenance.json", report)
    return report


def main(argv=None):
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
        report = build_context_dataset(args.anchors, args.human_manifest, args.out, video=args.video, work=args.work, calib=args.calib)
        print(json.dumps(report["counts"], indent=2))
    else:
        train_refinement(args.data, args.validation, args.base, args.out, device=args.device)


if __name__ == "__main__":
    main()
