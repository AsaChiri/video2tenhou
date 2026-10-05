# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Train the tile face classifier (resnet18 on 64x96 upright crops, 39 classes).

    uv run python -m video2tenhou.train.train_classifier --epochs 30

Writes models/classifier/weights.pt and models/classifier/meta.json (classes,
input size, temperature fitted on the held-out crops, per-view accuracy).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from video2tenhou.files import atomic_write_json
from video2tenhou.logging_setup import RESULT, command_logging
from video2tenhou.paths import DATA_DIR as ROOT
from video2tenhou.perception.classifier import make_model, to_tensor
from video2tenhou.perception.crops import CROP_H, CROP_W
from video2tenhou.perception.tiles import CLASS_INDEX, CLASSES

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

BLUR_PROBABILITY = 0.3
OCCLUSION_PROBABILITY = 0.3


TRAINING_SEED = 0
AUGMENTATION_RECIPE = "numpy-generator-pcg64-v1"


LOGGER = logging.getLogger("video2tenhou.train.train_classifier")


class Crops(Dataset):
    """Labeled crops with training-only augmentation and view metadata."""

    def __init__(self, root: Path, *, augment: bool, seed: int = TRAINING_SEED) -> None:
        """Index labeled crop files and choose whether to augment training items."""
        self.items = []
        for c in CLASSES:
            for p in sorted((root / c).glob("*.png")):
                self.items.append((p, CLASS_INDEX[c], p.stem.split("_")[0]))
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return the number of labeled crops in this dataset."""
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        """Read and optionally augment one crop, returning its label and view."""
        p, y, kind = self.items[index]
        img = cv2.imread(str(p))
        if img is None:
            raise OSError(f"Cannot read classifier training image: {p}")
        if self.augment:
            img = augment(img, self.rng)
        return to_tensor(img), y, kind


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply small training-only perturbations appropriate to physical tile faces."""
    h, w = img.shape[:2]
    # affine: rotation, scale, shear, translation
    ang = rng.uniform(-15, 15)
    sc = rng.uniform(0.85, 1.15)
    transform = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    transform[0, 1] += rng.uniform(-0.08, 0.08)
    transform[0, 2] += rng.uniform(-0.06, 0.06) * w
    transform[1, 2] += rng.uniform(-0.06, 0.06) * h
    img = cv2.warpAffine(
        img, transform, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    # colour
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.7, 1.3)
    hsv[..., 2] = hsv[..., 2] * rng.uniform(0.7, 1.3) + rng.uniform(-20, 20)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if rng.random() < BLUR_PROBABILITY:
        img = cv2.GaussianBlur(img, (3, 3), rng.uniform(0.3, 1.2))
    if rng.random() < OCCLUSION_PROBABILITY:  # partial occlusion
        ew, eh = int(w * rng.uniform(0.2, 0.5)), int(h * rng.uniform(0.2, 0.5))
        ex = int(rng.integers(0, w - ew + 1))
        ey = int(rng.integers(0, h - eh + 1))
        img[ey : ey + eh, ex : ex + ew] = rng.integers(0, 255, 3)
    return img


@torch.no_grad()
def predict_logits(
    model: torch.nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor, Sequence[str]]],
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collect evaluation logits and labels without changing model parameters."""
    model.eval()
    out, ys, kinds = [], [], []
    for x, y, k in loader:
        out.append(model(x.to(device)).float().cpu())
        ys.append(y)
        kinds += list(k)
    return torch.cat(out), torch.cat(ys), kinds


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Fit a scalar temperature on held-out logits to calibrate class probabilities."""
    best, best_nll = 1.0, float("inf")
    for temperature in np.linspace(0.5, 4.0, 71):
        nll = functional.cross_entropy(logits / temperature, y).item()
        if nll < best_nll:
            best, best_nll = float(temperature), nll
    return best


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse reproducible training settings and output locations."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="work/datasets/classifier")
    ap.add_argument("--out", default="models/classifier")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    return ap.parse_args(argv)


@command_logging
def main(argv: list[str] | None = None) -> None:
    """Train the classifier, calibrate it on validation crops, save weights and meta."""
    a = _arguments(argv)
    torch.manual_seed(TRAINING_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Crops(ROOT / a.data / "train", augment=True)
    va = Crops(ROOT / a.data / "val", augment=False)
    counts = Counter(y for _, y, _ in tr.items)
    weights = torch.tensor(
        [1.0 / np.sqrt(counts.get(i, 1)) for i in range(len(CLASSES))],
        dtype=torch.float32,
    )
    sample_w = [weights[y].item() for _, y, _ in tr.items]
    sampler = torch.utils.data.WeightedRandomSampler(
        sample_w, num_samples=len(tr), replacement=True
    )
    tl = DataLoader(
        tr, batch_size=a.batch, sampler=sampler, num_workers=0, drop_last=True
    )
    vl = DataLoader(va, batch_size=256, shuffle=False, num_workers=0)
    model = make_model(len(CLASSES), pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(tl)
    )
    best_acc, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for batch_x, batch_y, _ in tl:
            x, y = batch_x.to(device), batch_y.to(device)
            loss = functional.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        logits, ys, kinds = predict_logits(model, vl, device)
        acc = (logits.argmax(1) == ys).float().mean().item()
        LOGGER.info("epoch %2d loss %.4f val acc %.4f", ep + 1, tot / max(n, 1), acc)
        if acc > best_acc:
            best_acc, best_state = (
                acc,
                {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            )
    if best_state is None:
        raise RuntimeError(
            "Training produced no checkpoint with finite validation accuracy"
        )
    model.load_state_dict(best_state)
    logits, ys, kinds = predict_logits(model, vl, device)
    temperature = fit_temperature(logits, ys)
    per_kind = defaultdict(lambda: [0, 0])
    pred = logits.argmax(1)
    for p, y, k in zip(pred.tolist(), ys.tolist(), kinds, strict=False):
        per_kind[k][0] += p == y
        per_kind[k][1] += 1
    confusions = Counter(
        (CLASSES[y], CLASSES[p])
        for p, y in zip(pred.tolist(), ys.tolist(), strict=False)
        if p != y
    )
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "weights.pt")
    meta = {
        "classes": CLASSES,
        "crop": [CROP_W, CROP_H],
        "temperature": temperature,
        "training": {
            "seed": TRAINING_SEED,
            "augmentation": AUGMENTATION_RECIPE,
            "generator": type(tr.rng.bit_generator).__name__,
        },
        "val_acc": best_acc,
        "per_view": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in per_kind.items()},
        "train_crops": len(tr),
        "val_crops": len(va),
        "top_confusions": [
            [f"{a}->{b}", n] for (a, b), n in confusions.most_common(15)
        ],
    }
    atomic_write_json(out / "meta.json", meta, indent=1)
    RESULT.info("%s", json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
