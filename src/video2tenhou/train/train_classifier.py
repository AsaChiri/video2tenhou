"""Train the tile face classifier (resnet18 on 64x96 upright crops, 39 classes).

    uv run python -m video2tenhou.train.train_classifier --epochs 30

Writes models/classifier/weights.pt and models/classifier/meta.json (classes,
input size, temperature fitted on the held-out crops, per-view accuracy).
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import CLASSES, CLASS_INDEX, CROP_H, CROP_W

from ..paths import DATA_DIR as ROOT
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR pixels to an ImageNet-normalized float tensor in RGB CHW order."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1))


class Crops(Dataset):
    """Labeled face crops with optional training-only augmentation and view-kind metadata."""
    def __init__(self, root: Path, augment: bool):
        self.items = []
        for c in CLASSES:
            for p in sorted((root / c).glob("*.png")):
                self.items.append((p, CLASS_INDEX[c], p.stem.split("_")[0]))
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y, kind = self.items[i]
        img = cv2.imread(str(p))
        if self.augment:
            img = augment(img)
        return to_tensor(img), y, kind


def augment(img: np.ndarray) -> np.ndarray:
    """Apply small image perturbations appropriate to physical tile faces for training only."""
    h, w = img.shape[:2]
    # affine: rotation, scale, shear, translation
    ang = random.uniform(-15, 15)
    sc = random.uniform(0.85, 1.15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    M[0, 1] += random.uniform(-0.08, 0.08)
    M[0, 2] += random.uniform(-0.06, 0.06) * w
    M[1, 2] += random.uniform(-0.06, 0.06) * h
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # colour
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= random.uniform(0.7, 1.3)
    hsv[..., 2] = hsv[..., 2] * random.uniform(0.7, 1.3) + random.uniform(-20, 20)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 1.2))
    if random.random() < 0.3:  # partial occlusion
        ew, eh = int(w * random.uniform(0.2, 0.5)), int(h * random.uniform(0.2, 0.5))
        ex, ey = random.randint(0, w - ew), random.randint(0, h - eh)
        img[ey: ey + eh, ex: ex + ew] = np.random.randint(0, 255, 3)
    return img


def make_model(n: int, *, pretrained: bool = False) -> nn.Module:
    """Construct ResNet18; only training requests ImageNet weights for initialization.

    Inference immediately loads the complete local checkpoint, so downloading
    and then discarding a pretrained state is unnecessary and breaks offline use.
    """
    from torchvision.models import resnet18, ResNet18_Weights
    m = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m


@torch.no_grad()
def predict_logits(model, loader, device):
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
    for T in np.linspace(0.5, 4.0, 71):
        nll = F.cross_entropy(logits / T, y).item()
        if nll < best_nll:
            best, best_nll = float(T), nll
    return best


def main(argv=None):
    """Train the classifier, calibrate confidence on validation crops, and save weights and metadata."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="work/datasets/classifier")
    ap.add_argument("--out", default="models/classifier")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args(argv)
    random.seed(0); torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = Crops(ROOT / a.data / "train", augment=True)
    va = Crops(ROOT / a.data / "val", augment=False)
    counts = Counter(y for _, y, _ in tr.items)
    weights = torch.tensor([1.0 / np.sqrt(counts.get(i, 1)) for i in range(len(CLASSES))], dtype=torch.float32)
    sample_w = [weights[y].item() for _, y, _ in tr.items]
    sampler = torch.utils.data.WeightedRandomSampler(sample_w, num_samples=len(tr), replacement=True)
    tl = DataLoader(tr, batch_size=a.batch, sampler=sampler, num_workers=0, drop_last=True)
    vl = DataLoader(va, batch_size=256, shuffle=False, num_workers=0)
    model = make_model(len(CLASSES), pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tl))
    best_acc, best_state = -1.0, None
    for ep in range(a.epochs):
        model.train()
        tot, n = 0.0, 0
        for x, y, _ in tl:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y, label_smoothing=0.05)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        logits, ys, kinds = predict_logits(model, vl, device)
        acc = (logits.argmax(1) == ys).float().mean().item()
        print(f"epoch {ep + 1:2d} loss {tot / max(n, 1):.4f} val acc {acc:.4f}")
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    logits, ys, kinds = predict_logits(model, vl, device)
    T = fit_temperature(logits, ys)
    per_kind = defaultdict(lambda: [0, 0])
    pred = logits.argmax(1)
    for p, y, k in zip(pred.tolist(), ys.tolist(), kinds):
        per_kind[k][0] += p == y
        per_kind[k][1] += 1
    confusions = Counter((CLASSES[y], CLASSES[p]) for p, y in zip(pred.tolist(), ys.tolist()) if p != y)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "weights.pt")
    meta = {"classes": CLASSES, "crop": [CROP_W, CROP_H], "temperature": T, "val_acc": best_acc,
            "per_view": {k: {"acc": v[0] / v[1], "n": v[1]} for k, v in per_kind.items()},
            "train_crops": len(tr), "val_crops": len(va),
            "top_confusions": [[f"{a}->{b}", n] for (a, b), n in confusions.most_common(15)]}
    json.dump(meta, open(out / "meta.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
