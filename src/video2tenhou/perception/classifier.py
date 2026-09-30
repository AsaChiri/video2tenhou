"""Tile face classifier: resnet18 on upright 64x96 crops, calibrated posteriors over 39 classes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from ..files import sha256_file
from ..paths import MODEL_DIR
from ..train.data import CLASSES, to_crop
from ..train.train_classifier import make_model, to_tensor
from .device import select_device
from .runtime import runtime_signature

DEFAULT_DIR = MODEL_DIR / "classifier"
PREPROCESSING = "resnet18-bgr-upright64x96-to_tensor-v1"


class Classifier:
    """Calibrated local tile recognition; probabilities retain all 39 classes."""

    def __init__(self, model_dir: str | Path = DEFAULT_DIR, device: str | None = None):
        """Load the trained checkpoint and temperature; prefer CUDA when present."""
        model_dir = Path(model_dir)
        self.meta = json.load(open(model_dir / "meta.json"))
        self.classes: list[str] = self.meta["classes"]
        assert self.classes == CLASSES
        self.T = float(self.meta["temperature"])
        self.device = select_device(device)
        self.model = make_model(len(self.classes))
        self.model.load_state_dict(
            torch.load(model_dir / "weights.pt", map_location="cpu")
        )
        self.model.to(self.device).eval()
        weights_hash = sha256_file(model_dir / "weights.pt")
        self.id = hashlib.sha256(
            json.dumps(
                {
                    "weights": weights_hash,
                    "metadata": self.meta,
                    "preprocessing": PREPROCESSING,
                    "runtime": runtime_signature(self.device),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    @torch.no_grad()
    def posteriors(self, crops: list[np.ndarray]) -> np.ndarray:
        """(n, 39) calibrated posteriors for upright BGR crops of any size."""
        if not crops:
            return np.zeros((0, len(self.classes)), np.float32)
        out = []
        for i in range(0, len(crops), 512):
            x = torch.stack([to_tensor(to_crop(c)) for c in crops[i : i + 512]]).to(
                self.device
            )
            out.append(torch.softmax(self.model(x).float() / self.T, dim=1).cpu())
        return torch.cat(out).numpy()

    def classify(
        self, crops: list[np.ndarray], sideways: list[bool] | None = None
    ) -> np.ndarray:
        """Posteriors; a sideways crop is classified in both 90-degree turns and the turns are averaged."""
        if not sideways or not any(sideways):
            return self.posteriors(crops)
        batch, idx = [], []
        for i, (c, s) in enumerate(zip(crops, sideways, strict=False)):
            if s:
                batch += [
                    cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE),
                    cv2.rotate(c, cv2.ROTATE_90_COUNTERCLOCKWISE),
                ]
                idx += [i, i]
            else:
                batch.append(c)
                idx.append(i)
        p = self.posteriors(batch)
        out = np.zeros((len(crops), p.shape[1]), np.float32)
        cnt = np.zeros(len(crops), np.float32)
        for j, i in enumerate(idx):
            out[i] += p[j]
            cnt[i] += 1
        return out / cnt[:, None]

    def top(self, p: np.ndarray) -> tuple[str, float]:
        """Return the most likely tile notation and its calibrated probability."""
        i = int(np.argmax(p))
        return self.classes[i], float(p[i])
