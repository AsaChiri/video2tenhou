# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Calibrated posteriors over 39 tile classes from upright 64x96 ResNet18 crops."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models

from video2tenhou.files import sha256_file
from video2tenhou.paths import MODEL_DIR

from .crops import to_crop
from .device import select_device
from .runtime import runtime_signature
from .tiles import CLASSES

DEFAULT_DIR = MODEL_DIR / "classifier"
PREPROCESSING = "resnet18-bgr-upright64x96-to_tensor-v1"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """Convert BGR pixels to an ImageNet-normalized float tensor in RGB CHW order."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - MEAN) / STD).transpose(2, 0, 1))


def make_model(n: int, *, pretrained: bool = False) -> nn.Module:
    """Construct ResNet18; only training requests ImageNet weights for initialization.

    Inference immediately loads the complete local checkpoint, so downloading
    and then discarding a pretrained state is unnecessary and breaks offline use.
    """
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, n)
    return m


@dataclass(frozen=True)
class ClassifierConfig:
    """Checkpoint metadata, temperature, device and evidence identity of a classifier.

    Resolving it hashes the checkpoint and selects the device but loads no weights.
    """

    model_dir: Path
    meta: dict
    classes: list[str]
    T: float
    device: str
    id: str


def classifier_config(
    model_dir: str | Path = DEFAULT_DIR, device: str | None = None
) -> ClassifierConfig:
    """Resolve metadata, device and the weights/preprocessing/runtime identity."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    if meta["classes"] != CLASSES:
        raise ValueError("Classifier classes must match the ordered tile vocabulary")
    temperature = float(meta["temperature"])
    selected = select_device(device)
    identity = {
        "weights": sha256_file(model_dir / "weights.pt"),
        "metadata": meta,
        "preprocessing": PREPROCESSING,
        "runtime": runtime_signature(selected),
    }
    return ClassifierConfig(
        model_dir=model_dir,
        meta=meta,
        classes=meta["classes"],
        T=temperature,
        device=selected,
        id=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
    )


# Row c is to_tensor's normalized RGB channel c for each byte value: a lookup
# reproduces its float32 arithmetic exactly.
PIXEL_VALUES = to_tensor(
    np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 3, axis=2)
)[:, 0]


class Classifier:
    """Calibrated local tile recognition; probabilities retain all 39 classes."""

    def __init__(
        self, model_dir: str | Path = DEFAULT_DIR, device: str | None = None
    ) -> None:
        """Load the checkpoint described by ``classifier_config``; prefer CUDA."""
        config = classifier_config(model_dir, device)
        self.meta, self.classes, self.T = config.meta, config.classes, config.T
        self.device, self.id = config.device, config.id
        self.model = make_model(len(self.classes))
        self.model.load_state_dict(
            torch.load(config.model_dir / "weights.pt", map_location="cpu")
        )
        self.model.to(self.device).eval()

    @torch.no_grad()
    def posteriors(self, crops: list[np.ndarray]) -> np.ndarray:
        """(n, 39) calibrated posteriors for upright BGR crops of any size.

        Resized bytes are uploaded and normalized on the device through
        ``to_tensor``'s own per-channel values, giving identical model inputs.
        """
        if not crops:
            return np.zeros((0, len(self.classes)), np.float32)
        values = PIXEL_VALUES.to(self.device)
        channels = torch.arange(3, device=self.device).view(1, 3, 1, 1)
        out = []
        for i in range(0, len(crops), 512):
            bgr = torch.from_numpy(np.stack([to_crop(c) for c in crops[i : i + 512]]))
            rgb = bgr.to(self.device).permute(0, 3, 1, 2).flip(1).long()
            x = values[channels, rgb].contiguous()  # to_tensor's stacked NCHW layout
            out.append(torch.softmax(self.model(x).float() / self.T, dim=1).cpu())
        return torch.cat(out).numpy()

    def classify(
        self, crops: list[np.ndarray], sideways: list[bool] | None = None
    ) -> np.ndarray:
        """Return posteriors; a sideways crop averages both of its 90-degree turns."""
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


class LazyClassifier:
    """A classifier's identity and calibration now, its weights on first use.

    Cache validation needs only ``id``, ``classes`` and ``T``. The loaded model must
    have the identity the caches were validated against.
    """

    def __init__(self, model_dir: str | Path = DEFAULT_DIR) -> None:
        """Resolve the checkpoint's configuration without loading weights."""
        config = classifier_config(model_dir)
        self.model_dir = config.model_dir
        self.id, self.classes, self.T = config.id, config.classes, config.T
        self._model: Classifier | None = None
        self._lock = Lock()

    def classify(
        self, crops: list[np.ndarray], sideways: list[bool] | None = None
    ) -> np.ndarray:
        """Classify crops as ``Classifier.classify``, loading weights when needed."""
        with self._lock:
            if self._model is None:
                model = Classifier(self.model_dir)
                if model.id != self.id:
                    raise RuntimeError(
                        "Classifier files changed after evidence was validated"
                    )
                self._model = model
        return self._model.classify(crops, sideways)
