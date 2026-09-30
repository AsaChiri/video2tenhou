# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared detector inference settings for loading, exporting and packaging."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, kw_only=True)
class InferenceOptions:
    """Inference overrides; unset values retain checkpoint metadata defaults."""

    imgsz: int | None = None
    confidence: float | None = None
    iou: float | None = None
    cuda_graph: bool | None = None


def inference_settings(value: dict, **overrides: float | bool | None) -> dict:
    """Resolve defaults and explicit overrides, then validate effective settings."""
    defaults = {"imgsz": 1024, "confidence": 0.15, "iou": 0.5, "cuda_graph": False}
    if not isinstance(value, dict) or set(value) - defaults.keys():
        msg = "Unsupported detector inference metadata"
        raise ValueError(msg)
    settings = {
        **defaults,
        **value,
        **{key: item for key, item in overrides.items() if item is not None},
    }
    if not isinstance(settings["cuda_graph"], bool):
        msg = "Detector CUDA graph setting must be boolean"
        raise TypeError(msg)
    size = settings["imgsz"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        msg = "Detector image size must be a positive integer"
        raise ValueError(msg)
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(item)
        or not 0 < item <= 1
        for item in (settings["confidence"], settings["iou"])
    ):
        msg = "Detector confidence and IoU must be finite numbers in (0, 1]"
        raise ValueError(msg)
    return settings


def checkpoint_metadata(weights: Path, weights_hash: str, backend: str | None) -> dict:
    """Load adjacent settings only after verifying checkpoint identity and backend."""
    model_meta = {}
    meta_path = weights.with_name("meta.json")
    if meta_path.exists():
        model_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(model_meta, dict) or model_meta.get("schema_version") != 1:
            msg = "Unsupported detector metadata schema"
            raise ValueError(msg)
        if model_meta.get("weights_sha256") != weights_hash:
            msg = "Detector metadata does not match checkpoint SHA-256"
            raise ValueError(msg)
        if model_meta.get("backend") != "libreyolo":
            msg = "Unsupported detector metadata backend"
            raise ValueError(msg)
        if backend is not None and backend != model_meta["backend"]:
            msg = "Explicit detector backend conflicts with checkpoint metadata"
            raise ValueError(msg)
    return model_meta
