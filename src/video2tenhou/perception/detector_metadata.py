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
        raise ValueError("Unsupported detector inference metadata")
    settings = {
        **defaults,
        **value,
        **{key: item for key, item in overrides.items() if item is not None},
    }
    if not isinstance(settings["cuda_graph"], bool):
        raise TypeError("Detector CUDA graph setting must be boolean")
    size = settings["imgsz"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("Detector image size must be a positive integer")
    if any(
        not isinstance(item, (int, float))
        or isinstance(item, bool)
        or not math.isfinite(item)
        or not 0 < item <= 1
        for item in (settings["confidence"], settings["iou"])
    ):
        raise ValueError("Detector confidence and IoU must be finite numbers in (0, 1]")
    return settings


def checkpoint_metadata(weights: Path, weights_hash: str) -> dict:
    """Load adjacent settings after verifying they describe these LibreYOLO bytes."""
    model_meta = {}
    meta_path = weights.with_name("meta.json")
    if meta_path.exists():
        model_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(model_meta, dict) or model_meta.get("schema_version") != 1:
            raise ValueError("Unsupported detector metadata schema")
        if model_meta.get("weights_sha256") != weights_hash:
            raise ValueError("Detector metadata does not match checkpoint SHA-256")
        if model_meta.get("backend") != "libreyolo":
            raise ValueError("Unsupported detector metadata backend")
    return model_meta
