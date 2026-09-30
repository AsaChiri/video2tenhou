# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Tile localization with an explicit backend and region-coordinate box contract.

Backend selection comes from hash-verified local metadata or an explicit argument
and never converts checkpoint formats. LibreYOLO requires standard YOLO9
face-only weights while preserving the region-coordinate interface.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from video2tenhou.files import sha256_file
from video2tenhou.paths import MODEL_DIR

from .detector_metadata import InferenceOptions, checkpoint_metadata, inference_settings
from .device import select_device
from .evidence_policy import resolve_policy
from .runtime import runtime_signature

if TYPE_CHECKING:
    import numpy as np

MODEL_STRIDE = 32
GRAPH_POLICY = "libreyolo-public-single-shape-v1"
DEFAULT_WEIGHTS = MODEL_DIR / "detector" / "weights.pt"


@dataclass
class Det:
    """One region-coordinate box; ``back`` distinguishes face-down tiles."""

    xyxy: tuple[float, float, float, float]
    conf: float
    back: bool


class Detector:
    """Local YOLO tile localisation with consistent single-image geometry."""

    backend = "libreyolo"

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        *,
        settings: InferenceOptions | None = None,
        device: str | int | None = None,
        backend: str | None = None,
    ) -> None:
        """Load local weights and optional adjacent ``meta.json`` inference defaults.

        Metadata must identify these exact checkpoint bytes. Explicit inference
        settings override its defaults; a conflicting backend is an error rather
        than a format conversion. Without metadata, defaults are LibreYOLO,
        1024 pixels, confidence 0.15 and IoU 0.5. LibreYOLO requires one face class.
        ``id`` fingerprints checkpoint content, backend version and inference
        preprocessing/settings, so changing any of them invalidates recognition
        caches. LibreYOLO outputs always have ``back=False``; no back labels are
        inferred from unknown tile classes.
        ``evidence_policy`` controls later retention and has its own stage identities;
        it is excluded from ``id`` so changing it can reuse raw sparse readings.
        All inference uses LibreYOLO's public ``predict`` API. ``cuda_graph``
        enables capture on CUDA; default is false unless checkpoint metadata
        requests it. Public ``release_graphs`` drops captures before the padded
        input shape changes, keeping their decode-grid storage valid. Repeated
        shapes can replay; a shape change requires a new capture. This graph
        policy has its own recognition identity.
        """
        weights = Path(weights)
        weights_hash = sha256_file(weights)
        model_meta = checkpoint_metadata(weights, weights_hash, backend)
        self.evidence_policy = resolve_policy(model_meta.get("evidence_policy"))
        backend = backend or model_meta.get("backend", "libreyolo")
        if backend != "libreyolo":
            msg = f"Unsupported detector backend: {backend}"
            raise ValueError(msg)
        inference = inference_settings(
            model_meta.get("inference", {}),
            **asdict(settings or InferenceOptions()),
        )
        imgsz, conf, iou, cuda_graph = (
            inference[key] for key in ("imgsz", "confidence", "iou", "cuda_graph")
        )
        device = select_device(device)
        self.backend = backend
        from libreyolo import LibreYOLO  # noqa: PLC0415

        device = f"cuda:{device}" if str(device).isdigit() else str(device)
        self.model = LibreYOLO(str(weights), device=device)
        if self.model.names != {0: "face"}:
            msg = "LibreYOLO detector must declare exactly one class: face"
            raise ValueError(msg)
        if self.model.family != "yolo9":
            msg = "LibreYOLO detector requires the standard yolo9 model family"
            raise ValueError(msg)
        if model_meta:
            classes = model_meta.get("classes")
            expected = {str(key): value for key, value in self.model.names.items()}
            if classes != expected:
                msg = "Detector metadata classes differ from checkpoint classes"
                raise ValueError(msg)
            if model_meta.get("architecture") != f"yolo9-{self.model.size}":
                msg = "LibreYOLO metadata architecture differs from checkpoint"
                raise ValueError(msg)
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device
        self._prediction_lock = Lock()
        version = metadata.version(backend)
        self.cuda_graph = cuda_graph and device.startswith("cuda")
        self._graph_shape: tuple[int, int] | None = None
        identity = {
            "weights_sha256": weights_hash,
            "backend": backend,
            "version": version,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "color": "BGR",
            "classes": self.model.names,
            "cuda_graph": self.cuda_graph,
            "graph_policy": GRAPH_POLICY if self.cuda_graph else None,
            "runtime": runtime_signature(device),
            "preprocessing": "libreyolo9-top-left-rect-stride32-v1",
        }
        self.id = (
            "detector:"
            + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        )

    def predict(self, img: np.ndarray) -> list[Det]:
        """Detect an upright BGR crop, serializing this model's mutable buffers.

        The lock includes CPU materialization: concurrent web previews must not
        overwrite graph/head storage while another request consumes its output.
        """
        with self._prediction_lock:
            return self._predict(img)

    def _predict(self, img: np.ndarray) -> list[Det]:
        height, width = img.shape[:2]
        ratio = self.imgsz / max(height, width)
        shape = (
            math.ceil(height * ratio / MODEL_STRIDE) * MODEL_STRIDE,
            math.ceil(width * ratio / MODEL_STRIDE) * MODEL_STRIDE,
        )
        if self.cuda_graph:
            # YOLO9 replaces its decode grids on a shape change. Release captures
            # before that happens, including after a previous prediction failed.
            if self._graph_shape is not None and self._graph_shape != shape:
                self.model.release_graphs()
            self._graph_shape = shape
        r = self.model.predict(
            img,
            imgsz=shape,
            conf=self.conf,
            iou=self.iou,
            color_format="bgr",
            max_det=300,
            cuda_graph=self.cuda_graph,
        )
        if r.boxes is None:
            return []
        if any(int(value) != 0 for value in r.boxes.cls.tolist()):
            msg = "Face-only checkpoint returned an unexpected class ID"
            raise ValueError(msg)
        return [
            Det((float(x0), float(y0), float(x1), float(y1)), float(c), back=False)
            for (x0, y0, x1, y1), c in zip(
                r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), strict=False
            )
        ]

    def predict_batch(self, imgs: list[np.ndarray]) -> list[list[Det]]:
        """Return detections in input order with single-image numerical behavior.

        Mixed image shapes would make YOLO switch from rectangular to square
        padding. Even matching shapes can use different CUDA kernels when
        batched, changing box coordinates and therefore classifier crop pixels.
        Classifier inference is batched separately by the region reader.
        The instance lock spans the whole batch, including graph release and
        result materialization, so concurrent previews cannot interleave calls.
        """
        if not imgs:
            return []
        with self._prediction_lock:
            return [self._predict(img) for img in imgs]
