# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Tile-face localization with LibreYOLO and a region-coordinate box contract.

The detector loads standard YOLO9 face-only weights described by hash-verified
local metadata and never converts checkpoint formats.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from importlib import metadata
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from video2tenhou.files import sha256_file
from video2tenhou.paths import MODEL_DIR

from . import yolo9_direct
from .detector_metadata import InferenceOptions, checkpoint_metadata, inference_settings
from .device import select_device
from .evidence_policy import EvidencePolicy, resolve_policy
from .runtime import runtime_signature

if TYPE_CHECKING:
    import numpy as np

MODEL_STRIDE = 32
# Part of the recognition identity. Per-shape replay gives bit-identical outputs to
# the policy this name was recorded under, so the name is kept to keep caches valid.
GRAPH_POLICY = "libreyolo-public-single-shape-v1"
DEFAULT_WEIGHTS = MODEL_DIR / "detector" / "weights.pt"
FACE_CLASSES = {0: "face"}


def _file_stamp(path: Path) -> tuple[int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ino


@dataclass
class Det:
    """One region-coordinate tile-face box and its confidence."""

    xyxy: tuple[float, float, float, float]
    conf: float


@dataclass(frozen=True)
class DetectorConfig:
    """Checkpoint, effective inference settings and evidence identity of a detector.

    Resolving it hashes the checkpoint and selects the device but loads no weights.
    """

    weights: Path
    metadata: dict
    imgsz: int
    conf: float
    iou: float
    device: str
    cuda_graph: bool
    evidence_policy: EvidencePolicy
    id: str


def detector_config(
    weights: str | Path = DEFAULT_WEIGHTS,
    *,
    settings: InferenceOptions | None = None,
    device: str | int | None = None,
) -> DetectorConfig:
    """Resolve checkpoint metadata, settings, device and the recognition identity.

    Metadata must identify these exact checkpoint bytes and LibreYOLO. Explicit
    inference settings override its defaults. Without metadata, defaults are
    1024 pixels, confidence 0.15 and IoU 0.5. ``id`` fingerprints checkpoint
    content, LibreYOLO version, preprocessing, settings, graph policy and the
    numerical runtime, so changing any of them invalidates recognition caches.
    The evidence policy is excluded from ``id`` so changing it can reuse raw
    sparse readings. CUDA graphs are off unless metadata or settings request
    them on a CUDA device.
    """
    weights = Path(weights)
    weights_hash = sha256_file(weights)
    model_meta = checkpoint_metadata(weights, weights_hash)
    inference = inference_settings(
        model_meta.get("inference", {}),
        **asdict(settings or InferenceOptions()),
    )
    selected = select_device(device)
    selected = f"cuda:{selected}" if str(selected).isdigit() else str(selected)
    cuda_graph = inference["cuda_graph"] and selected.startswith("cuda")
    identity = {
        "weights_sha256": weights_hash,
        "backend": "libreyolo",
        "version": metadata.version("libreyolo"),
        "imgsz": inference["imgsz"],
        "conf": inference["confidence"],
        "iou": inference["iou"],
        "color": "BGR",
        "classes": FACE_CLASSES,
        "cuda_graph": cuda_graph,
        "graph_policy": GRAPH_POLICY if cuda_graph else None,
        "runtime": runtime_signature(selected),
        "preprocessing": "libreyolo9-top-left-rect-stride32-v1",
    }
    return DetectorConfig(
        weights=weights,
        metadata=model_meta,
        imgsz=inference["imgsz"],
        conf=inference["confidence"],
        iou=inference["iou"],
        device=selected,
        cuda_graph=cuda_graph,
        evidence_policy=resolve_policy(model_meta.get("evidence_policy")),
        id="detector:"
        + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
    )


class Detector:
    """Local YOLO tile localisation with consistent single-image geometry."""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        *,
        settings: InferenceOptions | None = None,
        device: str | int | None = None,
    ) -> None:
        """Load and check the checkpoint described by ``detector_config``.

        LibreYOLO requires one face class and the standard yolo9 family; metadata
        classes and architecture must match the checkpoint. Inference uses
        ``yolo9_direct``, which gives the model the same input tensors as
        LibreYOLO's public ``predict``. YOLO9 rebuilds its decode grids when the
        padded input shape changes, which would invalidate a capture of another
        shape, so with CUDA graphs each padded shape gets its own model instance
        that only replays its own capture.
        """
        stamp = _file_stamp(Path(weights))
        config = detector_config(weights, settings=settings, device=device)
        from libreyolo import LibreYOLO  # noqa: PLC0415

        self._load_model = partial(LibreYOLO, str(config.weights), device=config.device)
        self.model = self._load_model()
        if self.model.names != FACE_CLASSES:
            raise ValueError("LibreYOLO detector must declare exactly one class: face")
        if self.model.family != "yolo9":
            raise ValueError(
                "LibreYOLO detector requires the standard yolo9 model family"
            )
        if config.metadata:
            classes = config.metadata.get("classes")
            if classes != {str(key): value for key, value in FACE_CLASSES.items()}:
                raise ValueError(
                    "Detector metadata classes differ from checkpoint classes"
                )
            if config.metadata.get("architecture") != f"yolo9-{self.model.size}":
                raise ValueError(
                    "LibreYOLO metadata architecture differs from checkpoint"
                )
        self.imgsz, self.conf, self.iou = config.imgsz, config.conf, config.iou
        self.device, self.cuda_graph = config.device, config.cuda_graph
        self.evidence_policy, self.id = config.evidence_policy, config.id
        self._prediction_lock = Lock()
        self._weights, self._weights_stamp = config.weights, stamp
        self._shape_models = {}

    def predict(self, img: np.ndarray) -> list[Det]:
        """Detect an upright BGR crop, serializing this model's mutable buffers.

        The lock includes CPU materialization: concurrent web previews must not
        overwrite graph/head storage while another request consumes its output.
        """
        with self._prediction_lock:
            return self._detect(img, self._letterbox(img))

    def _letterbox(self, img: np.ndarray) -> np.ndarray:
        """Pad to the stride-aligned rectangle of this crop's own proportions."""
        height, width = img.shape[:2]
        ratio = self.imgsz / max(height, width)
        shape = (
            math.ceil(height * ratio / MODEL_STRIDE) * MODEL_STRIDE,
            math.ceil(width * ratio / MODEL_STRIDE) * MODEL_STRIDE,
        )
        return yolo9_direct.letterbox(img, shape)

    def _add_shape_model(self, shape: tuple[int, int]) -> None:
        """Give ``shape`` its own model; the first shape uses the validated one."""
        if not self._shape_models:
            self._shape_models[shape] = self.model
            return
        if _file_stamp(self._weights) != self._weights_stamp:
            raise RuntimeError(
                f"Detector weights changed after loading: {self._weights}"
            )
        self._shape_models[shape] = self._load_model()

    def _detect(self, img: np.ndarray, padded: np.ndarray) -> list[Det]:
        shape = padded.shape[:2]
        if self.cuda_graph and shape not in self._shape_models:
            self._add_shape_model(shape)
        model = self._shape_models[shape] if self.cuda_graph else self.model
        with model.cuda_graph_scope(self.cuda_graph):
            boxes, scores = yolo9_direct.detect(
                model, padded, (img.shape[1], img.shape[0]), self.conf, self.iou
            )
        return [
            Det((x0, y0, x1, y1), score)
            for (x0, y0, x1, y1), score in zip(boxes, scores, strict=True)
        ]

    def predict_batch(self, imgs: list[np.ndarray]) -> list[list[Det]]:
        """Return detections in input order with single-image numerical behavior.

        Mixed image shapes would make YOLO switch from rectangular to square
        padding. Even matching shapes can use different CUDA kernels when
        batched, changing box coordinates and therefore classifier crop pixels.
        Classifier inference is batched separately by the region reader.
        A worker thread letterboxes the next crops while the GPU works.
        The instance lock spans the whole batch, including model loading and
        result materialization, so concurrent previews cannot interleave calls.
        """
        if not imgs:
            return []
        with (
            self._prediction_lock,
            ThreadPoolExecutor(1, thread_name_prefix="detector-input") as pool,
        ):
            padded = pool.map(self._letterbox, imgs)
            return [
                self._detect(img, pixels)
                for img, pixels in zip(imgs, padded, strict=True)
            ]


class LazyDetector:
    """A detector's identity and policy now, its weights on first prediction.

    Cache validation needs only ``id`` and ``evidence_policy``. The loaded model
    must have the identity the caches were validated against.
    """

    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS) -> None:
        """Resolve the checkpoint's configuration without loading weights."""
        config = detector_config(weights)
        self.weights = config.weights
        self.id, self.evidence_policy = config.id, config.evidence_policy
        self._model: Detector | None = None
        self._lock = Lock()

    def _loaded(self) -> Detector:
        with self._lock:
            if self._model is None:
                model = Detector(self.weights)
                if model.id != self.id:
                    raise RuntimeError(
                        "Detector files changed after evidence was validated"
                    )
                self._model = model
            return self._model

    def predict(self, img: np.ndarray) -> list[Det]:
        """Detect tiles in one upright BGR crop, loading weights when first needed."""
        return self._loaded().predict(img)

    def predict_batch(self, imgs: list[np.ndarray]) -> list[list[Det]]:
        """Detect tiles in several crops, loading weights when first needed."""
        return self._loaded().predict_batch(imgs) if imgs else []
