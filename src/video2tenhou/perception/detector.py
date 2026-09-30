"""Tile localization with an explicit backend and region-coordinate box contract.

Backend selection comes from hash-verified local metadata or an explicit argument
and never converts checkpoint formats. LibreYOLO requires standard YOLO9
face-only weights while preserving the region-coordinate interface.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from threading import Lock

import numpy as np

from ..files import sha256_file
from ..paths import MODEL_DIR
from .detector_flow import infer_prepared, input_shape, one_ahead, prepare_input
from .detector_metadata import inference_settings
from .device import select_device
from .evidence_policy import resolve_policy
from .graphs import GRAPH_POLICY, enable_owned_graphs
from .runtime import runtime_signature

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
        imgsz: int | None = None,
        conf: float | None = None,
        iou: float | None = None,
        device: str | int | None = None,
        backend: str | None = None,
        cuda_graph: bool | None = None,
    ):
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
        ``cuda_graph`` opts into bounded, grid-owning CUDA graphs for LibreYOLO
        1.5.x YOLO9. Invalid graph state fails explicitly; default is false
        unless qualified checkpoint metadata requests it.
        Supported graph models prepare uint8 BGR inputs directly from arrays;
        batches overlap one next CPU preparation with serial inference. This
        input path has its own cache identity. Eager inference uses the public
        prediction API and its existing identity.
        """
        weights = Path(weights)
        weights_hash = sha256_file(weights)
        model_meta = {}
        meta_path = weights.with_name("meta.json")
        if meta_path.exists():
            model_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (
                not isinstance(model_meta, dict)
                or model_meta.get("schema_version") != 1
            ):
                raise ValueError("Unsupported detector metadata schema")
            if model_meta.get("weights_sha256") != weights_hash:
                raise ValueError("Detector metadata does not match checkpoint SHA-256")
            if model_meta.get("backend") != "libreyolo":
                raise ValueError("Unsupported detector metadata backend")
            if backend is not None and backend != model_meta["backend"]:
                raise ValueError(
                    "Explicit detector backend conflicts with checkpoint metadata"
                )
        self.evidence_policy = resolve_policy(model_meta.get("evidence_policy"))
        backend = backend or model_meta.get("backend", "libreyolo")
        if backend != "libreyolo":
            raise ValueError(f"Unsupported detector backend: {backend}")
        inference = inference_settings(
            model_meta.get("inference", {}),
            imgsz=imgsz,
            confidence=conf,
            iou=iou,
            cuda_graph=cuda_graph,
        )
        imgsz, conf, iou, cuda_graph = (
            inference[key] for key in ("imgsz", "confidence", "iou", "cuda_graph")
        )
        device = select_device(device)
        self.backend = backend
        from libreyolo import LibreYOLO

        device = f"cuda:{device}" if str(device).isdigit() else str(device)
        self.model = LibreYOLO(str(weights), device=device)
        if self.model.names != {0: "face"}:
            raise ValueError("LibreYOLO detector must declare exactly one class: face")
        if self.model.family != "yolo9":
            raise ValueError(
                "LibreYOLO detector requires the standard yolo9 model family"
            )
        if model_meta:
            classes = model_meta.get("classes")
            expected = {str(key): value for key, value in self.model.names.items()}
            if classes != expected:
                raise ValueError(
                    "Detector metadata classes differ from checkpoint classes"
                )
            if model_meta.get("architecture") != f"yolo9-{self.model.size}":
                raise ValueError(
                    "LibreYOLO metadata architecture differs from checkpoint"
                )
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device
        self._prediction_lock = Lock()
        version = metadata.version(backend)
        self.cuda_graph = cuda_graph and enable_owned_graphs(self.model, device=device)
        self._direct_preprocessing = self.cuda_graph
        identity = dict(
            weights_sha256=weights_hash,
            backend=backend,
            version=version,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            color="BGR",
            classes=self.model.names,
            cuda_graph=self.cuda_graph,
            graph_policy=GRAPH_POLICY if self.cuda_graph else None,
            runtime=runtime_signature(device),
            preprocessing="libreyolo9-top-left-rect-stride32-v1",
        )
        if self._direct_preprocessing:
            identity["input_path"] = "ndarray-one-ahead-v1"
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
        if self._direct_preprocessing and self._supported_input(img):
            return self._consume_prepared(prepare_input(img, self.imgsz))
        return self._public_predict(img)

    @staticmethod
    def _supported_input(img):
        return (
            isinstance(img, np.ndarray)
            and img.dtype == np.uint8
            and img.ndim == 3
            and img.shape[2] == 3
            and min(img.shape[:2]) > 4
        )

    def _consume_prepared(self, prepared):
        data = infer_prepared(
            self.model, prepared, self.conf, self.iou, self.cuda_graph
        )
        if any(int(value) != 0 for value in data["classes"]):
            raise ValueError("Face-only checkpoint returned an unexpected class ID")
        return [
            Det(tuple(float(v) for v in box), float(conf), False)
            for box, conf in zip(data["boxes"], data["scores"], strict=False)
        ]

    def _public_predict(self, img: np.ndarray) -> list[Det]:
        shape = input_shape(img, self.imgsz)
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
            raise ValueError("Face-only checkpoint returned an unexpected class ID")
        return [
            Det(tuple(float(v) for v in xyxy), float(c), False)
            for xyxy, c in zip(
                r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), strict=False
            )
        ]

    def predict_batch(self, imgs: list[np.ndarray]) -> list[list[Det]]:
        """Return detections in input order with single-image numerical behavior.

        Mixed image shapes would make YOLO switch from rectangular to square
        padding. Even matching shapes can use different CUDA kernels when
        batched, changing box coordinates and therefore classifier crop pixels.
        Classifier inference is batched separately by the region reader.
        On supported graph models, one CPU input is prepared ahead of the
        current inference. The instance lock spans the whole batch and worker
        cleanup, including when preparation or inference raises an exception.
        """
        if not imgs:
            return []
        with self._prediction_lock:
            if not self._direct_preprocessing or not all(
                self._supported_input(img) for img in imgs
            ):
                return [self._public_predict(img) for img in imgs]
            # CPU-only preparation owns one next item; all GPU work and output
            # materialization remain serial under this instance's lock.
            return one_ahead(
                imgs, lambda img: prepare_input(img, self.imgsz), self._consume_prepared
            )
