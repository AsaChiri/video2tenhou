"""Tile localization with an explicit backend and region-coordinate box contract.

Backend selection comes from hash-verified local metadata or an explicit argument
and never converts checkpoint formats. LibreYOLO requires standard YOLO9
face-only weights while preserving the region-coordinate interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
from threading import Lock

import numpy as np
from ..paths import MODEL_DIR
from .graphs import GRAPH_POLICY, enable_owned_graphs
from .runtime import runtime_signature
from .evidence_policy import resolve_policy

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
    def __init__(self, weights: str | Path = DEFAULT_WEIGHTS, *, imgsz: int | None = None,
                 conf: float | None = None, iou: float | None = None,
                 device: str | int | None = None, backend: str | None = None,
                 cuda_graph: bool | None = None):
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
        1.5.0 YOLO9. Unsupported runtimes use eager inference; default is false
        unless qualified checkpoint metadata requests it.
        """
        import torch
        weights = Path(weights)
        digest = hashlib.sha256()
        with weights.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        model_meta = {}
        meta_path = weights.with_name("meta.json")
        if meta_path.exists():
            model_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(model_meta, dict) or model_meta.get("schema_version") != 1:
                raise ValueError("Unsupported detector metadata schema")
            if model_meta.get("weights_sha256") != digest.hexdigest():
                raise ValueError("Detector metadata does not match checkpoint SHA-256")
            if model_meta.get("backend") != "libreyolo":
                raise ValueError("Unsupported detector metadata backend")
            if backend is not None and backend != model_meta["backend"]:
                raise ValueError("Explicit detector backend conflicts with checkpoint metadata")
        self.evidence_policy = resolve_policy(model_meta.get("evidence_policy"))
        backend = backend or model_meta.get("backend", "libreyolo")
        if backend != "libreyolo":
            raise ValueError(f"Unsupported detector backend: {backend}")
        inference = model_meta.get("inference", {})
        if not isinstance(inference, dict) or set(inference) - {"imgsz", "confidence", "iou", "cuda_graph"}:
            raise ValueError("Unsupported detector inference metadata")
        imgsz = imgsz if imgsz is not None else inference.get("imgsz", 1024)
        conf = conf if conf is not None else inference.get("confidence", .15)
        iou = iou if iou is not None else inference.get("iou", .5)
        cuda_graph = cuda_graph if cuda_graph is not None else inference.get("cuda_graph", False)
        if not isinstance(cuda_graph, bool) or cuda_graph and backend != "libreyolo":
            raise ValueError("Detector CUDA graphs require a boolean LibreYOLO setting")
        if not isinstance(imgsz, int) or isinstance(imgsz, bool) or imgsz <= 0:
            raise ValueError("Detector image size must be a positive integer")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or
               not math.isfinite(value) or not 0 < value <= 1 for value in (conf, iou)):
            raise ValueError("Detector confidence and IoU must be finite numbers in (0, 1]")
        device = device if device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.backend = backend
        from libreyolo import LibreYOLO
        device = f"cuda:{device}" if str(device).isdigit() else str(device)
        self.model = LibreYOLO(str(weights), device=device)
        if self.model.names != {0: "face"}:
            raise ValueError("LibreYOLO detector must declare exactly one class: face")
        if self.model.family != "yolo9":
            raise ValueError("LibreYOLO detector requires the standard yolo9 model family")
        if model_meta:
            classes = model_meta.get("classes")
            expected = {str(key): value for key, value in self.model.names.items()}
            if classes != expected:
                raise ValueError("Detector metadata classes differ from checkpoint classes")
            if backend == "libreyolo" and model_meta.get("architecture") != f"yolo9-{self.model.size}":
                raise ValueError("LibreYOLO metadata architecture differs from checkpoint")
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device
        self._prediction_lock = Lock()
        version = metadata.version(backend)
        self.cuda_graph = cuda_graph and enable_owned_graphs(self.model, version=version, device=device)
        identity = dict(weights_sha256=digest.hexdigest(), backend=backend, version=version,
                        imgsz=imgsz, conf=conf, iou=iou, color="BGR", classes=self.model.names,
                        cuda_graph=self.cuda_graph, graph_policy=GRAPH_POLICY if self.cuda_graph else None,
                        runtime=runtime_signature(device),
                        preprocessing="libreyolo9-top-left-rect-stride32-v1")
        self.id = "detector:" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

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
        shape = (math.ceil(height * ratio / 32) * 32, math.ceil(width * ratio / 32) * 32)
        r = self.model.predict(img, imgsz=shape, conf=self.conf, iou=self.iou,
                               color_format="bgr", max_det=300, cuda_graph=self.cuda_graph)
        if r.boxes is None:
            return []
        if any(int(value) != 0 for value in r.boxes.cls.tolist()):
            raise ValueError("Face-only checkpoint returned an unexpected class ID")
        return [Det(tuple(float(v) for v in xyxy), float(c), False)
                for xyxy, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]

    def predict_batch(self, imgs: list[np.ndarray]) -> list[list[Det]]:
        """Return detections in input order with single-image numerical behavior.

        Mixed image shapes would make YOLO switch from rectangular to square
        padding. Even matching shapes can use different CUDA kernels when
        batched, changing box coordinates and therefore classifier crop pixels.
        Classifier inference is batched separately by the region reader.
        """
        return [self.predict(img) for img in imgs]
