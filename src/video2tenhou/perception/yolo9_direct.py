# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Direct LibreYOLO 1.5 YOLO9 inference for upright uint8 BGR crops.

LibreYOLO's public ``predict`` converts each crop through PIL and divides the
whole padded image in float32 on the CPU. This path pads the same uint8 pixels
with LibreYOLO9's resize and fill, uploads bytes and maps them through a table
built with the same float32 division, so the model receives identical values
with identical strides. Forward dispatch, including CUDA graph replay inside the
caller's ``cuda_graph_scope``, and postprocessing are the model's own private
methods: ``libreyolo`` is pinned to the 1.5 series and
``tests/perception/test_yolo9_direct.py`` fails if they change.
"""

from __future__ import annotations

from functools import cache
from typing import Protocol

import cv2
import numpy as np
import torch

PAD_VALUE = 114
MAX_DETECTIONS = 300


class YOLO9Model(Protocol):
    """The LibreYOLO 1.5 YOLO9 members this path calls."""

    device: torch.device

    def _forward_graphed(self, input_tensor: torch.Tensor) -> object:
        """Run the forward, replaying a captured graph inside a graph scope."""
        ...

    def _postprocess(self, output: object, *args: object, **kwargs: object) -> dict:
        """Decode, threshold and suppress boxes in original-image pixels."""
        ...


@cache
def _byte_values(device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.arange(256, dtype=np.float32) / 255.0).to(device)


def letterbox(bgr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Return RGB uint8 pixels resized and padded to ``shape`` as LibreYOLO9 does."""
    if bgr.dtype != np.uint8 or bgr.shape[2:] != (3,):
        raise ValueError("The detector expects uint8 BGR images")
    height, width = shape
    ratio = min(height / bgr.shape[0], width / bgr.shape[1])
    new_h = max(int(bgr.shape[0] * ratio), 1)
    new_w = max(int(bgr.shape[1] * ratio), 1)
    padded = np.full((height, width, 3), PAD_VALUE, np.uint8)
    padded[:new_h, :new_w] = cv2.resize(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )
    return padded


def model_input(padded: np.ndarray, device: torch.device) -> torch.Tensor:
    """Return the (1, 3, H, W) tensor LibreYOLO9's public preprocessing makes."""
    pixels = torch.from_numpy(padded).to(device)
    return _byte_values(device)[pixels.long()].permute(2, 0, 1).unsqueeze(0)


def detect(
    model: YOLO9Model,
    padded: np.ndarray,
    original_size: tuple[int, int],
    conf: float,
    iou: float,
) -> tuple[list[list[float]], list[float]]:
    """Return xyxy boxes in ``original_size`` (width, height) pixels and confidences.

    ``padded`` is the crop's ``letterbox``; its shape is the model input shape.
    """
    with torch.no_grad():
        output = model._forward_graphed(model_input(padded, model.device))  # noqa: SLF001  pinned LibreYOLO 1.5 graph dispatch
    found = model._postprocess(  # noqa: SLF001  pinned LibreYOLO 1.5 YOLO9 postprocessing
        output,
        conf,
        iou,
        original_size,
        max_det=MAX_DETECTIONS,
        ratio=1.0,
        classes=None,
        input_size=padded.shape[:2],
    )
    if found["num_detections"] == 0:
        return [], []
    boxes = torch.as_tensor(found["boxes"], dtype=torch.float32)
    scores = torch.as_tensor(found["scores"], dtype=torch.float32)
    return boxes.tolist(), scores.tolist()
