# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""The direct YOLO9 path must reproduce LibreYOLO 1.5's public prediction exactly."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch
from libreyolo import LibreYOLO, LibreYOLO9
from libreyolo.models.base.model import BaseModel
from libreyolo.preprocess.yolo9 import preprocess_image
from libreyolo.utils.results import Results

from video2tenhou import video
from video2tenhou.calm import REGIONS
from video2tenhou.layout import Calibration
from video2tenhou.paths import DATA_DIR
from video2tenhou.perception import yolo9_direct
from video2tenhou.perception.crops import region_upright
from video2tenhou.perception.detector import DEFAULT_WEIGHTS, MODEL_STRIDE, Detector

REFERENCE_VIDEO = DATA_DIR / "samples" / "full_1080p.mp4"


def padded_shape(image: np.ndarray, size: int) -> tuple[int, int]:
    """Return the detector's stride-aligned rectangular input shape."""
    ratio = size / max(image.shape[:2])
    return (
        math.ceil(image.shape[0] * ratio / MODEL_STRIDE) * MODEL_STRIDE,
        math.ceil(image.shape[1] * ratio / MODEL_STRIDE) * MODEL_STRIDE,
    )


def table_crops(*sizes: tuple[int, int]) -> list[np.ndarray]:
    """Return felt-coloured noisy crops with bright tile-like rectangles."""
    rng = np.random.default_rng(len(sizes))
    crops = []
    for height, width in sizes:
        crop = np.empty((height, width, 3), np.uint8)
        crop[:] = (120, 110, 30)
        for x in range(4, width - 24, max(30, width // 5)):
            crop[4 : min(height - 2, 4 + height // 2), x : x + 22] = 235
        noise = rng.integers(-12, 13, crop.shape)
        crops.append(np.clip(crop + noise, 0, 255).astype(np.uint8))
    return crops


def assert_same_detections(
    model: LibreYOLO9, crops: list[np.ndarray], size: int, *, cuda_graph: bool
) -> None:
    """Compare the direct and public paths, requiring at least some boxes."""
    found = 0
    for crop in crops:
        shape = padded_shape(crop, size)
        public = model.predict(
            crop,
            imgsz=shape,
            conf=0.001,
            iou=0.5,
            color_format="bgr",
            max_det=300,
            cuda_graph=cuda_graph,
        )
        assert isinstance(public, Results)
        assert public.boxes is not None
        pixels = yolo9_direct.letterbox(crop, shape)
        original_size = (crop.shape[1], crop.shape[0])
        with model.cuda_graph_scope(cuda_graph):
            boxes, scores = yolo9_direct.detect(
                model, pixels, original_size, 0.001, 0.5
            )
        assert boxes == public.boxes.xyxy.tolist()
        assert scores == public.boxes.conf.tolist()
        found += len(boxes)
    assert found


def test_private_members_keep_the_signatures_this_path_calls() -> None:
    """Fail loudly when a LibreYOLO update moves the private members used."""
    assert list(inspect.signature(BaseModel._forward_graphed).parameters) == [
        "self",
        "input_tensor",
    ]
    assert list(inspect.signature(LibreYOLO9._postprocess).parameters) == [
        "self",
        "output",
        "conf_thres",
        "iou_thres",
        "original_size",
        "max_det",
        "ratio",
        "kwargs",
    ]
    assert list(inspect.signature(BaseModel.cuda_graph_scope).parameters) == [
        "self",
        "mode",
    ]
    assert LibreYOLO9.SUPPORTS_CUDA_GRAPH


@pytest.mark.parametrize(
    "size", [(675, 1080), (37, 101), (300, 300), (1500, 400), (5, 7)]
)
def test_model_input_matches_public_preprocessing(size: tuple[int, int]) -> None:
    """Values and strides equal LibreYOLO9's PIL-based preprocessing."""
    rng = np.random.default_rng(sum(size))
    bgr = rng.integers(0, 256, (*size, 3), dtype=np.uint8)
    shape = padded_shape(bgr, 1024)
    reference, _, original = preprocess_image(bgr, input_size=shape, color_format="bgr")
    direct = yolo9_direct.model_input(
        yolo9_direct.letterbox(bgr, shape), torch.device("cpu")
    )
    assert original == (size[1], size[0])
    assert torch.equal(direct, reference)
    assert direct.stride() == reference.stride()


def test_rejects_images_that_are_not_bgr_bytes() -> None:
    """The direct path accepts exactly the crops the readers produce."""
    for image in (np.zeros((8, 8), np.uint8), np.zeros((8, 8, 3), np.float32)):
        with pytest.raises(ValueError, match="uint8 BGR"):
            yolo9_direct.letterbox(image, (32, 32))


def test_detections_match_public_prediction_with_a_fresh_model() -> None:
    """Exercise forward and postprocessing on CPU without any trained weights."""
    torch.manual_seed(0)
    model = LibreYOLO9(None, size="t", nb_classes=1, device="cpu")
    model.model.eval()
    with torch.no_grad():  # He-normal convolutions give varied scores and boxes
        for module in model.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.weight.normal_(0, (2 / module.weight[0].numel()) ** 0.5)
                if module.bias is not None:
                    module.bias.zero_()
    crops = table_crops((60, 120), (90, 40), (64, 64))
    assert_same_detections(model, crops, 256, cuda_graph=False)


@pytest.mark.skipif(
    not (DEFAULT_WEIGHTS.exists() and REFERENCE_VIDEO.exists()),
    reason="local detector weights or the reference recording are not installed",
)
def test_installed_detector_matches_public_prediction_on_recorded_regions() -> None:
    """Every region shape of two reference frames, with the deployed settings."""
    current = Detector()
    cal = Calibration.load("pml")
    public: dict[tuple[int, int], LibreYOLO9] = {}
    found = 0
    for t in (700.0, 1505.0):
        frame = video.frame_at(REFERENCE_VIDEO, t)
        for region in REGIONS:
            kind, _, corner = region.partition(":")
            crop, _ = region_upright(frame, cal, kind, corner)
            shape = padded_shape(crop, current.imgsz)
            if shape not in public:  # each graph-captured shape keeps its own grids
                model = LibreYOLO(str(DEFAULT_WEIGHTS), device=current.device)
                assert isinstance(model, LibreYOLO9)
                public[shape] = model
            expected = public[shape].predict(
                crop,
                imgsz=shape,
                conf=current.conf,
                iou=current.iou,
                color_format="bgr",
                max_det=300,
                cuda_graph=current.cuda_graph,
            )
            assert isinstance(expected, Results)
            assert expected.boxes is not None
            detections = current.predict(crop)
            assert [list(d.xyxy) for d in detections] == expected.boxes.xyxy.tolist()
            assert [d.conf for d in detections] == expected.boxes.conf.tolist()
            found += len(detections)
    assert found
