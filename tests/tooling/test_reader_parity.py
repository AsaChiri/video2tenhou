# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""The reader parity check flags any reading that batching changes."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tools import check_reader_parity as parity
from video2tenhou.perception.detector import Det
from video2tenhou.perception.tiles import CLASS_INDEX


class TwoTiles:
    """Find the same two upright tiles in every region."""

    def predict(self, _img: np.ndarray, /) -> list[Det]:
        """Return two fixed face boxes."""
        return [
            Det((10, 10, 30, 40), 0.9),
            Det((32, 10, 52, 40), 0.9),
        ]

    def predict_batch(self, imgs: list[np.ndarray], /) -> list[list[Det]]:
        """Detect each image independently."""
        return [self.predict(img) for img in imgs]


class BatchClassifier:
    """Name tiles 1m, or 2m when ``sensitive`` and called with a large batch."""

    def __init__(self, *, sensitive: bool) -> None:
        """Choose whether the batch size changes the identity."""
        self.sensitive = sensitive

    def classify(
        self, crops: list[np.ndarray], _sideways: list[bool] | None = None, /
    ) -> np.ndarray:
        """Return one-hot posteriors."""
        tile = "2m" if self.sensitive and len(crops) > 2 else "1m"
        posteriors = np.zeros((len(crops), len(CLASS_INDEX)), np.float32)
        posteriors[:, CLASS_INDEX[tile]] = 1
        return posteriors


@pytest.mark.parametrize("sensitive", [False, True])
def test_parity_reports_only_readings_changed_by_batching(
    tmp_path: Path, *, sensitive: bool
) -> None:
    """Identical readings pass; a batch-dependent identity names both images."""
    image = np.full((60, 80, 3), 40, np.uint8)
    for name in ("hand_TL_1.jpg", "meld_BR_2.jpg"):
        assert cv2.imwrite(str(tmp_path / name), image)
    result = parity.compare(tmp_path, TwoTiles(), BatchClassifier(sensitive=sensitive))
    assert result["images"] == 2
    assert len(result["changed_images"]) == (2 if sensitive else 0)
    assert result["max_posterior_delta"] == 0
