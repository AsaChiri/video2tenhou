# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0
"""Recognition identities for tests that replace the inference boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

from video2tenhou.layout import Calibration
from video2tenhou.perception.detector import Det
from video2tenhou.perception.evidence_policy import DEFAULT_POLICY, EvidencePolicy
from video2tenhou.perception.tiles import CLASSES
from video2tenhou.read import ReadModels


def models_stub() -> ReadModels:
    """Provide valid identities for tests that replace all video acquisition."""
    return (
        RecognitionStub("detector"),
        RecognitionStub("classifier"),
        Path("unread-test-video.mp4"),
        Calibration.load("pml"),
    )


@dataclass
class RecognitionStub:
    """Carry real cache metadata and fail if unexpected inference is attempted."""

    id: str
    classes: list[str] = field(default_factory=lambda: list(CLASSES))
    T: float = 1.0
    evidence_policy: EvidencePolicy = DEFAULT_POLICY

    def predict(self, _img: np.ndarray, /) -> list[Det]:
        """Reject inference outside the test's explicit replacement boundary."""
        pytest.fail("The test must replace the inference boundary")

    def predict_batch(self, _imgs: list[np.ndarray], /) -> list[list[Det]]:
        """Reject inference outside the test's explicit replacement boundary."""
        pytest.fail("The test must replace the inference boundary")

    def classify(
        self, _crops: list[np.ndarray], _sideways: list[bool] | None = None, /
    ) -> np.ndarray:
        """Reject inference outside the test's explicit replacement boundary."""
        pytest.fail("The test must replace the inference boundary")
