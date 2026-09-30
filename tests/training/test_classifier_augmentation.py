# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Modern augmentation generators reproduce independent training sample streams."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from video2tenhou.train import train_classifier

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_seeded_crop_streams_are_reproducible_and_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interleaved datasets agree for equal seeds without repeating each sample."""
    folder = tmp_path / "1m"
    folder.mkdir()
    image = np.arange(64 * 32 * 3, dtype=np.uint8).reshape(64, 32, 3)
    assert cv2.imwrite(str(folder / "hand_TL_0.png"), image)
    monkeypatch.setattr(train_classifier, "OCCLUSION_PROBABILITY", 1.0)
    first = train_classifier.Crops(tmp_path, augment=True, seed=7)
    repeated = train_classifier.Crops(tmp_path, augment=True, seed=7)
    different = train_classifier.Crops(tmp_path, augment=True, seed=8)
    first_image, first_label, first_kind = first[0]
    other_image, _, _ = different[0]
    repeated_image, repeated_label, repeated_kind = repeated[0]
    assert torch.equal(first_image, repeated_image)
    assert (first_label, first_kind) == (repeated_label, repeated_kind)
    assert not torch.equal(first_image, other_image)
    next_image, _, _ = first[0]
    repeated_next, _, _ = repeated[0]
    assert torch.equal(next_image, repeated_next)
    assert not torch.equal(first_image, next_image)


def test_validation_crops_do_not_consume_augmentation_randomness(
    tmp_path: Path,
) -> None:
    """Validation reads retain original pixels and leave their generator untouched."""
    folder = tmp_path / "1m"
    folder.mkdir()
    image = np.full((64, 32, 3), 120, dtype=np.uint8)
    assert cv2.imwrite(str(folder / "hand_TL_0.png"), image)
    validation = train_classifier.Crops(tmp_path, augment=False)
    expected = train_classifier.to_tensor(image)
    initial_state = validation.rng.bit_generator.state
    actual, _, _ = validation[0]
    assert torch.equal(actual, expected)
    assert validation.rng.bit_generator.state == initial_state
