# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0
"""Construct real decoder instances at a specified pipeline stage."""

from video2tenhou.engine.decode import DecodeOptions, HandDecoder
from video2tenhou.record import HandResult


def hand_decoder(**fields: object) -> HandDecoder:
    """Initialize a decoder, then install the stage inputs supplied by a test."""
    decoder = HandDecoder(
        {},
        {},
        HandResult(0, 0, 0, {}, "draw"),
        {},
        options=DecodeOptions(time_limit=1.0, models=None, work_dir=None),
    )
    for name, value in fields.items():
        setattr(decoder, name, value)
    return decoder
