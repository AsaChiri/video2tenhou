# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared objective-gap policy for solver stopping and human review decisions."""

MARGIN_REVIEW = 0.5
MARGIN_TOLERANCE = 1e-9


def low_margin(margin: float | None) -> bool:
    """Whether a measured choice remains uncertain, including the review boundary.

    Objectives use fixed-point costs but are subtracted as floats. The tiny
    tolerance prevents a mathematical margin of 0.5 rounding just above it.
    Missing margins are not measured decisions (for example call turns).
    """
    return margin is not None and margin <= MARGIN_REVIEW + MARGIN_TOLERANCE


def confidence_state(margin: float | None, alternative_gap: float | None) -> str:
    """Separate a proof, a competing witness, and an unfinished calculation."""
    if margin is None:
        return "unmeasured"
    if margin > MARGIN_REVIEW + MARGIN_TOLERANCE:
        return "resolved"
    if low_margin(alternative_gap):
        return "ambiguous"
    return "unresolvable"
