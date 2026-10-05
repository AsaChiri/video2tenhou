# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Shared objective-gap policy for solver certificates and human review decisions."""

import math
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Certificate:
    """How much objective cost any reconstruction that changes one decision adds.

    ``margin`` is a certified lower bound on that cost increase. ``gap`` is the
    increase of an alternative actually found (None when none was found), and
    ``runner_up`` its tile when the decision is a single tile.
    """

    margin: float
    gap: float | None = None
    runner_up: str | None = None

    @property
    def state(self) -> str:
        """Resolved, ambiguous or unresolvable under the review threshold."""
        return confidence_state(self.margin, self.gap)


# A decision a human fact or a rule fixes: no reconstruction changes it.
FIXED = Certificate(math.inf, math.inf)
