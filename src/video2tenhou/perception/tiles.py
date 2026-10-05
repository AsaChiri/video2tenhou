# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Ordered classifier vocabulary: 34 tile kinds, red fives, face-down and none."""

KINDS = [f"{n}{s}" for s in "mps" for n in range(1, 10)] + [
    f"{n}z" for n in range(1, 8)
]
CLASSES = [*KINDS, "0m", "0p", "0s", "X", "none"]  # 39
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
