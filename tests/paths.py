# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Stable source and fixture locations, independent of test package depth."""

from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
DATA = TESTS / "data"
