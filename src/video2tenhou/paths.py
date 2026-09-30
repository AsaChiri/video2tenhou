# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Separate immutable package resources from the user's writable workspace.

Set VIDEO2TENHOU_HOME before starting the process to keep models, labels and
jobs somewhere other than the current directory. Import-time paths deliberately
stay stable for the lifetime of a run, even if a caller changes directory.
"""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("VIDEO2TENHOU_HOME", Path.cwd())).expanduser().resolve()
ASSET_DIR = Path(__file__).resolve().parent / "assets"
MODEL_DIR = DATA_DIR / "models"
LABEL_DIR = DATA_DIR / "labels"
