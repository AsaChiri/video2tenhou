# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Resolve external command dependencies before launching a child process."""

import shutil
from pathlib import Path


def executable(name: str) -> str:
    """Return an absolute executable from PATH, or fail before starting work."""
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"Required executable {name!r} was not found on PATH")
    return str(Path(path).resolve())
