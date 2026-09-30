# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Run a recording's border checks in a fresh process and emit one JSON result."""

import argparse
import json
from pathlib import Path

from video2tenhou.calibfit import check_all
from video2tenhou.layout import Calibration
from video2tenhou.logging_setup import RESULT, command_logging
from video2tenhou.perception.detector import Detector


@command_logging
def main(argv: list[str] | None = None) -> None:
    """Check the current geometry while keeping status logs separate from the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("calibration")
    parser.add_argument("work", type=Path)
    args = parser.parse_args(argv)
    checks = check_all(
        args.video,
        Calibration.load(args.calibration, str(args.video)),
        Detector(),
        args.work,
    )
    result = {
        check.region: {
            "level": check.level,
            "held": check.held,
            "cut": check.cut,
            "note": check.note,
        }
        for check in checks
    }
    RESULT.info("%s", json.dumps(result))


if __name__ == "__main__":
    main()
