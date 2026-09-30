# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Rebuild selected hands in a fresh process using current installed engine code."""

import argparse
import json
from pathlib import Path

from video2tenhou import cli
from video2tenhou.engine import decode
from video2tenhou.layout import Calibration
from video2tenhou.record import from_dict


def main(argv: list[str] | None = None) -> None:
    """Reconstruct requested hands, then publish the complete set of game outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("hands", help="Comma-separated hand IDs, or all")
    parser.add_argument("video", type=Path)
    parser.add_argument("calibration")
    parser.add_argument("out", type=Path)
    args = parser.parse_args(argv)
    hands = json.loads((args.work / "hands.json").read_text(encoding="utf-8"))
    games = [
        from_dict(row)
        for row in json.loads((args.work / "record.json").read_text(encoding="utf-8"))
    ]
    selected = (
        None if args.hands == "all" else {int(value) for value in args.hands.split(",")}
    )
    decode.run_decode(
        args.work,
        hands,
        games,
        log=lambda _message: None,
        options=decode.DecodeRunOptions(
            force=True,
            only=selected,
            video_path=args.video,
            cal=Calibration.load(args.calibration, str(args.video)),
        ),
    )
    decodes = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.work / "decode").glob("*.json"))
    ]
    cli.write_outputs(args.out, games, decodes, hands, args.video.stem)


if __name__ == "__main__":
    main()
