# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test wheel resources outside the checkout, using installed dependencies."""

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def check(wheel: Path) -> None:
    """Extract a wheel and validate its isolated imports and required static assets."""
    with tempfile.TemporaryDirectory(prefix="video2tenhou-wheel-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if any(
                name.startswith(("labels/", "work/", "models/", "out/"))
                for name in names
            ):
                raise ValueError("Wheel contains private workspace data")
            archive.extractall(root)
        script = """
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
os.environ["VIDEO2TENHOU_HOME"] = str(Path.cwd() / "local-data")
from video2tenhou import layout, paths
from video2tenhou.logging_setup import command_logging
from video2tenhou.tool import server
import logging
assert Path(layout.__file__).is_relative_to(Path(sys.argv[1]))
assert layout.Calibration.load("pml").name == "pml"
assert paths.LABEL_DIR == (Path.cwd() / "local-data" / "labels").resolve()
static = Path(server.__file__).parent / "static"
assert (static / "index.html").is_file()
assert not (static / "studio.html").exists()
assert list((static / "assets").glob("*.js")), "Frontend JavaScript missing from wheel"
assert list((static / "assets").glob("*.css")), "Frontend styles missing from wheel"
assert list(static.rglob("*.svg")), "Tile art missing from wheel"
@command_logging
def report():
    logging.getLogger("video2tenhou.check_dist").info(
        "Installed wheel resources and workspace isolation verified"
    )
report()
"""
        subprocess.run(  # noqa: S603
            [sys.executable, "-I", "-c", script, str(root)], cwd=root, check=True
        )


def main() -> None:
    """Validate a supplied wheel, or the most recently named wheel in dist/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="?", type=Path)
    args = parser.parse_args()
    wheel = args.wheel or max(Path("dist").glob("*.whl"))
    check(wheel.resolve())


if __name__ == "__main__":
    main()
