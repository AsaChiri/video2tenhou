# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Check that an sdist ships exactly the tracked release files plus the built UI.

Run from the Git checkout after ``uv build``. A small subset of the shipped tests
then runs from the extracted archive with an empty data directory, proving that
they need no private labels, videos, models or caches. The full suite runs
separately from the checkout.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ONLY = (".editorconfig", ".gitattributes", ".github/", "CLAUDE.md")
BUILT_INDEX = "src/video2tenhou/tool/static/index.html"
BUILT_ASSETS = "src/video2tenhou/tool/static/assets/"
SMOKE = (
    "tests/pipeline/test_layout.py",
    "tests/web/test_frontend_assets.py",
    "tests/integration/test_contradicted_meld.py",
)


def release_files() -> set[str]:
    """Return tracked files that belong in the source archive."""
    git = shutil.which("git")
    if git is None:
        raise FileNotFoundError(
            "The source-release check needs git to list tracked files"
        )
    listing = subprocess.run(  # noqa: S603  fixed git query of this checkout
        [git, "-C", str(ROOT), "ls-files", "-z"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout
    return {
        name
        for name in listing.split("\0")
        if name and not name.startswith(REPOSITORY_ONLY)
    }


def check_contents(names: set[str], expected: set[str]) -> None:
    """Reject missing or unexpected members, including private local data."""
    assets = {name for name in names if name.startswith(BUILT_ASSETS)}
    if BUILT_INDEX not in names or not {
        PurePosixPath(name).suffix for name in assets
    } >= {".js", ".css"}:
        raise ValueError("Source archive is missing the built frontend")
    missing = sorted(expected - names)
    unexpected = sorted(names - expected - assets - {BUILT_INDEX, "PKG-INFO"})
    if missing or unexpected:
        raise ValueError(
            f"Source archive differs from tracked files: {missing=} {unexpected=}"
        )


def check(archive: Path) -> None:
    """Verify the archive inventory, then run the smoke tests from its contents."""
    with tempfile.TemporaryDirectory(prefix="video2tenhou-source-") as directory:
        root = Path(directory)
        with tarfile.open(archive) as source:
            members = [member for member in source.getmembers() if member.isfile()]
            source.extractall(root, filter="data")
        (checkout,) = root.iterdir()
        check_contents(
            {member.name.split("/", 1)[1] for member in members}, release_files()
        )
        env = {
            **os.environ,
            "PYTHONPATH": str(checkout / "src"),
            "VIDEO2TENHOU_HOME": str(checkout / "local-data"),
            "PYTHONUTF8": "1",
        }
        subprocess.run(  # noqa: S603  current interpreter runs fixed shipped tests
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *SMOKE],
            cwd=checkout,
            env=env,
            check=True,
        )


def main() -> None:
    """Check a supplied source archive or the latest named sdist in dist/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path)
    args = parser.parse_args()
    archive = args.archive or max(Path("dist").glob("*.tar.gz"))
    check(archive.resolve())


if __name__ == "__main__":
    main()
