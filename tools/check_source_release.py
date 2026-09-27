"""Run the shipped tests from an sdist without private workspace data.

Uses the current environment's dependencies. The wheel smoke check separately
verifies installed resource lookup. Run after ``uv build``.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def check(archive: Path) -> None:
    """Extract the source archive into a temporary workspace and run its own tests."""
    with tempfile.TemporaryDirectory(prefix="video2tenhou-source-") as directory:
        root = Path(directory)
        with tarfile.open(archive) as source:
            source.extractall(root, filter="data")
        checkout, = root.iterdir()
        env = {**os.environ, "PYTHONPATH": str(checkout / "src"),
               "VIDEO2TENHOU_HOME": str(checkout / "local-data"), "PYTHONUTF8": "1"}
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-ra"], cwd=checkout, env=env, check=True)


def main() -> None:
    """Check a supplied source archive or the latest named sdist in dist/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path)
    args = parser.parse_args()
    archive = args.archive or sorted(Path("dist").glob("*.tar.gz"))[-1]
    check(archive.resolve())


if __name__ == "__main__":
    main()
