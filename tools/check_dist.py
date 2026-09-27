"""Smoke-test wheel resources outside the checkout, using installed dependencies."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


def check(wheel: Path) -> None:
    """Extract a wheel and validate its isolated imports and required static assets."""
    with tempfile.TemporaryDirectory(prefix="video2tenhou-wheel-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            assert not any(name.startswith(("labels/", "work/", "models/", "out/")) for name in names)
            archive.extractall(root)
        script = '''
import os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
os.environ["VIDEO2TENHOU_HOME"] = str(Path.cwd() / "local-data")
from video2tenhou import layout, overlay, paths
from video2tenhou.tool import server
assert Path(layout.__file__).is_relative_to(Path(sys.argv[1]))
assert layout.Calibration.load("pml").name == "pml"
assert overlay._DIGITS.ready and overlay._WIND.ready
assert paths.LABEL_DIR == Path.cwd() / "local-data" / "labels"
static = Path(server.__file__).parent / "static"
assert (static / "index.html").is_file()
assert (static / "studio.html").is_file()
assert list(static.rglob("*.svg")), "Tile art missing from wheel"
print("Installed wheel resources and workspace isolation verified")
'''
        subprocess.run([sys.executable, "-I", "-c", script, str(root)], cwd=root, check=True)


def main() -> None:
    """Validate a supplied wheel, or the most recently named wheel in dist/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="?", type=Path)
    args = parser.parse_args()
    wheel = args.wheel or sorted(Path("dist").glob("*.whl"))[-1]
    check(wheel.resolve())


if __name__ == "__main__":
    main()
