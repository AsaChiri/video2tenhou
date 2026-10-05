# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Build the release model bundle and starter download locally; never upload them.

    uv run python tools/package_release.py models
    uv run python tools/package_release.py starter --source SDIST --models ZIP --out ZIP

The model bundle holds only runtime weights, metadata, provenance and notices,
with a per-file SHA-256 manifest; never training data, videos or base
checkpoints. The starter combines the built source release with a bundle whose
manifest matches. Identical inputs give identical archives, each published
atomically beside a ``.sha256`` file.
"""

import argparse
import hashlib
import json
import logging
import stat
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

from video2tenhou.paths import MODEL_DIR
from video2tenhou.perception.detector_metadata import inference_settings
from video2tenhou.perception.evidence_policy import resolve_policy

MODEL_FILES = (
    "detector/weights.pt",
    "detector/meta.json",
    "detector/provenance.json",
    "detector/LICENSE",
    "classifier/weights.pt",
    "classifier/meta.json",
    "classifier/provenance.json",
)
OPTIONAL_MODEL_FILES = ("classifier/LICENSE",)
DETECTOR_ARCHITECTURES = {"yolo9-t", "yolo9-s", "yolo9-m", "yolo9-c"}
MANIFEST = "models/manifest.json"
STARTER_FILES = (
    "Start.cmd",
    "start.sh",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "src/video2tenhou/tool/static/index.html",
)
BUILT_ASSETS = "src/video2tenhou/tool/static/assets/"

LOGGER = logging.getLogger("tools.package_release")


def sha256(content: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of in-memory content."""
    return hashlib.sha256(content).hexdigest()


def write_zip(
    destination: Path, members: dict[str, bytes], *, executable: str = ""
) -> Path:
    """Publish a deterministic ZIP and its checksum, keeping any old file on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as stream:
        pending = Path(stream.name)
    try:
        with zipfile.ZipFile(pending, "w") as archive:
            for name, content in sorted(members.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                mode = 0o755 if name == executable else 0o644
                info.external_attr = (stat.S_IFREG | mode) << 16
                archive.writestr(info, content)
        digest = sha256(pending.read_bytes())
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    destination.with_suffix(".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8"
    )
    return destination


def bundle_models(models: Path, destination: Path) -> Path:
    """Package runtime models whose detector metadata matches its weights."""
    missing = [name for name in MODEL_FILES if not (models / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing release model files: " + ", ".join(missing))
    members = {
        f"models/{name}": (models / name).read_bytes()
        for name in MODEL_FILES + OPTIONAL_MODEL_FILES
        if (models / name).is_file()
    }
    meta = json.loads(members["models/detector/meta.json"])
    if (
        not isinstance(meta, dict)
        or meta.get("schema_version") != 1
        or meta.get("backend") != "libreyolo"
        or meta.get("classes") != {"0": "face"}
        or meta.get("architecture") not in DETECTOR_ARCHITECTURES
    ):
        raise ValueError(
            "Release metadata must describe a standard LibreYOLO YOLO9 face detector"
        )
    if meta.get("weights_sha256") != sha256(members["models/detector/weights.pt"]):
        raise ValueError("Detector metadata does not match its weights SHA-256")
    inference_settings(meta.get("inference", {}))
    resolve_policy(meta.get("evidence_policy"))
    files = {
        name: {"bytes": len(content), "sha256": sha256(content)}
        for name, content in sorted(members.items())
    }
    manifest = {"format": 1, "files": files}
    members[MANIFEST] = (json.dumps(manifest, indent=2) + "\n").encode()
    return write_zip(destination, members)


def _source_files(source: Path) -> tuple[str, dict[str, bytes]]:
    """Read an sdist's files below its root and require a launchable application."""
    files = {}
    with tarfile.open(source) as archive:
        for member in archive.getmembers():
            content = archive.extractfile(member) if member.isfile() else None
            if content is not None:
                files[member.name.split("/", 1)[1]] = content.read()
    built = {
        PurePosixPath(name).suffix for name in files if name.startswith(BUILT_ASSETS)
    }
    missing = [name for name in STARTER_FILES if name not in files]
    if missing or not built >= {".js", ".css"}:
        raise ValueError(
            f"Source release lacks launchers, project files or built UI: {missing}"
        )
    version = tomllib.loads(files["pyproject.toml"].decode("utf-8"))["project"]
    return version["version"], files


def _bundled_models(models: Path) -> dict[str, bytes]:
    """Read a model bundle whose files exactly match its checksum manifest."""
    with zipfile.ZipFile(models) as archive:
        files = {
            info.filename: archive.read(info)
            for info in archive.infolist()
            if not info.is_dir()
        }
    inventory = json.loads(files.get(MANIFEST, b"{}")).get("files", {})
    actual = {
        name: {"bytes": len(content), "sha256": sha256(content)}
        for name, content in files.items()
        if name != MANIFEST
    }
    required = {f"models/{name}" for name in MODEL_FILES}
    if inventory != actual or not required <= actual.keys():
        raise ValueError("Model bundle is incomplete or differs from its manifest")
    return files


def bundle_starter(source: Path, models: Path, destination: Path) -> Path:
    """Combine the application and verified models under one versioned folder."""
    version, files = _source_files(source)
    root = f"video2tenhou-{version}/"
    members = {root + name: content for name, content in files.items()}
    members |= {
        root + name: content for name, content in _bundled_models(models).items()
    }
    return write_zip(destination, members, executable=root + "start.sh")


def main(argv: list[str] | None = None) -> None:
    """Build the selected release archive and log its path."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    models = commands.add_parser("models", help="bundle the local runtime models")
    models.add_argument("--models", type=Path, default=MODEL_DIR)
    models.add_argument(
        "--out", type=Path, default=Path("dist/video2tenhou-pml-models.zip")
    )
    starter = commands.add_parser("starter", help="combine a source release and models")
    for name in ("source", "models", "out"):
        starter.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "models":
        LOGGER.info("%s", bundle_models(args.models, args.out))
    else:
        LOGGER.info("%s", bundle_starter(args.source, args.models, args.out))


if __name__ == "__main__":
    main()
