# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepare local weights for a separately reviewed release; never upload them.

Run from the repository with ``uv run python tools/package_models.py``. The
archive includes only runtime models/metadata plus SHA-256 checksums, never
training data, videos or external base checkpoints.
"""

import argparse
import hashlib
import json
import logging
import sys
import tempfile
import zipfile
from pathlib import Path

from video2tenhou.paths import MODEL_DIR
from video2tenhou.perception.detector_metadata import inference_settings
from video2tenhou.perception.evidence_policy import resolve_policy

FILES = (
    "detector/weights.pt",
    "detector/meta.json",
    "classifier/weights.pt",
    "classifier/meta.json",
)
METADATA = (
    "detector/provenance.json",
    "detector/LICENSE",
    "classifier/provenance.json",
    "classifier/LICENSE",
)


LOGGER = logging.getLogger("tools.package_models")


def _members(models: Path) -> dict[str, bytes]:
    missing = [name for name in FILES if not (models / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing runtime models: " + ", ".join(missing))
    members = {
        name: (models / name).read_bytes()
        for name in FILES + METADATA
        if (models / name).is_file()
    }
    meta = json.loads(members["detector/meta.json"])
    if (
        not isinstance(meta, dict)
        or meta.get("schema_version") != 1
        or meta.get("backend") != "libreyolo"
        or meta.get("classes") != {"0": "face"}
        or meta.get("architecture") not in {"yolo9-t", "yolo9-s", "yolo9-m", "yolo9-c"}
    ):
        msg = "Release metadata must describe a standard LibreYOLO YOLO9 face detector"
        raise ValueError(msg)
    actual = hashlib.sha256(members["detector/weights.pt"]).hexdigest()
    if meta.get("weights_sha256") != actual:
        msg = "Detector metadata does not match its weights SHA-256"
        raise ValueError(msg)

    inference_settings(meta.get("inference", {}))

    resolve_policy(meta.get("evidence_policy"))
    required = ("detector/provenance.json", "detector/LICENSE")
    missing = [name for name in required if name not in members]
    if missing:
        raise FileNotFoundError(
            "Missing detector release provenance/notices: " + ", ".join(missing)
        )
    return members


def _write_member(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    # Stable metadata makes identical inputs produce identical release hashes.
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content)


def bundle(models: Path, destination: Path) -> Path:
    """Write a reproducible runtime bundle, preserving model provenance/notices.

    Only allowlisted files enter the archive. Detector metadata is required and
    must describe the supported face-only backend and match its checkpoint;
    incomplete or mismatched inputs leave any existing bundle untouched. This
    tool packages files locally and makes no distribution-license decision.
    """
    members = _members(models)
    manifest = {"format": 1, "files": {}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".models-", suffix=".zip", delete=False
    ) as pending:
        temporary = Path(pending.name)
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for name, content in sorted(members.items()):
                manifest["files"]["models/" + name] = {
                    "bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                _write_member(archive, "models/" + name, content)
            _write_member(
                archive,
                "models/manifest.json",
                (json.dumps(manifest, indent=2) + "\n").encode(),
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    destination.with_suffix(".sha256").write_text(
        hashlib.sha256(destination.read_bytes()).hexdigest()
        + "  "
        + destination.name
        + "\n",
        encoding="utf-8",
    )
    return destination


def main() -> None:
    """Build a local model archive; custom paths permit separate data directories."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--out", type=Path, default=Path("dist/video2tenhou-pml-models.zip")
    )
    args = parser.parse_args()
    LOGGER.info("%s", bundle(args.models, args.out))


if __name__ == "__main__":
    main()
