# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Combine a built source release and authenticated model bundle for local startup.

Inputs are read without extraction. No environment, model or source file is
modified, and no dependencies are imported beyond the Python standard library.
"""

import argparse
import hashlib
import json
import logging
import re
import stat
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

MODEL_REQUIRED = frozenset(
    "models/" + name
    for name in (
        "detector/weights.pt",
        "detector/meta.json",
        "detector/provenance.json",
        "detector/LICENSE",
        "classifier/weights.pt",
        "classifier/meta.json",
        "classifier/provenance.json",
    )
)
MODEL_ALLOWED = MODEL_REQUIRED | {"models/classifier/LICENSE"}
SOURCE_REQUIRED = {
    "pyproject.toml",
    "uv.lock",
    "Start.cmd",
    "start.sh",
    "README.md",
    "LICENSE",
    "src/video2tenhou/cli.py",
    "src/video2tenhou/tool/static/index.html",
    "THIRD_PARTY_NOTICES.md",
}


LOGGER = logging.getLogger("tools.package_starter")
FIRST_PRINTABLE_CHARACTER = 32


def _path(name: str) -> str:
    # Reject Windows aliases too: the ZIP is intended for both supported OSes.
    if not name or "\\" in name or name.startswith("/") or ":" in name:
        msg = f"Unsafe archive path: {name!r}"
        raise ValueError(msg)
    value = name.removesuffix("/")
    parts = value.split("/")
    reserved = re.compile(
        r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", re.IGNORECASE
    )
    if any(
        part in ("", ".", "..")
        or part.endswith((".", " "))
        or reserved.fullmatch(part)
        or any(ord(char) < FIRST_PRINTABLE_CHARACTER for char in part)
        for part in parts
    ):
        msg = f"Unsafe archive path: {name!r}"
        raise ValueError(msg)
    return value


def _register(seen: dict[str, bool], name: str, *, directory: bool) -> str:
    value = _path(name)
    key = value.casefold()
    if key in seen:
        msg = f"Duplicate or colliding archive entry: {name}"
        raise ValueError(msg)
    parents = [
        str(parent).casefold()
        for parent in PurePosixPath(value).parents
        if str(parent) != "."
    ]
    if any(parent in seen and not seen[parent] for parent in parents) or (
        not directory and any(old.startswith(key + "/") for old in seen)
    ):
        msg = f"File/directory collision: {name}"
        raise ValueError(msg)
    seen[key] = directory
    return value


def _source(path: Path) -> tuple[str, dict[str, bytes]]:
    entries, seen = {}, {}
    roots = set()
    with tarfile.open(path, "r:*") as archive:
        for info in archive:
            if not (info.isfile() or info.isdir()):
                msg = f"Unsupported source archive entry: {info.name}"
                raise ValueError(msg)
            name = _register(seen, info.name, directory=info.isdir())
            roots.add(name.split("/")[0])
            if info.isfile():
                stream = archive.extractfile(info)
                if stream is None:
                    msg = f"Source archive file has no content: {info.name}"
                    raise ValueError(msg)
                with stream:
                    entries[name] = stream.read()
    if len(roots) != 1:
        msg = "Source archive must contain exactly one package root"
        raise ValueError(msg)
    prefix = roots.pop() + "/"
    if any(not name.startswith(prefix) for name in entries):
        msg = "Source files must be below the package root"
        raise ValueError(msg)
    entries = {name[len(prefix) :]: data for name, data in entries.items()}
    return _source_version(entries), entries


def _source_version(entries: dict[str, bytes]) -> str:
    """Validate launchers, built UI assets and project identity before bundling."""
    if not entries.keys() >= SOURCE_REQUIRED:
        msg = "Source release is missing required launchers or project files"
        raise ValueError(msg)
    for suffix in (".js", ".css"):
        if not any(
            name.startswith("src/video2tenhou/tool/static/assets/")
            and name.endswith(suffix)
            for name in entries
        ):
            message = "Source release is missing built frontend assets"
            raise ValueError(message)
    project = tomllib.loads(entries["pyproject.toml"].decode("utf-8")).get(
        "project", {}
    )
    if not isinstance(project, dict):
        msg = "Source release has no project metadata table"
        raise TypeError(msg)
    version = project.get("version", "")
    if (
        project.get("name") != "video2tenhou"
        or not isinstance(version, str)
        or not re.fullmatch(r"[0-9][0-9A-Za-z.+-]*", version)
    ):
        msg = "Source release must identify video2tenhou and a safe static version"
        raise ValueError(msg)
    return version


def _models(path: Path) -> dict[str, bytes]:
    entries, seen = {}, {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                msg = f"Unsupported model archive entry: {info.filename}"
                raise ValueError(msg)
            name = _register(seen, info.filename, directory=info.is_dir())
            if info.is_dir():
                if not any(allowed.startswith(name + "/") for allowed in MODEL_ALLOWED):
                    msg = f"Unexpected model directory: {name}"
                    raise ValueError(msg)
            else:
                if name not in MODEL_ALLOWED | {"models/manifest.json"}:
                    msg = f"Unexpected model file: {name}"
                    raise ValueError(msg)
                entries[name] = archive.read(info)
    _verify_model_inventory(entries)
    return entries


def _verify_model_inventory(entries: dict[str, bytes]) -> None:
    """Require a complete model manifest and authenticate every bundled file."""
    if not MODEL_REQUIRED | {"models/manifest.json"} <= entries.keys():
        msg = "Model archive is missing required weights, metadata or provenance"
        raise ValueError(msg)
    manifest = json.loads(entries["models/manifest.json"])
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("format")) is not int
        or manifest["format"] != 1
    ):
        msg = "Unsupported model manifest"
        raise ValueError(msg)
    inventory = manifest.get("files")
    if not isinstance(inventory, dict) or set(inventory) != entries.keys() - {
        "models/manifest.json"
    }:
        msg = "Model manifest inventory differs from archive files"
        raise ValueError(msg)
    for name, expected in inventory.items():
        data = entries[name]
        if (
            not isinstance(expected, dict)
            or type(expected.get("bytes")) is not int
            or expected["bytes"] != len(data)
            or expected.get("sha256") != hashlib.sha256(data).hexdigest()
        ):
            msg = f"Model checksum/size mismatch: {name}"
            raise ValueError(msg)


def bundle(source: Path, models: Path, destination: Path) -> Path:
    """Publish a deterministic starter ZIP after validating both input archives.

    Reject unsafe paths, links, duplicate/colliding entries, incomplete source
    and unauthenticated model inventory before replacing an existing output.
    Launchers and models share one versioned root. The shell launcher is marked
    executable for ZIP extractors that preserve Unix modes. A sibling SHA-256
    file authenticates the resulting archive; nothing is uploaded.
    """
    inputs = {source.resolve(), models.resolve()}
    if (
        destination.resolve() in inputs
        or destination.with_suffix(".sha256").resolve() in inputs
    ):
        msg = "Starter output must not overwrite an input archive"
        raise ValueError(msg)
    version, entries = _source(source)
    models_data = _models(models)
    seen = {}
    for name in (*entries, *models_data):
        _register(seen, name, directory=False)
    entries.update(models_data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=".starter-", suffix=".zip", delete=False
    ) as stream:
        pending = Path(stream.name)
    try:
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in sorted(entries.items()):
                info = zipfile.ZipInfo(
                    f"video2tenhou-{version}/{name}", date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = (
                    stat.S_IFREG | (0o755 if name == "start.sh" else 0o644)
                ) << 16
                archive.writestr(info, data)
        digest = hashlib.sha256(pending.read_bytes()).hexdigest()
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    destination.with_suffix(".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8"
    )
    return destination


def main() -> None:
    """Build a starter from source/model releases without changing a runtime."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    LOGGER.info("%s", bundle(args.source, args.models, args.out))


if __name__ == "__main__":
    main()
