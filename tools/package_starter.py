"""Combine a built source release and authenticated model bundle for local startup.

Inputs are read without extraction. No environment, model or source file is
modified, and no dependencies are imported beyond the Python standard library.
"""

import argparse
import hashlib
import json
import re
import stat
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


def _path(name: str) -> str:
    # Reject Windows aliases too: the ZIP is intended for both supported OSes.
    if not name or "\\" in name or name.startswith("/") or ":" in name:
        raise ValueError(f"Unsafe archive path: {name!r}")
    value = name.removesuffix("/")
    parts = value.split("/")
    reserved = re.compile(
        r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", re.IGNORECASE
    )
    if any(
        part in ("", ".", "..")
        or part.endswith((".", " "))
        or reserved.fullmatch(part)
        or any(ord(char) < 32 for char in part)
        for part in parts
    ):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return value


def _register(seen: dict[str, bool], name: str, directory: bool) -> str:
    value = _path(name)
    key = value.casefold()
    if key in seen:
        raise ValueError(f"Duplicate or colliding archive entry: {name}")
    parents = [
        str(parent).casefold()
        for parent in PurePosixPath(value).parents
        if str(parent) != "."
    ]
    if any(parent in seen and not seen[parent] for parent in parents) or (
        not directory and any(old.startswith(key + "/") for old in seen)
    ):
        raise ValueError(f"File/directory collision: {name}")
    seen[key] = directory
    return value


def _source(path: Path) -> tuple[str, dict[str, bytes]]:
    entries, seen = {}, {}
    roots = set()
    with tarfile.open(path, "r:*") as archive:
        for info in archive:
            if not (info.isfile() or info.isdir()):
                raise ValueError(f"Unsupported source archive entry: {info.name}")
            name = _register(seen, info.name, info.isdir())
            roots.add(name.split("/")[0])
            if info.isfile():
                with archive.extractfile(info) as stream:
                    entries[name] = stream.read()
    if len(roots) != 1:
        raise ValueError("Source archive must contain exactly one package root")
    prefix = roots.pop() + "/"
    if any(not name.startswith(prefix) for name in entries):
        raise ValueError("Source files must be below the package root")
    entries = {name[len(prefix) :]: data for name, data in entries.items()}
    if not entries.keys() >= SOURCE_REQUIRED:
        raise ValueError(
            "Source release is missing required launchers or project files"
        )
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
        raise ValueError("Source release has no project metadata table")
    version = project.get("version", "")
    if (
        project.get("name") != "video2tenhou"
        or not isinstance(version, str)
        or not re.fullmatch(r"[0-9][0-9A-Za-z.+-]*", version)
    ):
        raise ValueError(
            "Source release must identify video2tenhou and a safe static version"
        )
    return version, entries


def _models(path: Path) -> dict[str, bytes]:
    entries, seen = {}, {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise ValueError(f"Unsupported model archive entry: {info.filename}")
            name = _register(seen, info.filename, info.is_dir())
            if info.is_dir():
                if not any(allowed.startswith(name + "/") for allowed in MODEL_ALLOWED):
                    raise ValueError(f"Unexpected model directory: {name}")
            else:
                if name not in MODEL_ALLOWED | {"models/manifest.json"}:
                    raise ValueError(f"Unexpected model file: {name}")
                entries[name] = archive.read(info)
    if not MODEL_REQUIRED | {"models/manifest.json"} <= entries.keys():
        raise ValueError(
            "Model archive is missing required weights, metadata or provenance"
        )
    manifest = json.loads(entries["models/manifest.json"])
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("format")) is not int
        or manifest["format"] != 1
    ):
        raise ValueError("Unsupported model manifest")
    inventory = manifest.get("files")
    if not isinstance(inventory, dict) or set(inventory) != entries.keys() - {
        "models/manifest.json"
    }:
        raise ValueError("Model manifest inventory differs from archive files")
    for name, expected in inventory.items():
        data = entries[name]
        if (
            not isinstance(expected, dict)
            or type(expected.get("bytes")) is not int
            or expected["bytes"] != len(data)
            or expected.get("sha256") != hashlib.sha256(data).hexdigest()
        ):
            raise ValueError(f"Model checksum/size mismatch: {name}")
    return entries


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
        raise ValueError("Starter output must not overwrite an input archive")
    version, entries = _source(source)
    models_data = _models(models)
    seen = {}
    for name in (*entries, *models_data):
        _register(seen, name, False)
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
    """Build a starter from explicit source/model releases without touching a runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(bundle(args.source, args.models, args.out))


if __name__ == "__main__":
    main()
