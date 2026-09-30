# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Starter archives validate untrusted paths and preserve release inputs."""

import hashlib
import io
import json
import stat
import tarfile
import zipfile
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tools import package_starter as package

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


FRONTEND_ASSETS = {
    "src/video2tenhou/tool/static/assets/index-test.js",
    "src/video2tenhou/tool/static/assets/index-test.css",
}


def source_archive(
    path: "Path",
    extra: "Sequence[tuple[tarfile.TarInfo, bytes]]" = (),
    missing: "str | None" = None,
) -> "Path":
    """Build a source archive with controlled missing or extra members."""
    members = {
        name: name.encode() for name in package.SOURCE_REQUIRED | FRONTEND_ASSETS
    }
    members["pyproject.toml"] = b'[project]\nname="video2tenhou"\nversion="0.1.0"\n'
    if missing:
        members.pop(missing)
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo("video2tenhou-0.1.0/" + name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        for info, data in extra:
            archive.addfile(info, io.BytesIO(data) if info.isfile() else None)
    return path


def model_archive(
    path: "Path",
    *,
    change: "Callable[[dict[str, bytes], dict], None] | None" = None,
    extra: "Sequence[tuple[str | zipfile.ZipInfo, bytes]]" = (),
) -> "Path":
    """Build a model bundle with controlled manifest and member changes."""
    members = {name: name.encode() for name in package.MODEL_REQUIRED}
    manifest = {
        "format": 1,
        "files": {
            name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in members.items()
        },
    }
    if change:
        change(members, manifest)
    members["models/manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        for name, data in extra:
            archive.writestr(name, data)
    return path


@pytest.fixture
def archives(tmp_path: "Path") -> "tuple[Path, Path]":
    """Create source and model archives for starter packaging tests."""
    return source_archive(tmp_path / "source.tar.gz"), model_archive(
        tmp_path / "models.zip"
    )


def test_starter_is_reproducible_and_colocates_models_and_launchers(
    archives: "tuple[Path, Path]", tmp_path: "Path"
) -> None:
    """Verify starter is reproducible and colocates models and launchers."""
    source, models = archives
    before = source.read_bytes(), models.read_bytes()
    first = package.bundle(source, models, tmp_path / "starter.zip")
    second = package.bundle(source, models, tmp_path / "second.zip")
    assert first.read_bytes() == second.read_bytes()
    assert (source.read_bytes(), models.read_bytes()) == before
    with zipfile.ZipFile(first) as archive:
        prefix = "video2tenhou-0.1.0/"
        expected = (
            package.SOURCE_REQUIRED
            | FRONTEND_ASSETS
            | package.MODEL_REQUIRED
            | {"models/manifest.json"}
        )
        assert set(archive.namelist()) == {prefix + name for name in expected}
        assert archive.getinfo(prefix + "start.sh").external_attr >> 16 & 0o111 == 0o111
        manifest = json.loads(archive.read(prefix + "models/manifest.json"))
        assert set(manifest["files"]) == package.MODEL_REQUIRED
        assert not any(str(tmp_path) in name for name in archive.namelist())
    assert (
        first.with_suffix(".sha256").read_text().split()[0]
        == hashlib.sha256(first.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("missing", sorted(FRONTEND_ASSETS))
def test_starter_rejects_unbuilt_frontend(tmp_path: Path, missing: str) -> None:
    """Reject an incomplete application before publishing its starter archive."""
    source = source_archive(tmp_path / "source.tar.gz", missing=missing)
    models = model_archive(tmp_path / "models.zip")
    with pytest.raises(ValueError, match="built frontend"):
        package.bundle(source, models, tmp_path / "starter.zip")
    assert not (tmp_path / "starter.zip").exists()


@pytest.mark.parametrize(
    "name",
    [
        "/escape",
        "../escape",
        "a/../escape",
        "a\\escape",
        "C:/escape",
        "video2tenhou-0.1.0/README.md",
        "video2tenhou-0.1.0/readme.md",
        "video2tenhou-0.1.0/src",
        "video2tenhou-0.1.0/NUL.txt",
    ],
)
def test_source_rejects_unsafe_duplicate_and_colliding_paths(
    archives: "tuple[Path, Path]", tmp_path: "Path", name: str
) -> None:
    """Verify source rejects unsafe duplicate and colliding paths."""
    source, models = archives
    info = tarfile.TarInfo(name)
    info.size = 1
    source_archive(source, [(info, b"x")])
    previous = tmp_path / "starter.zip"
    previous.write_bytes(b"previous release")
    with pytest.raises(ValueError, match=r"Unsafe archive path|[Cc]olliding|collision"):
        package.bundle(source, models, previous)
    assert previous.read_bytes() == b"previous release"
    assert not list(tmp_path.glob(".starter-*"))


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_source_rejects_links(
    archives: "tuple[Path, Path]", tmp_path: "Path", kind: "bytes"
) -> None:
    """Verify source rejects links."""
    source, models = archives
    info = tarfile.TarInfo("video2tenhou-0.1.0/link")
    info.type, info.linkname = kind, "../outside"
    source_archive(source, [(info, b"")])
    with pytest.raises(ValueError, match="Unsupported source"):
        package.bundle(source, models, tmp_path / "starter.zip")


def test_source_rejects_a_second_root_even_when_empty(
    archives: "tuple[Path, Path]", tmp_path: "Path"
) -> None:
    """Verify source rejects a second root even when empty."""
    source, models = archives
    info = tarfile.TarInfo("another-package/")
    info.type = tarfile.DIRTYPE
    source_archive(source, [(info, b"")])
    with pytest.raises(ValueError, match="exactly one package root"):
        package.bundle(source, models, tmp_path / "starter.zip")


@pytest.mark.parametrize(
    "name",
    [
        "/escape",
        "../escape",
        "models\\escape",
        "C:/escape",
        "models/detector/weights.pt",
        "models/Detector/weights.pt",
        "models/detector",
        "models/private-video.mp4",
    ],
)
def test_models_reject_unsafe_duplicate_and_unlisted_entries(
    archives: "tuple[Path, Path]", tmp_path: "Path", name: str
) -> None:
    """Verify models reject unsafe duplicate and unlisted entries."""
    source, models = archives
    with (
        pytest.warns(UserWarning, match="Duplicate name")
        if name == "models/detector/weights.pt"
        else nullcontext()
    ):
        model_archive(models, extra=[(name, b"x")])
    with pytest.raises(
        ValueError,
        match=r"Unsafe archive path|[Cc]olliding|collision|Unexpected model file",
    ):
        package.bundle(source, models, tmp_path / "starter.zip")


def test_model_symlink_is_never_followed(
    archives: "tuple[Path, Path]", tmp_path: "Path"
) -> None:
    """Verify model symlink is never followed."""
    source, models = archives
    info = zipfile.ZipInfo("models/classifier/LICENSE")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    model_archive(models, extra=[(info, b"/outside")])
    with pytest.raises(ValueError, match="Unsupported model"):
        package.bundle(source, models, tmp_path / "starter.zip")


@pytest.mark.parametrize(
    "defect", ["missing", "hash", "size", "extra_inventory", "missing_inventory"]
)
def test_partial_or_unauthenticated_models_preserve_previous_release(
    archives: "tuple[Path, Path]", tmp_path: "Path", defect: str
) -> None:
    """Verify partial or unauthenticated models preserve previous release."""
    source, models = archives

    def damage(members: "dict[str, bytes]", manifest: "dict") -> None:
        name = "models/classifier/weights.pt"
        if defect == "missing":
            members.pop(name)
        elif defect == "hash":
            members[name] = b"wrong weights"
        elif defect == "size":
            manifest["files"][name]["bytes"] += 1
        elif defect == "extra_inventory":
            manifest["files"]["models/unlisted"] = {"bytes": 0, "sha256": "0" * 64}
        else:
            manifest["files"].pop(name)

    model_archive(models, change=damage)
    destination = tmp_path / "starter.zip"
    destination.write_bytes(b"previous")
    destination.with_suffix(".sha256").write_text("previous checksum")
    with pytest.raises(
        ValueError,
        match=r"missing required weights|inventory differs|checksum/size mismatch",
    ):
        package.bundle(source, models, destination)
    assert destination.read_bytes() == b"previous"
    assert destination.with_suffix(".sha256").read_text() == "previous checksum"


def test_source_requires_launchers_and_cannot_collide_with_models(
    archives: "tuple[Path, Path]", tmp_path: "Path"
) -> None:
    """Verify source requires launchers and cannot collide with models."""
    source, models = archives
    source_archive(source, missing="Start.cmd")
    with pytest.raises(ValueError, match="missing required"):
        package.bundle(source, models, tmp_path / "starter.zip")
    info = tarfile.TarInfo("video2tenhou-0.1.0/models")
    info.size = 1
    source_archive(source, [(info, b"x")])
    with pytest.raises(ValueError, match="collision"):
        package.bundle(source, models, tmp_path / "starter.zip")


def test_publication_failure_preserves_archive_and_cleans_partial_file(
    archives: "tuple[Path, Path]", tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """Verify publication failure preserves archive and cleans partial file."""
    source, models = archives
    destination = tmp_path / "starter.zip"
    destination.write_bytes(b"previous")

    def interrupted(*_unused_args: object, **_unused_kwargs: object) -> None:
        msg = "disk full"
        raise OSError(msg)

    monkeypatch.setattr(package.zipfile.ZipFile, "writestr", interrupted)
    with pytest.raises(OSError, match="disk full"):
        package.bundle(source, models, destination)
    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".starter-*"))


def test_output_cannot_replace_input_archive(archives: "tuple[Path, Path]") -> None:
    """Verify output cannot replace input archive."""
    source, models = archives
    before = models.read_bytes()
    with pytest.raises(ValueError, match="overwrite an input"):
        package.bundle(source, models, models)
    assert models.read_bytes() == before
