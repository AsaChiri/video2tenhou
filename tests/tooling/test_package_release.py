# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Release archives are reproducible, authenticated and keep provenance."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools import package_release as package

ASSETS = {
    package.BUILT_ASSETS + "index-test.js",
    package.BUILT_ASSETS + "index-test.css",
}


@pytest.fixture
def models(tmp_path: Path) -> Path:
    """Create the runtime model files a release bundle requires."""
    root = tmp_path / "models"
    for name in package.MODEL_FILES:
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(name.encode())
    (root / "training-private.json").write_text("not for distribution")
    meta = {
        "schema_version": 1,
        "backend": "libreyolo",
        "architecture": "yolo9-s",
        "classes": {"0": "face"},
        "weights_sha256": package.sha256(b"detector/weights.pt"),
    }
    (root / "detector/meta.json").write_text(json.dumps(meta))
    return root


def source_archive(path: Path, missing: str | None = None) -> Path:
    """Build an sdist-shaped archive, optionally without one required file."""
    members = {name: name.encode() for name in {*package.STARTER_FILES, *ASSETS}}
    members["pyproject.toml"] = b'[project]\nname = "video2tenhou"\nversion = "0.1.0"\n'
    members.pop(missing or "", None)
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo("video2tenhou-0.1.0/" + name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def rewrite(path: Path, name: str, content: bytes | None) -> None:
    """Replace or remove one member of a model bundle without updating its manifest."""
    with zipfile.ZipFile(path) as archive:
        files = {info.filename: archive.read(info) for info in archive.infolist()}
    if content is None:
        files.pop(name)
    else:
        files[name] = content
    with zipfile.ZipFile(path, "w") as archive:
        for member, data in files.items():
            archive.writestr(member, data)


def test_model_bundle_is_reproducible_and_authenticates_only_runtime_files(
    models: Path, tmp_path: Path
) -> None:
    """Identical models give identical bytes and private files stay out."""
    first = package.bundle_models(models, tmp_path / "first.zip")
    second = package.bundle_models(models, tmp_path / "second.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(archive.read(package.MANIFEST))
        assert set(archive.namelist()) == set(manifest["files"]) | {package.MANIFEST}
        assert set(manifest["files"]) == {f"models/{n}" for n in package.MODEL_FILES}
        for name, entry in manifest["files"].items():
            assert entry["sha256"] == package.sha256(archive.read(name))
    assert (
        first.with_suffix(".sha256").read_text().split()[0]
        == hashlib.sha256(first.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "defect",
    [
        {"weights_sha256": "0" * 64},
        {"backend": "unsupported"},
        {"classes": {"0": "face", "1": "back"}},
        {"evidence_policy": {"schema_version": 99}},
        {"inference": {"confidence": float("nan")}},
        "detector/LICENSE",
        "detector/provenance.json",
        "classifier/weights.pt",
    ],
)
def test_invalid_models_leave_an_existing_bundle_untouched(
    models: Path, tmp_path: Path, defect: dict | str
) -> None:
    """Mismatched metadata or a missing notice never replaces a release."""
    if isinstance(defect, str):
        (models / defect).unlink()
    else:
        meta = json.loads((models / "detector/meta.json").read_text())
        (models / "detector/meta.json").write_text(json.dumps(meta | defect))
    target = tmp_path / "release.zip"
    target.write_bytes(b"previous release")
    with pytest.raises((ValueError, FileNotFoundError)):
        package.bundle_models(models, target)
    assert target.read_bytes() == b"previous release"
    assert [path.name for path in tmp_path.glob(".release.zip.*")] == []


def test_starter_is_reproducible_and_colocates_application_and_models(
    models: Path, tmp_path: Path
) -> None:
    """One versioned folder holds the launchers, built UI and verified models."""
    source = source_archive(tmp_path / "source.tar.gz")
    bundle = package.bundle_models(models, tmp_path / "models.zip")
    first = package.bundle_starter(source, bundle, tmp_path / "starter.zip")
    second = package.bundle_starter(source, bundle, tmp_path / "second.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive, zipfile.ZipFile(bundle) as bundled:
        expected = {*package.STARTER_FILES, *ASSETS, *bundled.namelist()}
        assert set(archive.namelist()) == {f"video2tenhou-0.1.0/{n}" for n in expected}
        mode = archive.getinfo("video2tenhou-0.1.0/start.sh").external_attr >> 16
        assert mode & 0o111 == 0o111


@pytest.mark.parametrize("missing", ["Start.cmd", *sorted(ASSETS)])
def test_starter_requires_launchers_and_built_frontend(
    models: Path, tmp_path: Path, missing: str
) -> None:
    """An unbuilt or incomplete application is never packaged."""
    source = source_archive(tmp_path / "source.tar.gz", missing)
    bundle = package.bundle_models(models, tmp_path / "models.zip")
    with pytest.raises(ValueError, match="launchers, project files or built UI"):
        package.bundle_starter(source, bundle, tmp_path / "starter.zip")
    assert not (tmp_path / "starter.zip").exists()


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("models/classifier/weights.pt", b"different weights"),
        ("models/classifier/weights.pt", None),
        ("models/unlisted.bin", b"extra"),
        (package.MANIFEST, None),
    ],
)
def test_starter_rejects_models_that_differ_from_their_manifest(
    models: Path, tmp_path: Path, name: str, content: bytes | None
) -> None:
    """Changed, missing or unlisted model files keep the previous starter."""
    source = source_archive(tmp_path / "source.tar.gz")
    bundle = package.bundle_models(models, tmp_path / "models.zip")
    rewrite(bundle, name, content)
    destination = tmp_path / "starter.zip"
    destination.write_bytes(b"previous")
    with pytest.raises(ValueError, match="differs from its manifest"):
        package.bundle_starter(source, bundle, destination)
    assert destination.read_bytes() == b"previous"


def test_interrupted_write_keeps_previous_archive_and_removes_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed publication retains the old archive and leaves no partial file."""
    destination = tmp_path / "starter.zip"
    destination.write_bytes(b"previous")

    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(package.zipfile.ZipFile, "writestr", interrupted)
    with pytest.raises(OSError, match="disk full"):
        package.write_zip(destination, {"a.txt": b"a"})
    assert destination.read_bytes() == b"previous"
    assert [path.name for path in tmp_path.iterdir()] == ["starter.zip"]
