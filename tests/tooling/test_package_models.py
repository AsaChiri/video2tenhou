# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Release bundles retain provenance, exclude local data and authenticate weights."""

import hashlib
import json
import zipfile
from typing import TYPE_CHECKING

import pytest

from tools import package_models as package

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def models(tmp_path: "Path") -> "Path":
    """Create the model files required by the release bundle contract."""
    root = tmp_path / "models"
    for name in package.FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}" if name.endswith(".json") else name.encode())
    (root / "training-private.json").write_text("not for distribution")
    meta = {
        "schema_version": 1,
        "backend": "libreyolo",
        "architecture": "yolo9-s",
        "classes": {"0": "face"},
        "weights_sha256": hashlib.sha256(
            (root / "detector/weights.pt").read_bytes()
        ).hexdigest(),
    }
    (root / "detector/meta.json").write_text(json.dumps(meta))
    (root / "detector/provenance.json").write_text('{"base_revision":"pinned"}')
    (root / "detector/LICENSE").write_text("upstream notice")
    return root


def test_bundle_is_reproducible_and_every_runtime_member_is_authenticated(
    models: "Path", tmp_path: "Path"
) -> None:
    """Verify bundle is reproducible and every runtime member is authenticated."""
    first = package.bundle(models, tmp_path / "first.zip")
    second = package.bundle(models, tmp_path / "second.zip")
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(archive.read("models/manifest.json"))
        assert set(archive.namelist()) == set(manifest["files"]) | {
            "models/manifest.json"
        }
        assert "models/detector/LICENSE" in manifest["files"]
        assert "models/detector/provenance.json" in manifest["files"]
        assert not any("private" in name for name in archive.namelist())
        for name, info in manifest["files"].items():
            content = archive.read(name)
            assert info == {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
    assert (
        first.with_suffix(".sha256").read_text().split()[0]
        == hashlib.sha256(first.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "defect",
    [
        "mismatch",
        "missing-license",
        "missing-provenance",
        "missing-weights",
        "missing-detector-metadata",
        "unsupported-backend",
        "wrong-class-map",
        "invalid-policy",
        "invalid-inference",
    ],
)
def test_incomplete_or_mismatched_release_preserves_existing_archive(
    models: "Path", tmp_path: "Path", defect: str
) -> None:
    """Verify incomplete or mismatched release preserves existing archive."""
    target = tmp_path / "release.zip"
    target.write_bytes(b"previous release")
    if defect == "mismatch":
        (models / "detector/weights.pt").write_bytes(b"different checkpoint")
    elif defect in (
        "unsupported-backend",
        "wrong-class-map",
        "invalid-policy",
        "invalid-inference",
    ):
        path = models / "detector/meta.json"
        meta = json.loads(path.read_text())
        changes = {
            "unsupported-backend": {"backend": "unsupported"},
            "wrong-class-map": {"classes": {"0": "face", "1": "back"}},
            "invalid-policy": {"evidence_policy": {"schema_version": 99}},
            "invalid-inference": {"inference": {"confidence": float("nan")}},
        }
        meta.update(changes[defect])
        path.write_text(json.dumps(meta))
    else:
        name = {
            "missing-license": "detector/LICENSE",
            "missing-provenance": "detector/provenance.json",
            "missing-weights": "classifier/weights.pt",
            "missing-detector-metadata": "detector/meta.json",
        }[defect]
        (models / name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        package.bundle(models, target)
    assert target.read_bytes() == b"previous release"
    assert not list(tmp_path.glob(".models-*"))
