# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Refinement must not relabel pseudo data as reviewed human supervision."""

import json
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from video2tenhou.train import train_libreyolo

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(("origin", "reviewed"), [("pseudo", False), ("human", False)])
def test_human_refinement_rejects_unreviewed_inputs_before_model_loading(
    *, tmp_path: "Path", monkeypatch: "pytest.MonkeyPatch", origin: str, reviewed: bool
) -> None:
    """Verify human refinement rejects unreviewed inputs before model loading."""
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("nc: 1\n")
    (dataset / "manifest.jsonl").write_text(
        json.dumps({"origin": origin, "reviewed": reviewed, "boxes": 1}) + "\n"
    )
    monkeypatch.setattr(train_libreyolo.metadata, "version", lambda _: "1.5.0")
    monkeypatch.setitem(
        sys.modules,
        "libreyolo",
        SimpleNamespace(
            LibreYOLO=lambda *_unused_a, **_unused_k: pytest.fail(
                "Invalid dataset loaded a model"
            )
        ),
    )
    with pytest.raises(ValueError, match="only reviewed human"):
        train_libreyolo.main(
            [
                "--data",
                str(dataset / "data.yaml"),
                "--base",
                str(tmp_path / "missing.pt"),
                "--out",
                str(tmp_path / "run"),
                "--refine-human",
            ]
        )


@pytest.mark.parametrize(
    "worker_case",
    [("Windows", [], 0), ("Linux", [], 4), ("Windows", ["--workers", "2"], 2)],
)
@pytest.mark.parametrize("version", ["1.5.0", "1.5.1"])
def test_training_forwards_portable_worker_default_and_explicit_override(
    tmp_path: "Path",
    monkeypatch: "pytest.MonkeyPatch",
    worker_case: tuple[str, list[str], int],
    version: str,
) -> None:
    """Verify training forwards portable worker default and explicit override."""
    system, override, expected = worker_case
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("nc: 1\n")
    (dataset / "manifest.jsonl").write_text(
        json.dumps({"origin": "human", "reviewed": True, "boxes": 1}) + "\n"
    )
    (dataset / "provenance.json").write_text("{}")
    base = tmp_path / "base.pt"
    base.write_bytes(b"fake model, never loaded")
    captured = {}

    class Model:
        def __init__(self, *_args: "object", **_kwargs: "object") -> None:
            self.names = {0: "face"}

        def train(self, **kwargs: "object") -> dict:
            captured.update(kwargs)
            return {}

    monkeypatch.setattr(train_libreyolo.platform, "system", lambda: system)
    monkeypatch.setattr(train_libreyolo.metadata, "version", lambda _: version)
    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=Model))
    train_libreyolo.main(
        [
            "--data",
            str(dataset / "data.yaml"),
            "--base",
            str(base),
            "--out",
            str(tmp_path / "run"),
            "--refine-human",
            *override,
        ]
    )
    assert captured["workers"] == expected
    report = json.loads((tmp_path / "run" / "provenance.json").read_text())
    assert report["complete"]
    assert report["config"]["workers"] == expected
    assert report["environment"]["libreyolo"] == version
