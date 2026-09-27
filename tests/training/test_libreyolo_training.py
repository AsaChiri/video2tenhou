"""Refinement must not relabel pseudo data as reviewed human supervision."""
import json
import sys
from types import SimpleNamespace

import pytest

from video2tenhou.train import train_libreyolo


@pytest.mark.parametrize("origin,reviewed", [("pseudo", False), ("human", False)])
def test_human_refinement_rejects_unreviewed_inputs_before_model_loading(tmp_path, monkeypatch, origin, reviewed):
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("nc: 1\n")
    (dataset / "manifest.jsonl").write_text(json.dumps(dict(origin=origin, reviewed=reviewed, boxes=1))+"\n")
    monkeypatch.setattr(train_libreyolo.metadata, "version", lambda _: "1.5.0")
    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=lambda *a, **k: pytest.fail("Invalid dataset loaded a model")))
    with pytest.raises(ValueError, match="only reviewed human"):
        train_libreyolo.main(["--data", str(dataset / "data.yaml"), "--base", str(tmp_path / "missing.pt"),
                             "--out", str(tmp_path / "run"), "--refine-human"])


@pytest.mark.parametrize("system,override,expected", [("Windows", [], 0), ("Linux", [], 4), ("Windows", ["--workers", "2"], 2)])
def test_training_forwards_portable_worker_default_and_explicit_override(tmp_path, monkeypatch, system, override, expected):
    dataset = tmp_path / "data"
    dataset.mkdir()
    (dataset / "data.yaml").write_text("nc: 1\n")
    (dataset / "manifest.jsonl").write_text(json.dumps(dict(origin="human", reviewed=True, boxes=1))+"\n")
    (dataset / "provenance.json").write_text("{}")
    base = tmp_path / "base.pt"
    base.write_bytes(b"fake model, never loaded")
    captured = {}

    class Model:
        names = {0: "face"}

        def __init__(self, *args, **kwargs):
            pass

        def train(self, **kwargs):
            captured.update(kwargs)
            return {}

    monkeypatch.setattr(train_libreyolo.platform, "system", lambda: system)
    monkeypatch.setattr(train_libreyolo.metadata, "version", lambda _: "1.5.0")
    monkeypatch.setitem(sys.modules, "libreyolo", SimpleNamespace(LibreYOLO=Model))
    train_libreyolo.main(["--data", str(dataset / "data.yaml"), "--base", str(base),
                         "--out", str(tmp_path / "run"), "--refine-human", *override])
    assert captured["workers"] == expected
    report = json.loads((tmp_path / "run" / "provenance.json").read_text())
    assert report["complete"] and report["config"]["workers"] == expected
