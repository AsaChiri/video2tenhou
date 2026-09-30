"""The studio verifies its device and models before any jobs can be started."""

from types import SimpleNamespace

import pytest

from video2tenhou import cli
from video2tenhou.paths import DATA_DIR
from video2tenhou.perception import classifier, detector, device
from video2tenhou.tool import server


@pytest.fixture(autouse=True)
def isolated_model_loaders(monkeypatch):
    monkeypatch.setenv("VIDEO2TENHOU_DEVICE", "auto")
    monkeypatch.setattr(detector, "Detector", object)
    monkeypatch.setattr(classifier, "Classifier", object)


def test_checked_device_is_inherited_by_jobs(monkeypatch, capsys):
    import os

    monkeypatch.setenv("VIDEO2TENHOU_DEVICE", "auto")
    monkeypatch.setattr(device, "select_device", lambda: "cpu")
    calls = []
    monkeypatch.setattr(
        server,
        "serve_workspace",
        lambda *a, **k: calls.append(os.environ["VIDEO2TENHOU_DEVICE"]),
    )
    cli.cmd_web(SimpleNamespace(data=str(DATA_DIR), port=8899, no_browser=True))
    assert calls == ["cpu"]
    assert "CPU (processing will be slower)" in capsys.readouterr().out


def test_unusable_explicit_device_stops_before_studio(monkeypatch):
    def fail():
        raise RuntimeError("GPU kernel check failed")

    monkeypatch.setattr(device, "select_device", fail)
    monkeypatch.setattr(
        server,
        "serve_workspace",
        lambda *a, **k: pytest.fail("server started before runtime check"),
    )
    with pytest.raises(RuntimeError, match="GPU kernel"):
        cli.cmd_web(SimpleNamespace(data=str(DATA_DIR), port=8899, no_browser=True))
