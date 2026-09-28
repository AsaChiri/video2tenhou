"""GPU presence alone must never send model loading to unsupported kernels."""
import pytest
import torch

from video2tenhou.perception import device


@pytest.fixture(autouse=True)
def clear_selection(monkeypatch):
    device._select.cache_clear()
    monkeypatch.delenv("VIDEO2TENHOU_DEVICE", raising=False)
    yield
    device._select.cache_clear()


def test_unsupported_gpu_falls_back_once_for_both_models(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    def check(name):
        calls.append(name)
        if name.startswith("cuda"):
            raise RuntimeError("no kernel image is available for execution on the device")
    monkeypatch.setattr(device, "smoke_test", check)
    with pytest.warns(RuntimeWarning, match="Using CPU"):
        assert device.select_device() == "cpu"
    assert device.select_device() == "cpu"
    assert calls == ["cuda:0", "cpu"]


def test_working_gpu_is_selected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []
    monkeypatch.setattr(device, "smoke_test", calls.append)
    assert device.select_device() == "cuda:0"
    assert calls == ["cuda:0"]


def test_cpu_override_never_initializes_cuda(monkeypatch):
    monkeypatch.setenv("VIDEO2TENHOU_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CPU override must skip CUDA"))
    calls = []
    monkeypatch.setattr(device, "smoke_test", calls.append)
    assert device.select_device() == "cpu"
    assert calls == ["cpu"]


def test_explicit_gpu_failure_is_not_silently_overridden(monkeypatch):
    def fail(name):
        assert name == "cuda:1"
        raise RuntimeError("unavailable GPU")
    monkeypatch.setattr(device, "smoke_test", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        device.select_device(1)


def test_no_cuda_and_real_cpu_kernels(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device.select_device() == "cpu"


def test_broken_cpu_stack_is_not_hidden(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(device, "smoke_test", lambda _: (_ for _ in ()).throw(RuntimeError("broken torchvision")))
    with pytest.raises(RuntimeError, match="broken torchvision"):
        device.select_device()
