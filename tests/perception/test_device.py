# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""GPU presence alone must never send model loading to unsupported kernels."""

from typing import TYPE_CHECKING

import pytest
import torch

from video2tenhou.perception import device

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def clear_selection(monkeypatch: "pytest.MonkeyPatch") -> "Iterator[None]":
    """Reset cached device selection before and after each test."""
    device._select.cache_clear()
    monkeypatch.delenv("VIDEO2TENHOU_DEVICE", raising=False)
    yield
    device._select.cache_clear()


def test_unsupported_gpu_falls_back_once_for_both_models(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Verify unsupported gpu falls back once for both models."""
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def check(name: "str") -> None:
        calls.append(name)
        if name.startswith("cuda"):
            msg = "no kernel image is available for execution on the device"
            raise RuntimeError(msg)

    monkeypatch.setattr(device, "smoke_test", check)
    with pytest.warns(RuntimeWarning, match="Using CPU"):
        assert device.select_device() == "cpu"
    assert device.select_device() == "cpu"
    assert calls == ["cuda:0", "cpu"]


def test_working_gpu_is_selected(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Verify working gpu is selected."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []
    monkeypatch.setattr(device, "smoke_test", calls.append)
    assert device.select_device() == "cuda:0"
    assert calls == ["cuda:0"]


def test_cpu_override_never_initializes_cuda(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Verify cpu override never initializes cuda."""
    monkeypatch.setenv("VIDEO2TENHOU_DEVICE", "cpu")
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: pytest.fail("CPU override must skip CUDA")
    )
    calls = []
    monkeypatch.setattr(device, "smoke_test", calls.append)
    assert device.select_device() == "cpu"
    assert calls == ["cpu"]


def test_explicit_gpu_failure_is_not_silently_overridden(
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    """Verify explicit gpu failure is not silently overridden."""

    def fail(name: "str") -> None:
        assert name == "cuda:1"
        msg = "unavailable GPU"
        raise RuntimeError(msg)

    monkeypatch.setattr(device, "smoke_test", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        device.select_device(1)


def test_no_cuda_and_real_cpu_kernels(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Verify no cuda and real cpu kernels."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert device.select_device() == "cpu"


def test_broken_cpu_stack_is_not_hidden(monkeypatch: "pytest.MonkeyPatch") -> None:
    """Verify broken cpu stack is not hidden."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        device,
        "smoke_test",
        lambda _: (_ for _ in ()).throw(RuntimeError("broken torchvision")),
    )
    with pytest.raises(RuntimeError, match="broken torchvision"):
        device.select_device()
