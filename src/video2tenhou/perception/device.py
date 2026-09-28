"""Test the installed inference stack before trusting CUDA availability."""
from __future__ import annotations

from functools import lru_cache
import os
import warnings


def smoke_test(device: str) -> None:
    """Exercise kernels used by both models, including torchvision's native ops."""
    import torch
    from torchvision.ops import nms

    with torch.inference_mode():
        x = torch.ones((1, 3, 16, 16), device=device)
        weight = torch.ones((4, 3, 3, 3), device=device)
        result = torch.nn.functional.conv2d(x, weight)
        matrix = torch.ones((8, 8), device=device)
        if not torch.isfinite(result).all().item() or (matrix @ matrix)[0, 0].item() != 8:
            raise RuntimeError("Inference kernel check returned invalid values")
        boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device=device)
        scores = torch.tensor([.9, .8], device=device)
        if nms(boxes, scores, .5).tolist() != [0]:
            raise RuntimeError("torchvision detection kernel check failed")
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)


@lru_cache(maxsize=16)
def _select(request: str) -> str:
    import torch

    if request != "auto":
        selected = f"cuda:{request}" if request.isdigit() else request
        smoke_test(selected)  # Explicit choices fail clearly, never silently change devices.
        return selected
    try:
        if torch.cuda.is_available():
            smoke_test("cuda:0")
            return "cuda:0"
    except (RuntimeError, AssertionError, OSError, ImportError) as exc:
        warnings.warn(
            f"GPU check failed: {exc}. Using CPU; processing will be slower. "
            "Update uv and your NVIDIA driver, then relaunch Start.cmd or start.sh "
            "to choose a compatible runtime.", RuntimeWarning, stacklevel=2)
    smoke_test("cpu")
    return "cpu"


def select_device(device: str | int | None = None) -> str:
    """Share one verified device across detector and classifier in this process."""
    request = str(device) if device is not None else os.environ.get("VIDEO2TENHOU_DEVICE", "auto")
    return _select(request)
