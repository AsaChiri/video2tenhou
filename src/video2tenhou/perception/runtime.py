# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Numerical execution settings that distinguish cached model evidence."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def runtime_signature(device: str | int) -> dict:
    """Describe the effective device and inference math configuration.

    These settings can change floating-point results without changing model
    bytes. Call after selecting the inference device and configuring thread/math
    settings; the returned JSON-compatible mapping belongs in model cache IDs.
    """
    name = f"cuda:{device}" if str(device).isdigit() else str(device)
    selected = torch.device(name)
    gpu = None
    if selected.type == "cuda" and torch.cuda.is_available():
        index = (
            selected.index
            if selected.index is not None
            else torch.cuda.current_device()
        )
        name = f"cuda:{index}"
        gpu = {
            "name": torch.cuda.get_device_name(index),
            "capability": list(torch.cuda.get_device_capability(index)),
        }
    return {
        "device": name,
        "gpu": gpu,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "opencv_threads": cv2.getNumThreads(),
    }
