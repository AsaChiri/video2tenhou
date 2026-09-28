"""Shared detector inference settings for loading, exporting and packaging."""
import math


def inference_settings(value: dict, **overrides) -> dict:
    """Resolve defaults and explicit overrides, then validate effective settings."""
    defaults = dict(imgsz=1024, confidence=.15, iou=.5, cuda_graph=False)
    if not isinstance(value, dict) or set(value) - defaults.keys():
        raise ValueError("Unsupported detector inference metadata")
    settings = {**defaults, **value, **{key: item for key, item in overrides.items() if item is not None}}
    if not isinstance(settings["cuda_graph"], bool):
        raise ValueError("Detector CUDA graph setting must be boolean")
    size = settings["imgsz"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("Detector image size must be a positive integer")
    if any(not isinstance(item, (int, float)) or isinstance(item, bool) or
           not math.isfinite(item) or not 0 < item <= 1 for item in (settings["confidence"], settings["iou"])):
        raise ValueError("Detector confidence and IoU must be finite numbers in (0, 1]")
    return settings
