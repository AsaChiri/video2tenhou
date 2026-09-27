"""Bounded CPU input preparation for the pinned LibreYOLO9 graph adapter.

No changes to forward math, postprocessing, padding, output order or thresholds.
The original visualization image is omitted because this adapter never saves it.
"""
from concurrent.futures import ThreadPoolExecutor
import math


def prepare_input(image, image_size):
    """Return a CPU NCHW tensor, original W/H and stride-aligned target H/W.

    Callers supply uint8 BGR HWC crops validated by the detector. Preserve the
    reference normalization order and noncontiguous CHW strides; do not replace
    them with a GPU preprocessing recipe or change rectangular padding.
    """
    import torch
    import cv2
    from libreyolo.preprocess.yolo9 import preprocess_numpy
    height, width = image.shape[:2]
    ratio = image_size / max(height, width)
    shape = (math.ceil(height * ratio / 32) * 32, math.ceil(width * ratio / 32) * 32)
    # These bytes match ImageLoader._from_numpy's BGR→RGB conversion. Keep
    # preprocess_numpy's uint8 resize/pad, float32 division and CHW strides.
    chw, _ = preprocess_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), shape)
    return torch.from_numpy(chw).unsqueeze(0), (width, height), shape


def infer_prepared(model, prepared, confidence, iou, cuda_graph):
    """Run graph dispatch/postprocessing and return CPU-native detection values.

    The caller holds the detector lock through this function and materializing
    its results; graph inputs/outputs and head grids are mutable instance state.
    """
    import torch
    tensor, original_size, shape = prepared
    tensor = tensor.to(model.device)
    with torch.no_grad(), model.cuda_graph_scope(cuda_graph):
        output = model._forward_graphed(tensor)
    return model._postprocess(output, confidence, iou, original_size,
                              max_det=300, ratio=1.0, classes=None, input_size=shape)


def one_ahead(items, prepare, consume):
    """Consume in order with one CPU preparation ahead; join on every exit.

    Only the caller consumes model state. Producer exceptions propagate, and
    consumer failure cancels queued work and waits for an active preparation.
    """
    iterator = iter(items)
    sentinel = object()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detector-input")
    def advance():
        item = next(iterator, sentinel)
        return sentinel if item is sentinel else prepare(item)
    future = pool.submit(advance)
    result = []
    try:
        while True:
            prepared = future.result()
            if prepared is sentinel:
                return result
            future = pool.submit(advance)
            result.append(consume(prepared))
    finally:
        future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
