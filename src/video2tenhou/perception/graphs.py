"""Bounded LibreYOLO9 CUDA graphs with ownership of external decode grids.

LibreYOLO 1.5.0 warms the detection head before capture. Its anchor/stride grids
are therefore external graph inputs, but the head replaces them on shape changes.
Each captured record must retain its own tensors until that graph is released.
This adapter is instance-local and changes no tensor values or forward math.
"""

GRAPH_POLICY = "libreyolo-1.5.0-yolo9-owned-grids-v2-cap12"


def enable_owned_graphs(model, *, device: str) -> bool:
    """Install the verified grid-lifetime fix on a fresh supported CUDA model.

    CPU inference needs no CUDA graph. Invalid graph state fails rather than
    silently changing the requested execution mode. Capture failures propagate to LibreYOLO's
    bounded eager fallback; an unowned captured record is never returned. The
    runner caps its cache at twelve shape/dtype/device keys and uses
    eager inference for further shapes. Existing graph caches are not modified.
    """
    if not str(device).startswith("cuda"):
        return False
    from libreyolo.models.yolo9.nn import DDetect

    heads = [module for module in model.model.modules() if isinstance(module, DDetect)]
    if not heads:
        raise ValueError("YOLO9 graph model has no detection head")
    runner = model._get_graph_runner()
    capture = runner._capture
    if runner._graphs:
        raise ValueError(
            "Grid ownership must be installed before capturing CUDA graphs"
        )

    def capture_with_owned_grids(tensor):
        captured = capture(tensor)
        grids = tuple((head.anchors, head.strides) for head in heads)
        if any(
            not anchors.numel() or not strides.numel() for anchors, strides in grids
        ):
            raise RuntimeError("YOLO9 capture has no initialized decode grids")
        # Capture's warmup allocated these outside the graph's memory pool.
        # The record shares their lifetime even after the head changes shape.
        captured.retained_grid_storage = grids
        return captured

    # A standard layout has twelve camera regions; distinct crop proportions
    # can exceed the upstream eight-key limit even without changing the layout.
    runner.max_graphs = 12
    runner._capture = capture_with_owned_grids
    return True
