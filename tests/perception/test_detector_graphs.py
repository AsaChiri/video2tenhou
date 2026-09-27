"""Graph records must own shape-specific decode inputs, with safe eager limits."""
import gc
from types import SimpleNamespace
import weakref

import pytest

from video2tenhou.perception.graphs import enable_owned_graphs


class Grid:
    def __init__(self, size=1):
        self.size = size

    def numel(self):
        return self.size


def model_fixture():
    head = SimpleNamespace(anchors=Grid(0), strides=Grid(0), _grid=lambda: None)
    def capture(shape):
        head.anchors, head.strides = Grid(shape), Grid(shape)
        return SimpleNamespace()
    runner = SimpleNamespace(_graphs={}, _capture=capture, max_graphs=999)
    network = SimpleNamespace(named_modules=lambda: [("head", head)])
    model = SimpleNamespace(family="yolo9", model=network, _get_graph_runner=lambda: runner)
    return model, runner, head


def test_captured_record_retains_old_grids_until_cache_release():
    model, runner, head = model_fixture()
    other, other_runner, _ = model_fixture()
    untouched = other_runner._capture
    assert enable_owned_graphs(model, version="1.5.0", device="cuda:0")
    assert runner.max_graphs == 12 and other_runner._capture is untouched
    runner._graphs[1] = runner._capture(1)
    old_anchors, old_strides = weakref.ref(head.anchors), weakref.ref(head.strides)
    runner._graphs[2] = runner._capture(2)
    gc.collect()
    assert old_anchors() is not None and old_strides() is not None
    assert runner._graphs[1].retained_grid_storage[0] == (old_anchors(), old_strides())
    assert head.anchors is not old_anchors()
    del runner._graphs[1]
    gc.collect()
    assert old_anchors() is None and old_strides() is None
    assert runner._graphs[2].retained_grid_storage[0][0] is head.anchors


@pytest.mark.parametrize("version,device,family", [("1.6.0", "cuda:0", "yolo9"),
                                                   ("1.5.0", "cpu", "yolo9"),
                                                   ("1.5.0", "cuda:0", "yolox")])
def test_unsupported_runtime_stays_eager_without_mutating_runner(version, device, family):
    model, runner, _ = model_fixture()
    model.family = family
    original = runner._capture
    assert not enable_owned_graphs(model, version=version, device=device)
    assert runner._capture is original and runner.max_graphs == 999


def test_existing_or_uninitialized_graphs_are_not_accepted():
    model, runner, _ = model_fixture()
    runner._graphs[1] = object()
    original = runner._capture
    assert not enable_owned_graphs(model, version="1.5.0", device="cuda:0")
    assert runner._capture is original
    runner._graphs.clear()
    assert enable_owned_graphs(model, version="1.5.0", device="cuda:0")
    with pytest.raises(RuntimeError, match="no initialized"):
        runner._capture(0)
    assert runner._graphs == {}  # Upstream catches capture errors before caching.
