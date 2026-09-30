"""Graph records must own shape-specific decode inputs, with safe eager limits."""

import gc
import weakref
from types import SimpleNamespace

import pytest
from libreyolo.models.yolo9.nn import DDetect

from video2tenhou.perception.graphs import enable_owned_graphs


class Grid:
    def __init__(self, size=1):
        self.size = size

    def numel(self):
        return self.size


def model_fixture():
    head = DDetect.__new__(DDetect)
    head.anchors, head.strides = Grid(0), Grid(0)

    def capture(shape):
        head.anchors, head.strides = Grid(shape), Grid(shape)
        return SimpleNamespace()

    runner = SimpleNamespace(_graphs={}, _capture=capture, max_graphs=999)
    network = SimpleNamespace(modules=lambda: [head])
    model = SimpleNamespace(
        family="yolo9", model=network, _get_graph_runner=lambda: runner
    )
    return model, runner, head


def test_captured_record_retains_old_grids_until_cache_release():
    model, runner, head = model_fixture()
    other, other_runner, _ = model_fixture()
    untouched = other_runner._capture
    assert enable_owned_graphs(model, device="cuda:0")
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


def test_cpu_does_not_create_cuda_graphs():
    model, runner, _ = model_fixture()
    original = runner._capture
    assert not enable_owned_graphs(model, device="cpu")
    assert runner._capture is original and runner.max_graphs == 999


def test_existing_or_uninitialized_graphs_are_not_accepted():
    model, runner, _ = model_fixture()
    runner._graphs[1] = object()
    original = runner._capture
    with pytest.raises(ValueError, match="before capturing"):
        enable_owned_graphs(model, device="cuda:0")
    assert runner._capture is original
    runner._graphs.clear()
    assert enable_owned_graphs(model, device="cuda:0")
    with pytest.raises(RuntimeError, match="no initialized"):
        runner._capture(0)
    assert runner._graphs == {}  # Upstream catches capture errors before caching.
