"""Exact detector inputs, bounded preparation and shared model ownership."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Timer
import numpy as np
import pytest
import torch
from libreyolo.preprocess.yolo9 import preprocess_image
from video2tenhou.perception import detector
from video2tenhou.perception.detector_flow import prepare_input, one_ahead


@pytest.mark.parametrize("shape", [(37, 71), (91, 19), (33, 33)])
def test_input_matches_reference_values_strides_and_sizes(shape):
    image = np.random.default_rng(781).integers(0, 256, (*shape, 3), dtype=np.uint8)
    # Also exercise an ordinary noncontiguous region slice.
    image = image[:, ::-1]
    tensor, original, target = prepare_input(image, 128)
    expected, _, size = preprocess_image(image, input_size=target, color_format="bgr")
    assert torch.equal(tensor, expected)
    assert tensor.dtype == expected.dtype and tensor.stride() == expected.stride()
    assert original == size == (shape[1], shape[0])
    assert tensor.device.type == "cpu"


def test_one_ahead_preserves_order_and_cannot_prepare_a_third_pending_input():
    second = Event()
    prepared = []
    def prepare(value):
        prepared.append(value)
        if value == 1: second.set()
        return value * 2
    def consume(value):
        if value == 0:
            assert second.wait(1)
            assert prepared == [0, 1]
        return value + 1
    assert one_ahead(range(4), prepare, consume) == [1, 3, 5, 7]


def test_consumer_failure_joins_running_preparation_and_does_not_consume_it():
    running, release, finished = Event(), Event(), Event()
    consumed = []
    def prepare(value):
        if value == 1:
            running.set()
            assert release.wait(1)
            finished.set()
        return value
    def consume(value):
        consumed.append(value)
        assert running.wait(1)
        Timer(.03, release.set).start()
        raise ValueError("consumer failed")
    with pytest.raises(ValueError, match="consumer failed"):
        one_ahead(range(8), prepare, consume)
    assert finished.is_set() and consumed == [0]


def test_producer_failure_propagates_without_skipping_the_input():
    consumed = []
    def prepare(value):
        if value == 1: raise ValueError("producer failed")
        return value
    with pytest.raises(ValueError, match="producer failed"):
        one_ahead(range(4), prepare, lambda value: consumed.append(value))
    assert consumed == [0]


def test_batch_lock_covers_all_consumption_and_blocks_a_preview(monkeypatch):
    entered, release, attempted = Event(), Event(), Event()
    values = []
    current = detector.Detector.__new__(detector.Detector)
    current._direct_preprocessing = True
    current._prediction_lock = Lock()
    current.imgsz = 128
    monkeypatch.setattr(detector, "prepare_input", lambda image, size: int(image[0, 0, 0]))
    def consume(value):
        values.append(value)
        if value == 0:
            entered.set()
            assert release.wait(2)
        return [value]
    current._consume_prepared = consume
    images = [np.full((8, 8, 3), n, np.uint8) for n in range(3)]
    def preview():
        attempted.set()
        return current.predict(images[2])
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(current.predict_batch, images[:2])
        try:
            assert entered.wait(1)
            second = pool.submit(preview)
            assert attempted.wait(1)
            assert values == [0]
        finally:
            release.set()
        assert first.result(timeout=2) == [[0], [1]]
        assert second.result(timeout=2) == [2]
    assert values == [0, 1, 2]


def test_unsupported_input_or_runtime_uses_public_path():
    current = detector.Detector.__new__(detector.Detector)
    current._prediction_lock = Lock()
    current._direct_preprocessing = True
    called = []
    current._public_predict = lambda image: called.append(image) or []
    unsupported = np.zeros((8, 8), np.uint8)
    assert current.predict_batch([unsupported]) == [[]]
    current._direct_preprocessing = False
    image = np.zeros((8, 8, 3), np.uint8)
    assert current.predict(image) == []
    assert called[0] is unsupported and called[1] is image
