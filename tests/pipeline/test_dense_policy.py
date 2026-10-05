# Copyright 2026 video2tenhou contributors
# SPDX-License-Identifier: Apache-2.0

"""Retention changes reuse raw sparse reads and isolate derived dense caches."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.builders import one_hot
from tests.recognition import RecognitionStub
from video2tenhou import read
from video2tenhou.calm import Interval
from video2tenhou.layout import Calibration
from video2tenhou.perception.evidence_policy import (
    DEFAULT_POLICY,
    EvidencePolicy,
    resolve_policy,
)
from video2tenhou.perception.reader import RegionClassifier, RegionDetector
from video2tenhou.perception.tiles import CLASS_INDEX, CLASSES


def _policy(*, sparse_hand: float = 0.2, dense_hand: float = 0.2) -> EvidencePolicy:
    return resolve_policy(
        {
            "schema_version": 1,
            "sparse": {"hand": sparse_hand, "pond": 0.2, "meld": 0.35},
            "dense": {"hand": dense_hand, "pond": 0.2, "meld": 0.2},
        }
    )


def _recognition(monkeypatch: pytest.MonkeyPatch) -> tuple:
    calls = []
    posterior = one_hot("1s")
    boxes = [
        {
            "xyxy": [x, 0, x + 38, 58],
            "conf": confidence,
            "sideways": False,
            "p": posterior,
            "role": "other",
        }
        for x, confidence in [(0, 0.104), (40, 0.8)]
    ]

    def recognize(
        items: read.CropBatch, det: RegionDetector, clf: RegionClassifier
    ) -> list[SimpleNamespace]:
        items = list(items)
        calls.extend((t, region) for t, region, _ in items)
        results = []
        for t, region, _ in items:
            raw = {
                "t": t,
                "region": region,
                "size": [100, 60],
                "boxes": copy.deepcopy(boxes),
            }
            results.append(
                SimpleNamespace(
                    region=region, t=t, to_dict=lambda raw=raw: copy.deepcopy(raw)
                )
            )
        return results

    monkeypatch.setattr(
        read, "source_identity", lambda *_unused_args, **_unused_kwargs: "fixed-source"
    )
    monkeypatch.setattr(read, "read_regions", recognize)

    def sample(
        *_unused_args: object, **_unused_kwargs: object
    ) -> Iterator[tuple[float, np.ndarray]]:
        yield 0.0, np.zeros((4, 4, 3), np.uint8)

    monkeypatch.setattr(read.video, "sample", sample)
    monkeypatch.setattr(
        read, "region_upright", lambda frame, *_unused_args: (frame, None)
    )
    return calls, boxes


def test_dense_policy_recomputes_structure_and_only_dense_changes_invalidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, raw_boxes = _recognition(monkeypatch)
    det = RecognitionStub(id="unchanged-detector", evidence_policy=DEFAULT_POLICY)
    clf = RecognitionStub(id="unchanged-classifier", classes=CLASSES, T=1.0)
    cal = Calibration.load("pml")

    def acquire() -> dict:
        return read.dense_reads(
            read.ReadContext("recording", cal, tmp_path, det, clf),
            0.0,
            0.2,
            ["hand:TL"],
        )["hand:TL"][0]

    default = acquire()
    assert [box["conf"] for box in default["boxes"]] == [0.8]
    det.evidence_policy = _policy(dense_hand=0.1)
    retained = acquire()
    assert [box["conf"] for box in retained["boxes"]] == [0.104, 0.8]
    assert all(box["role"] == "tile" for box in retained["boxes"])
    assert all(box["p"][CLASS_INDEX["1s"]] == 1.0 for box in retained["boxes"])
    assert all(box["role"] == "other" for box in raw_boxes)  # raw evidence is untouched
    assert len(calls) == 2
    det.evidence_policy = _policy(sparse_hand=0.15, dense_hand=0.1)
    assert acquire() == retained  # sparse-only policy change reuses dense evidence
    assert len(calls) == 2
    assert len(list((tmp_path / "dense").glob("*.json"))) == 2


def test_policy_only_changes_leave_sparse_read_manifest_and_raw_bytes_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls, _ = _recognition(monkeypatch)
    det = RecognitionStub(id="unchanged-detector", evidence_policy=DEFAULT_POLICY)
    clf = RecognitionStub(id="unchanged-classifier", classes=CLASSES, T=1.0)
    cal = Calibration.load("pml")
    hands = [{"hand": 0, "t_start": 0.0, "t_end": 1.0}]
    intervals = [Interval("hand:TL", 0.0, 0.0, 1, calm=True, motion=0.0, skin=0.0)]
    read.run_read(
        read.ReadContext("recording", cal, tmp_path, det, clf), hands, intervals
    )
    paths = [tmp_path / "reads/00/done.json", tmp_path / "reads/00/hand_TL.jsonl"]
    before = [path.read_bytes() for path in paths]
    assert 0.104 in [box["conf"] for box in json.loads(before[1])["boxes"]]
    det.evidence_policy = _policy(sparse_hand=0.1, dense_hand=0.1)
    calls.clear()
    _, touched = read.run_read(
        read.ReadContext("recording", cal, tmp_path, det, clf), hands, intervals
    )
    read.validate_read_cache(
        read.ReadContext("recording", cal, tmp_path, det, clf), hands
    )
    assert touched == set()
    assert calls == []
    assert [path.read_bytes() for path in paths] == before
