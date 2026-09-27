"""Retention changes reuse raw sparse reads and isolate derived dense caches."""
import copy
import json
from types import SimpleNamespace

import numpy as np

from video2tenhou import read
from video2tenhou.calm import Interval
from video2tenhou.layout import Calibration
from video2tenhou.perception.evidence_policy import DEFAULT_POLICY, resolve_policy
from video2tenhou.train.data import CLASS_INDEX, CLASSES


def _policy(*, sparse_hand=.2, dense_hand=.2):
    return resolve_policy({'schema_version': 1,
                           'sparse': {'hand': sparse_hand, 'pond': .2, 'meld': .35},
                           'dense': {'hand': dense_hand, 'pond': .2, 'meld': .2}})


def _recognition(monkeypatch):
    calls = []
    posterior = np.zeros(len(CLASSES)).tolist()
    posterior[CLASS_INDEX['1s']] = 1.
    boxes = [dict(xyxy=[x, 0, x+38, 58], conf=confidence, sideways=False,
                  p=posterior, role='other') for x, confidence in [(0, .104), (40, .8)]]

    def recognize(items, det, clf):
        items = list(items)
        calls.extend((t, region) for t, region, _ in items)
        results = []
        for t, region, _ in items:
            raw = dict(t=t, region=region, size=[100, 60], boxes=copy.deepcopy(boxes))
            results.append(SimpleNamespace(region=region, t=t, to_dict=lambda raw=raw: copy.deepcopy(raw)))
        return results

    monkeypatch.setattr(read, 'source_identity', lambda *args, **kwargs: 'fixed-source')
    monkeypatch.setattr(read, 'read_regions', recognize)
    monkeypatch.setattr(read.video, 'sample', lambda *args, **kwargs: iter([(0., np.zeros((4, 4, 3), np.uint8))]))
    monkeypatch.setattr(read, 'region_upright', lambda frame, *args: (frame, None))
    return calls, boxes


def test_dense_policy_recomputes_structure_and_only_dense_changes_invalidate(tmp_path, monkeypatch):
    calls, raw_boxes = _recognition(monkeypatch)
    det = SimpleNamespace(id='unchanged-detector', evidence_policy=DEFAULT_POLICY)
    clf = SimpleNamespace(id='unchanged-classifier')
    cal = Calibration.load('pml')

    def acquire():
        return read.dense_reads('recording', cal, tmp_path, det, clf, 0., .2, ['hand:TL'])['hand:TL'][0]

    default = acquire()
    assert [box['conf'] for box in default['boxes']] == [.8]
    det.evidence_policy = _policy(dense_hand=.1)
    retained = acquire()
    assert [box['conf'] for box in retained['boxes']] == [.104, .8]
    assert all(box['role'] == 'tile' for box in retained['boxes'])
    assert all(box['p'][CLASS_INDEX['1s']] == 1. for box in retained['boxes'])
    assert all(box['role'] == 'other' for box in raw_boxes)  # raw evidence is untouched
    assert len(calls) == 2
    det.evidence_policy = _policy(sparse_hand=.15, dense_hand=.1)
    assert acquire() == retained  # sparse-only policy change reuses dense evidence
    assert len(calls) == 2
    assert len(list((tmp_path/'dense').glob('*.json'))) == 2


def test_policy_only_changes_leave_sparse_read_manifest_and_raw_bytes_reusable(tmp_path, monkeypatch):
    calls, _ = _recognition(monkeypatch)
    det = SimpleNamespace(id='unchanged-detector', evidence_policy=DEFAULT_POLICY)
    clf = SimpleNamespace(id='unchanged-classifier')
    cal = Calibration.load('pml')
    hands = [dict(hand=0, t_start=0., t_end=1.)]
    intervals = [Interval('hand:TL', 0., 0., 1, True, 0., 0.)]
    read.run_read('recording', cal, tmp_path, hands, intervals, det, clf)
    paths = [tmp_path/'reads/00/done.json', tmp_path/'reads/00/hand_TL.jsonl']
    before = [path.read_bytes() for path in paths]
    assert .104 in [box['conf'] for box in json.loads(before[1])['boxes']]
    det.evidence_policy = _policy(sparse_hand=.1, dense_hand=.1)
    calls.clear()
    _, touched = read.run_read('recording', cal, tmp_path, hands, intervals, det, clf)
    read.validate_read_cache('recording', cal, tmp_path, hands, det, clf)
    assert touched == set() and calls == []
    assert [path.read_bytes() for path in paths] == before
