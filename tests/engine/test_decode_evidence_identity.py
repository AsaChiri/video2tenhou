"""Review rebuilds cannot mix stale sparse evidence with current dense models."""
import json
from types import SimpleNamespace

import pytest

from video2tenhou import observe, read
from video2tenhou.engine import decode
from video2tenhou.layout import Calibration
from video2tenhou.perception import classifier, detector, evidence_policy



@pytest.fixture(autouse=True)
def _isolated_policy_metadata(monkeypatch):
    monkeypatch.setattr(evidence_policy, 'load_policy', lambda: evidence_policy.DEFAULT_POLICY)

def _inputs(tmp_path, monkeypatch):
    source = tmp_path / 'video.mp4'
    source.write_bytes(b'unchanged source')
    work = tmp_path / 'work'
    det = SimpleNamespace(id='detector:weights-and-runtime')
    clf = SimpleNamespace(id='classifier:weights-and-runtime', classes=['1m'], T=1.0)
    cal = Calibration.load('pml')
    hands = [dict(hand=i, t_start=i*10, t_end=(i+1)*10, game=0, site_index=i) for i in range(2)]
    for h in hands:
        directory = work / 'reads' / f"{h['hand']:02d}"
        directory.mkdir(parents=True)
        manifest = dict(detector=det.id, classifier=clf.id,
                        identity=read._read_identity(source, clf),
                        geometry={r: read.region_key(cal, r) for r in read.REGIONS},
                        window=[h['t_start'], h['t_end']+.5])
        (directory / 'done.json').write_text(json.dumps(manifest))
        (work / 'obs').mkdir(exist_ok=True)
        (work / 'obs' / f"{h['hand']:02d}.json").write_text('{}')
        (work / 'decode').mkdir(exist_ok=True)
        (work / 'decode' / f"{h['hand']:02d}.json").write_text('{"previous":true}')
    monkeypatch.setattr(detector, 'Detector', lambda: det)
    monkeypatch.setattr(classifier, 'Classifier', lambda: clf)
    monkeypatch.setattr(decode, 'load_facts', lambda path: [])
    return source, work, cal, hands, det, clf


@pytest.mark.parametrize('change', ['model_runtime', 'source_same_stat', 'geometry', 'legacy'])
def test_rebuild_rejects_stale_evidence_before_any_hand_is_overwritten(tmp_path, monkeypatch, change):
    import os

    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    read.validate_read_cache(source, cal, work, hands, det, clf)
    path = work / 'reads' / '01' / 'done.json'
    manifest = json.loads(path.read_text())
    if change == 'model_runtime':
        manifest['detector'] = 'same-weights-previous-runtime'
    elif change == 'geometry':
        manifest['geometry']['hand:BR'] = 'previous-crop'
    elif change == 'legacy':
        manifest.pop('identity')
    else:
        stat = source.stat()
        source.write_bytes(b'replacement data')  # same byte count and timestamp
        assert source.stat().st_size == stat.st_size
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(decode, 'decode_hand', lambda *a, **k: pytest.fail('stale batch reached decoding'))
    with pytest.raises(ValueError, match='Analyze recording'):
        decode.run_decode(work, hands, [], force=True, video_path=source, cal=cal)
    assert all((work / 'decode' / f"{h['hand']:02d}.json").read_text() == '{"previous":true}' for h in hands)


def test_read_identity_guard_checks_only_selected_hands(tmp_path, monkeypatch):
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    (work / 'reads' / '01' / 'done.json').unlink()
    read.validate_read_cache(source, cal, work, hands[:1], det, clf)
    with pytest.raises(ValueError, match=r'hand\(s\) 1'):
        read.validate_read_cache(source, cal, work, hands, det, clf)


def test_rebuild_rejects_unproven_observations_despite_matching_read_models(tmp_path, monkeypatch):
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    (work / 'calm.jsonl').write_text('')
    read.validate_read_cache(source, cal, work, hands, det, clf)
    monkeypatch.setattr(decode, 'decode_hand', lambda *a, **k: pytest.fail('unproven observation reached decoding'))
    with pytest.raises(ValueError, match='Analyze recording'):
        decode.run_decode(work, hands, [], force=True, video_path=source, cal=cal)
    assert all((work / 'decode' / f"{h['hand']:02d}.json").read_text() == '{"previous":true}' for h in hands)


def test_rebuild_accepts_current_observations_and_models(tmp_path, monkeypatch):
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    (work / 'calm.jsonl').write_text('')
    for h in hands:
        for region in read.REGIONS:
            (work / 'reads' / f"{h['hand']:02d}" / f"{region.replace(':', '_')}.jsonl").write_text('')
    observe.run_observe(work, hands, [], force=True)
    seen = []

    def rebuild(entry, obs, result, facts, **kwargs):
        assert kwargs['models'][:2] == (det, clf)
        seen.append(entry['hand'])
        return dict(stats=dict(turns=0, calls=0), dora=[],
                    solver=dict(status='ok', objective=0, low_margin=[]),
                    score=None, result=dict(han=0, fu=0), problems=[])

    monkeypatch.setattr(decode, 'decode_hand', rebuild)
    monkeypatch.setattr(decode, 'facts_for_hand', lambda *args: {})
    result = decode.run_decode(work, hands, [SimpleNamespace(hands=[None, None])],
                               force=True, video_path=source, cal=cal)
    assert seen == [0, 1] and len(result) == 2


def test_decode_resume_tracks_observation_and_provenance_after_publication(tmp_path, monkeypatch):
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    hands = hands[:1]
    monkeypatch.setattr(decode, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(decode, 'facts_for_hand', lambda *args: {})
    (work / 'calm.jsonl').write_text('')
    for region in read.REGIONS:
        (work / 'reads' / '00' / f"{region.replace(':', '_')}.jsonl").write_text('')
    observe.run_observe(work, hands, [], force=True)
    seen = []

    def rebuild(*args, **kwargs):
        seen.append(1)
        return dict(decoder_version=decode.DECODER_VERSION, stats=dict(turns=0, calls=0), dora=[],
                    solver=dict(status='ok', objective=0, low_margin=[]),
                    score=None, result=dict(han=0, fu=0), problems=[])

    monkeypatch.setattr(decode, 'decode_hand', rebuild)
    games = [SimpleNamespace(hands=[None])]
    first = decode.run_decode(work, hands, games)[0]  # offline reconstruction remains supported
    assert len(seen) == 1
    assert decode.run_decode(work, hands, games)[0] == first and len(seen) == 1

    # A completed observation publication followed by a restart returns no
    # changed hands. Identical votes still have different evidence provenance.
    manifest = work / 'reads' / '00' / 'done.json'
    manifest.write_text(manifest.read_text() + '\n')
    observe.run_observe(work, hands, [])
    assert observe.run_observe(work, hands, [])['changed_hands'] == []
    second = decode.run_decode(work, hands, games)[0]
    assert len(seen) == 2
    assert first['decode_inputs']['observation_sha256'] == second['decode_inputs']['observation_sha256']
    assert first['decode_inputs']['provenance_sha256'] != second['decode_inputs']['provenance_sha256']

    observation = work / 'obs' / '00.json'
    observation.write_text(observation.read_text() + '\n')
    decode.run_decode(work, hands, games)
    assert len(seen) == 3  # actual observation bytes independently invalidate
    (work / 'decode' / '00.json').write_text('{truncated')
    decode.run_decode(work, hands, games)
    assert len(seen) == 4  # incomplete output is a cache miss


def test_interrupted_decode_publication_preserves_complete_result(tmp_path, monkeypatch):
    path = tmp_path / '00.json'
    path.write_text('{"complete":true}')

    def interrupted_dump(value, stream, **kwargs):
        stream.write('{"partial":')
        raise OSError('interrupted')

    monkeypatch.setattr(decode.json, 'dump', interrupted_dump)
    with pytest.raises(OSError, match='interrupted'):
        decode._publish_decode(path, {})
    assert path.read_text() == '{"complete":true}'
    assert not list(tmp_path.glob('.decode-*'))


def _policy(sparse_hand=.2, dense_hand=.2):
    return evidence_policy.resolve_policy({'schema_version': 1,
        'sparse': {'hand': sparse_hand, 'pond': .2, 'meld': .35},
        'dense': {'hand': dense_hand, 'pond': .2, 'meld': .2}})


def test_review_rebuild_rejects_old_sparse_policy_before_any_write(tmp_path, monkeypatch):
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    (work/'calm.jsonl').write_text('')
    for hand in hands:
        for region in read.REGIONS:
            (work/'reads'/f"{hand['hand']:02d}"/f"{region.replace(':', '_')}.jsonl").write_text('')
    observe.run_observe(work, hands, [], force=True, policy=evidence_policy.DEFAULT_POLICY)
    det.evidence_policy = _policy(sparse_hand=.1)
    monkeypatch.setattr(evidence_policy, 'load_policy', lambda: det.evidence_policy)
    monkeypatch.setattr(decode, 'decode_hand', lambda *a, **k: pytest.fail('stale votes reached decoding'))
    # Raw recognition remains reusable; only the observation policy is stale.
    read.validate_read_cache(source, cal, work, hands, det, clf)
    with pytest.raises(ValueError, match='Analyze recording'):
        decode.run_decode(work, hands, [], force=True, video_path=source, cal=cal,
                          evidence_policy=det.evidence_policy)
    assert all((work/'decode'/f"{h['hand']:02d}.json").read_text() == '{"previous":true}' for h in hands)


@pytest.mark.parametrize('failure', ['invalid_metadata', 'policy_changed'])
def test_policy_failures_cannot_become_optional_model_fallback(tmp_path, monkeypatch, failure):
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    if failure == 'invalid_metadata':
        def invalid():
            raise ValueError('Invalid evidence metadata')
        monkeypatch.setattr(evidence_policy, 'load_policy', invalid)
    else:
        monkeypatch.setattr(evidence_policy, 'load_policy', lambda: _policy(dense_hand=.1))
    monkeypatch.setattr(detector, 'Detector', lambda: pytest.fail('policy preflight must precede model loading'))
    with pytest.raises(ValueError):
        decode.run_decode(work, hands, [], force=True, video_path=source, cal=cal,
                          evidence_policy=evidence_policy.DEFAULT_POLICY)
    assert all((work/'decode'/f"{h['hand']:02d}.json").read_text() == '{"previous":true}' for h in hands)


def test_dense_only_policy_invalidates_decode_but_preserves_observation_bytes(tmp_path, monkeypatch):
    source, work, cal, hands, det, clf = _inputs(tmp_path, monkeypatch)
    hands = hands[:1]
    monkeypatch.setattr(decode, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(decode, 'facts_for_hand', lambda *args: {})
    (work/'calm.jsonl').write_text('')
    for region in read.REGIONS:
        (work/'reads/00'/f"{region.replace(':', '_')}.jsonl").write_text('')
    observe.run_observe(work, hands, [], force=True, policy=evidence_policy.DEFAULT_POLICY)
    paths = [work/'obs/00.json', work/'obs/provenance/00.json']
    before = [path.read_bytes() for path in paths]
    calls = []

    def rebuild(*args, **kwargs):
        calls.append(1)
        return dict(decoder_version=decode.DECODER_VERSION, stats=dict(turns=0, calls=0), dora=[],
                    solver=dict(status='optimal', objective=0, low_margin=[]), score=None,
                    result=dict(han=0, fu=0), problems=[])

    monkeypatch.setattr(decode, 'decode_hand', rebuild)
    games = [SimpleNamespace(hands=[None])]
    first = decode.run_decode(work, hands, games, evidence_policy=evidence_policy.DEFAULT_POLICY)[0]
    policy = _policy(dense_hand=.1)
    observe.validate_observation_cache(work, hands, policy=policy)
    second = decode.run_decode(work, hands, games, evidence_policy=policy)[0]
    assert len(calls) == 2
    assert [path.read_bytes() for path in paths] == before
    assert first['decode_inputs']['dense_policy'] != second['decode_inputs']['dense_policy']
    assert first['decode_inputs']['observation_sha256'] == second['decode_inputs']['observation_sha256']
    # Omitting the explicit policy uses metadata only, even for nondefault policies.
    monkeypatch.setattr(evidence_policy, 'load_policy', lambda: policy)
    monkeypatch.setattr(detector, 'Detector', lambda: pytest.fail('valid bound cache must not load models'))
    assert decode.run_decode(work, hands, games, video_path=source)[0] == second
    assert len(calls) == 2