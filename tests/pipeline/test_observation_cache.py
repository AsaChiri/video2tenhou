"""Resume and interruption boundaries for real sparse readings and voted evidence."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video2tenhou import observe as stage
from video2tenhou.calm import Interval
from video2tenhou.train.data import CLASSES
from video2tenhou.perception.evidence_policy import DEFAULT_POLICY


def _write_reads(path, tile="1m"):
    posterior = [0.] * len(CLASSES)
    posterior[CLASSES.index(tile)] = 1.
    rows = [dict(t=t, region="pond:TL", size=[400, 700], boxes=[
        dict(xyxy=[10, 10, 50, 70], conf=.9, sideways=False, p=posterior)])
        for t in (0., 1., 2.)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture
def cache(tmp_path):
    hdir = tmp_path / "reads/00"
    hdir.mkdir(parents=True)
    reads = hdir / "pond_TL.jsonl"
    _write_reads(reads)
    done = hdir / "done.json"
    done.write_text('{"identity":"original"}')
    ivs = [Interval("pond:TL", 0., 2., 3, True, 0., 0.)]
    (tmp_path / "calm.jsonl").write_text(json.dumps(ivs[0].to_dict()) + "\n")
    hands = [dict(hand=0, t_start=0., t_end=2.)]
    result = SimpleNamespace(work=tmp_path, hands=hands, ivs=ivs, reads=reads, done=done,
                             output=tmp_path / "obs/00.json", proof=tmp_path / "obs/provenance/00.json")
    assert stage.run_observe(tmp_path, hands, ivs, log=lambda _: None) == {"observations": 1, "hands": 1, "changed_hands": [0]}
    stage.validate_observation_cache(tmp_path, hands)
    return result


def _run(cache, **options):
    options.setdefault('policy', getattr(cache, 'policy', None))
    return stage.run_observe(cache.work, cache.hands, cache.ivs, log=lambda _: None, **options)


def _votes(cache):
    return stage.load_obs(cache.work, 0)["pond:TL"]


def test_resume_reuses_intact_votes_but_detects_new_reads_with_empty_touched(cache, monkeypatch):
    original = cache.output.read_bytes()
    original_vote = stage.observe_interval
    def unexpected(*_):
        raise AssertionError("An intact cache should not vote again")
    monkeypatch.setattr(stage, "observe_interval", unexpected)
    assert _run(cache, touched=set()) == {"changed_hands": []}
    assert cache.output.read_bytes() == original
    monkeypatch.setattr(stage, "observe_interval", original_vote)
    _write_reads(cache.reads, "2p")  # Simulate prior process stopping after sparse-read publication.
    with pytest.raises(ValueError, match="Analyze recording"):
        stage.validate_observation_cache(cache.work, cache.hands)
    assert _run(cache, touched=set())["changed_hands"] == [0]
    assert _votes(cache)[0]["slots"][0]["tile"] == "2p"
    stage.validate_observation_cache(cache.work, cache.hands)


@pytest.mark.parametrize("change", ["interval", "partial", "settings", "completion", "window"])
def test_current_inputs_control_reuse_without_touched_hint(cache, monkeypatch, change):
    before = cache.proof.read_bytes()
    if change == "interval":
        cache.ivs = [replace(cache.ivs[0], t0=1.)]
    elif change == "partial":
        cache.ivs = [replace(cache.ivs[0], partial=True)]
    elif change == "settings":
        cache.policy = DEFAULT_POLICY.to_dict()
        cache.policy['sparse']['pond'] = .95
    elif change == "completion":
        cache.done.write_text('{"identity":"new recognition runtime"}')
    else:
        cache.hands = [dict(hand=0, t_start=.5, t_end=2.)]
    with pytest.raises(ValueError, match="stale"):
        stage.validate_observation_cache(cache.work, cache.hands, cache.ivs, policy=getattr(cache, 'policy', None))
    assert _run(cache, touched=set())["hands"] == 1
    assert cache.proof.read_bytes() != before
    stage.validate_observation_cache(cache.work, cache.hands, cache.ivs, policy=getattr(cache, 'policy', None))
    if change == "interval":
        assert _votes(cache)[0]["n_readings"] == 2 and _votes(cache)[0]["t0"] == 1.
    if change == "partial":
        assert _votes(cache)[0]["partial"]
    if change == "settings":
        assert _votes(cache)[0]["slots"] == []


def test_unrelated_calm_intervals_do_not_invalidate_hand(cache):
    cache.ivs += [Interval("pond:TL", 20., 21., 2, True, 0., 0.),
                  Interval("pond:TL", 0., 2., 3, False, 9., 0.)]
    assert _run(cache) == {"changed_hands": []}
    stage.validate_observation_cache(cache.work, cache.hands, cache.ivs, policy=getattr(cache, 'policy', None))


@pytest.mark.parametrize("damage", ["legacy", "corrupt_output", "missing_output", "corrupt_proof"])
def test_unproven_or_corrupted_output_is_recomputed(cache, damage):
    if damage == "legacy":
        cache.proof.unlink()
    elif damage == "corrupt_output":
        cache.output.write_text('{"pond:TL":')
    elif damage == "missing_output":
        cache.output.unlink()
    else:
        cache.proof.write_text("[]")
    with pytest.raises(ValueError, match="Analyze recording"):
        stage.validate_observation_cache(cache.work, cache.hands)
    assert _run(cache, touched=set())["hands"] == 1
    assert _votes(cache)[0]["slots"][0]["tile"] == "1m"
    stage.validate_observation_cache(cache.work, cache.hands)


@pytest.mark.parametrize("publication", ["output", "manifest"])
def test_failed_atomic_publication_never_authenticates_changed_evidence(cache, monkeypatch, publication):
    before_output, before_proof = cache.output.read_bytes(), cache.proof.read_bytes()
    _write_reads(cache.reads, "2p")
    target = cache.output if publication == "output" else cache.proof
    original_replace = Path.replace
    def fail_publication(path, destination):
        if Path(destination) == target:
            raise OSError("simulated publication failure")
        return original_replace(path, destination)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "replace", fail_publication)
        with pytest.raises(OSError, match="publication failure"):
            _run(cache)
    assert cache.proof.read_bytes() == before_proof
    if publication == "output":
        assert cache.output.read_bytes() == before_output
    else:
        assert _votes(cache)[0]["slots"][0]["tile"] == "2p"
    assert not list((cache.work / "obs").rglob(".*.tmp"))
    with pytest.raises(ValueError, match="stale"):
        stage.validate_observation_cache(cache.work, cache.hands)
    assert _run(cache, touched=set())["hands"] == 1
    stage.validate_observation_cache(cache.work, cache.hands)


def test_failed_forced_output_write_preserves_previous_usable_cache(cache, monkeypatch):
    original = cache.output.read_bytes(), cache.proof.read_bytes()
    real_dump = json.dump
    def fail_write(value, stream, **options):
        stream.write('{"interrupted":')
        raise OSError("disk full")
    with monkeypatch.context() as patch:
        patch.setattr(stage.json, "dump", fail_write)
        with pytest.raises(OSError, match="disk full"):
            _run(cache, force=True)
    assert stage.json.dump is real_dump
    assert (cache.output.read_bytes(), cache.proof.read_bytes()) == original
    assert not list((cache.work / "obs").rglob(".*.tmp"))
    stage.validate_observation_cache(cache.work, cache.hands)


def test_inputs_changing_during_vote_abort_before_publication(cache, monkeypatch):
    original = cache.output.read_bytes(), cache.proof.read_bytes()
    vote = stage.observe_interval
    def changed_input(*args, **kwargs):
        result = vote(*args, **kwargs)
        _write_reads(cache.reads, "2p")
        return result
    monkeypatch.setattr(stage, "observe_interval", changed_input)
    with pytest.raises(ValueError, match="inputs changed"):
        _run(cache, force=True)
    assert (cache.output.read_bytes(), cache.proof.read_bytes()) == original
    with pytest.raises(ValueError, match="stale"):
        stage.validate_observation_cache(cache.work, cache.hands)


def test_default_validation_rejects_unreadable_saved_calm_without_modifying_votes(cache):
    before = cache.output.read_bytes(), cache.proof.read_bytes()
    (cache.work / "calm.jsonl").write_text("not JSON")
    with pytest.raises(ValueError, match="cannot be verified"):
        stage.validate_observation_cache(cache.work, cache.hands)
    assert (cache.output.read_bytes(), cache.proof.read_bytes()) == before


def test_dense_only_policy_change_reuses_sparse_observations(cache):
    policy = DEFAULT_POLICY.to_dict()
    policy["dense"]["hand"] = .1
    before = cache.reads.read_bytes(), cache.output.read_bytes(), cache.proof.read_bytes()
    assert _run(cache, policy=policy) == {"changed_hands": []}
    stage.validate_observation_cache(cache.work, cache.hands, policy=policy)
    assert (cache.reads.read_bytes(), cache.output.read_bytes(), cache.proof.read_bytes()) == before


def test_sparse_policy_change_revotes_saved_bytes_and_roundtrips(cache):
    original = cache.reads.read_bytes(), cache.output.read_bytes()
    policy = DEFAULT_POLICY.to_dict()
    policy["sparse"]["pond"] = .95
    assert _run(cache, policy=policy)["changed_hands"] == [0]
    assert _votes(cache)[0]["slots"] == []
    assert cache.reads.read_bytes() == original[0]
    stage.validate_observation_cache(cache.work, cache.hands, policy=policy)
    with pytest.raises(ValueError, match="stale"):
        stage.validate_observation_cache(cache.work, cache.hands)
    assert _run(cache)["changed_hands"] == [0]
    assert cache.output.read_bytes() == original[1]
    assert cache.reads.read_bytes() == original[0]
