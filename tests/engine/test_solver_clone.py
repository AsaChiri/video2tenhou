"""Cloning must preserve CP-SAT search inputs, not merely equivalent constraints."""
import numpy as np
import pytest

from video2tenhou.engine import rules
from video2tenhou.engine.solver import (
    DrawEvidence, HandEvidence, HandModel, NT, SeatTurn, TI, _clone_draw_model,
)


def model_fixture():
    turns = {seat: [] for seat in rules.SEATS}
    turns['S'] = [SeatTurn(0, 'draw', '9m', 10, 20), SeatTurn(1, 'draw', '2z', 30, 40)]
    model = HandModel('E', turns, ['7z'])
    tiles = ['1m', '2m', '3m', '6p', '7p', '4s', '5s', '6s', '6s', '7s', '8s', '8s', '9m']
    model.facts.haipai['S'] = tiles
    posterior = np.zeros(NT)
    posterior[TI['2z']], posterior[TI['3z']] = 0.8, 0.2
    model.draw_ev = [DrawEvidence('S', 0, posterior, 1.0)]
    counts = np.zeros(NT)
    for tile in tiles:
        counts[TI[tile]] += 1
    model.hand_ev = [HandEvidence('S', -1, False, counts, 1.0, 0, 5)]
    return model


@pytest.mark.parametrize('repair', [False, True])
def test_draw_clones_preserve_full_proto_and_independent_hints(repair):
    model = model_fixture()
    model.repair = repair
    model.turns['S'][0].discard_p = np.ones(NT) / NT
    # These add constraints and variables after the exclusion insertion point.
    model.forbidden_hands = [('S', 0, ['1m'] * 13)]
    model.bound_hands = [('S', 1, model.facts.haipai['S'])]
    base, h0, draws, _ = model.build()
    untouched = str(base.proto)
    for key, tile in [(('S', 0), '2z'), (('S', 1), '3z')]:
        cloned, ch, cd = _clone_draw_model(base, h0, draws, (*key, tile))
        rebuilt, rh, rd, _ = model.build(forbid=(*key, tile))
        for m, hs, ds in [(cloned, ch, cd), (rebuilt, rh, rd)]:
            for seat, variables in hs.items():
                for v in variables:
                    m.AddHint(v, 0)
            for other_key, variables in ds.items():
                if other_key != key:
                    for v in variables:
                        m.AddHint(v, 0)
        assert str(cloned.proto) == str(rebuilt.proto)
        assert str(base.proto) == untouched


def test_prior_margins_require_unchanged_evidence_and_objective(monkeypatch):
    model = model_fixture()
    # Establish all starting hands once, so this test isolates draw alternatives.
    first = model.solve(margins=False, workers=1)
    assert first.ok
    model.facts.haipai.update(first.haipai)
    calls = []

    def resolve(*args, **kwargs):
        calls.append(kwargs['watch'])
        return 2.0, '3z', 4.0

    monkeypatch.setattr(model, '_resolve', resolve)
    first = model.solve(workers=1)
    assert len(calls) == 2
    calls.clear()
    same = model.solve(workers=1, prior=first)
    assert not calls and same.margins == first.margins
    assert same.alternative_gaps == first.alternative_gaps
    # A changed baseline objective makes the old separation unsafe too.
    same.objective += 1
    model.solve(workers=1, prior=same)
    assert len(calls) == 2
    calls.clear()
    model.draw_ev[0].weight = 0.9
    changed = model.solve(workers=1, prior=first)
    assert changed.draws == first.draws
    assert changed.model_fingerprint != first.model_fingerprint
    assert len(calls) == 2


def test_prior_hints_allow_new_facts_and_valid_cloned_alternatives():
    model = model_fixture()
    first = model.solve(workers=1, margins=False)
    model.facts.haipai.update(first.haipai)
    first = model.solve(workers=1)
    assert first.draws[('S', 0)] == '2z'
    # A newly reviewed tile must override the old hint. The other draw still
    # computes an actual cloned counterfactual, which rejects duplicate hints.
    model.facts.draws[('S', 0)] = '3z'
    updated = model.solve(workers=1, prior=first)
    cold = model.solve(workers=1)
    assert updated.status == cold.status == 'optimal'
    assert updated.draws[('S', 0)] == '3z'
    assert updated.objective == cold.objective
    assert updated.model_fingerprint == cold.model_fingerprint
    assert updated.margins[('S', 1)] == cold.margins[('S', 1)] > 0


def test_prior_haipai_certificates_require_same_model_objective_and_multiset(monkeypatch):
    model = model_fixture()
    calls = []

    def resolve(*args, **kwargs):
        if 'forbid_haipai' in kwargs:
            calls.append(kwargs['forbid_haipai'][0])
        return 2., None, 3.

    monkeypatch.setattr(model, '_resolve', resolve)
    first = model.solve(workers=1)
    assert set(calls) == {'E', 'W', 'N'}
    calls.clear()
    again = model.solve(workers=1, prior=first)
    assert not calls
    assert again.haipai_margins == first.haipai_margins
    # Same tiles do not excuse changed evidence or an inconsistent objective.
    first.objective += .01
    model.solve(workers=1, prior=first)
    assert set(calls) == {'E', 'W', 'N'}


def test_proof_stopping_keeps_equal_cost_alternative_uncertain():
    from collections import Counter
    from video2tenhou.engine.solver import TILES
    model = model_fixture()
    model.draw_ev[0].p[TI['2z']] = model.draw_ev[0].p[TI['3z']] = .5
    sol = model.solve(workers=1, margins=False)
    baseline = model.build()[:3]
    hint = ({s: [Counter(tiles)[t] for t in TILES] for s, tiles in sol.haipai.items()},
            {key: [int(t == tile) for t in TILES] for key, tile in sol.draws.items()})
    key = ('S', 0)
    results = [model._resolve(hint, sol.objective, 1, baseline=baseline, watch=key,
                              forbid=(*key, sol.draws[key]), stop_when_certified=stop)
               for stop in (False, True)]
    assert results[0] == results[1]
    assert results[0][0] == results[0][2] == 0
