"""Recorded evidence honors narrow human answers without inventing a full hand fact."""

from tests.paths import DATA
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from video2tenhou.engine.assemble import kyoku_from_decode
from video2tenhou.engine.decode import HandDecoder
from video2tenhou.engine.review import facts_for_hand
from video2tenhou.engine.solver import DrawEvidence, Facts, HandEvidence, HandModel, SeatTurn, Solution, WinSpec
from video2tenhou.engine.scoring import score_hand
from video2tenhou.record import HandResult
from video2tenhou.tenhou6 import replay_kyoku


def recorded_case(index):
    with gzip.open(DATA / "week11_adjudicated_models.json.gz", 'rt', encoding='utf-8') as stream:
        return next(case for case in json.load(stream) if case['hand'] == index)


def restore_model(data):
    turns = {seat: [SeatTurn(**{**turn, 'discard_p': np.array(turn['discard_p']) if turn['discard_p'] is not None else None})
                    for turn in rows] for seat, rows in data['turns'].items()}
    model = HandModel(data['dealer'], turns, data['indicators'], tsumo_winner=data['tsumo_winner'], ura=data['ura'])
    model.hand_ev = [HandEvidence(**{**row, 'e': np.array(row['e'])}) for row in data['hand_ev']]
    model.draw_ev = [DrawEvidence(**{**row, 'p': np.array(row['p'])}) for row in data['draw_ev']]
    facts = data['facts']
    model.facts = Facts(**{**facts, 'draws': {(row['seat'], row['j']): row['tile'] for row in facts['draws']}})
    for name in ('end_prior', 'repair', 'forbidden_hands', 'bound_hands', 'tenpai', 'result_constraints'):
        setattr(model, name, data[name])
    if data['win']:
        win = data['win']
        model.win = WinSpec(**{**win, 'ron_from': tuple(win['ron_from']) if win['ron_from'] is not None else None})
    return model


@pytest.mark.parametrize('index', [7, 24])
def test_confirmed_fields_survive_recorded_model_and_legal_export(index):
    case = recorded_case(index)
    model = restore_model(case['model'])
    facts = facts_for_hand(case['new_facts'], case['entry'])
    decoder = SimpleNamespace(facts=facts, problems=[])
    HandDecoder._apply_hand_facts(decoder, model)
    reference = case['reference']
    prior = Solution('optimal', reference['solver']['objective'], reference['haipai'],
                     {(key.split(':')[0], int(key.split(':')[1])): tile for key, tile in reference['draws'].items()}, {})
    sol = model.solve(time_limit=60, workers=2, margins=False, prior=prior)
    assert sol.ok
    for key, tile in model.facts.draws.items():
        assert sol.draws[key] == tile
    if index == 7:
        assert model.turns['S'][17].discard == '1m'
        assert model.turns['S'][17].discard_p is None
        assert ('S', 17) not in sol.discards
    else:
        assert sol.draws[('S', 0)] == '4z'
        assert sol.haipai['S'].count('2p') == sol.haipai['S'].count('4z') == 1
        assert 'S' not in model.facts.haipai  # only two counts were reviewed
        winner = model.win.seat
        concealed = list(sol.hands[(winner, model.win.j)])
        winning = sol.draws[(winner, model.win.j)]
        concealed.remove(winning)
        context = reference['score']['context']
        score = score_hand(concealed, winning, reference['score']['melds'], tsumo=True,
                           riichi=winner in reference['riichi'], seat=winner, round_wind='S',
                           dora=reference['dora'], ura=context.get('ura', reference['ura']),
                           ippatsu=context['ippatsu'], haitei=context['haitei'], rinshan=context['rinshan'])
        assert score.ok and (score.han, score.fu) == (6, 20)

    decoded = copy.deepcopy(reference)
    decoded['haipai'] = sol.haipai
    decoded['draws'] = {f'{seat}:{j}': tile for (seat, j), tile in sol.draws.items()}
    for turn in decoded['turns']:
        key = turn['seat'], turn['j']
        turn['draw'] = sol.draws.get(key)
        turn['draw2'] = sol.draws2.get(key)
        turn['discard'] = sol.discards.get(key, model._discard_of(key))
        turn['tsumogiri'] = (turn['draw2'] or turn['draw']) == turn['discard']
    kyoku, _ = kyoku_from_decode(decoded, case['entry'], HandResult(**case['result']))
    assert replay_kyoku(kyoku.dump()) == []
