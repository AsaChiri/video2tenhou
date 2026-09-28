"""Restore recorded solver inputs without changing their evidence or constraints."""

import numpy as np

from video2tenhou.engine.solver import DrawEvidence, Facts, HandEvidence, HandModel, SeatTurn, WinSpec


def restore_model(data):
    """Load recorded evidence in the current solver schema."""
    turns = {seat: [SeatTurn(**{**turn, 'discard_p': np.array(turn['discard_p']) if turn['discard_p'] is not None else None})
                    for turn in rows] for seat, rows in data['turns'].items()}
    model = HandModel(data['dealer'], turns, data['indicators'], tsumo_winner=data['tsumo_winner'], ura=data['ura'])
    model.hand_ev = [HandEvidence(**{**row, 'e': np.array(row['e'])}) for row in data['hand_ev']]
    model.draw_ev = [DrawEvidence(**{**row, 'p': np.array(row['p'])}) for row in data['draw_ev']]
    facts = data['facts']
    model.facts = Facts(**{**facts, 'draws': {(row['seat'], row['j']): row['tile'] for row in facts['draws']}})
    for name in ('repair', 'forbidden_hands', 'bound_hands', 'tenpai', 'result_constraints'):
        setattr(model, name, data[name])
    if data['win']:
        win = data['win']
        model.win = WinSpec(**{**win, 'ron_from': tuple(win['ron_from']) if win['ron_from'] is not None else None})
    return model
