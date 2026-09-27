"""Open-kan acquisition fills missing evidence without overriding human answers."""
import numpy as np

from video2tenhou.engine.review import draws_to_reread, turn_key
from video2tenhou.engine.solver import DrawEvidence, HandModel, SeatTurn, Solution, TILES


def test_open_kan_replacement_acquisition_is_unique_and_respects_facts_and_evidence():
    turns = [SeatTurn(j, 'kan', '1s', 10*j, 10*j+5, kan='daiminkan', rinshan=True)
             for j in range(6)]
    turns[4].kan, turns[4].two_draws = 'ankan', True
    turns[5].kind = 'call'  # Winning daiminkan: no ordinary draw variable here.
    model = HandModel('E', {'E': [], 'S': [], 'W': [], 'N': turns}, [])
    model.facts.draws[('N', 1)] = '1s'
    model.draw_ev.append(DrawEvidence('N', 2, np.ones(len(TILES))/len(TILES), 1.))
    result = Solution('optimal', 0., {}, {}, {},
                      margins={('N', j): 0. if j == 3 else 10. for j in range(6)},
                      alternative_gaps={('N', j): 0. if j == 3 else 10. for j in range(6)})
    # Already-uncertain j3 is not duplicated; certified j0 still gets evidence.
    assert draws_to_reread(result, model) == [('N', 3), ('N', 0)]
    # API draw facts identify the sole open-kan replacement, not a self-kan's
    # second draw. No candidate tile is inserted into the acquisition policy.
    assert turn_key(model, {'seat': 'N', 'j': 0, 't': 5}, 1) == ('N', 0)
