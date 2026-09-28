"""Real omitted kan-window evidence corrects a draw without a human constraint."""

from tests.paths import DATA
from tests.integration.helpers import restore_model
import copy
import gzip
import json
from types import SimpleNamespace

from video2tenhou.engine import dense
from video2tenhou.engine.assemble import kyoku_from_decode
from video2tenhou.engine.review import draws_to_reread
from video2tenhou.engine.solver import Solution
from video2tenhou.record import HandResult
from video2tenhou.tenhou6 import replay_kyoku


def test_recorded_open_kan_window_corrects_confident_inference(monkeypatch, tmp_path):
    with gzip.open(DATA / "week11_open_kan.json.gz", 'rt', encoding='utf-8') as stream:
        case = json.load(stream)
    model = restore_model(case['model'])
    reference = case['reference']
    prior = Solution('optimal', reference['solver']['objective'], reference['haipai'],
                     {(key.split(':')[0], int(key.split(':')[1])): value for key, value in reference['draws'].items()}, {})
    assert prior.draws[('N', 6)] == '5s'  # Recorded error, not a constraint.
    assert ('N', 6) not in model.facts.draws and 'N' not in model.facts.haipai
    selected = draws_to_reread(prior, model)
    assert selected == [('N', 6)]
    requests = []

    def recorded(video, cal, work, det, clf, lo, hi, regions, **kwargs):
        requests.append((lo, hi, regions))
        assert (lo, hi, regions) == (case['request']['lo'], case['request']['hi'], case['request']['regions'])
        return case['request']['data']

    monkeypatch.setattr(dense, 'dense_reads', recorded)
    melds_before = {seat: {int(j): n for j, n in rows.items()} for seat, rows in case['melds_before'].items()}
    before = len(model.hand_ev)
    dense.draws(selected, model, [SimpleNamespace(**t) for t in reference['turns']], melds_before,
                case['entry'], {}, (None, None, None, None), tmp_path, case['t0'], [])
    assert len(requests) == 1 and len(model.hand_ev) == before + 9
    solution = model.solve(margins=False, time_limit=60, workers=2, prior=prior)
    assert solution.ok and solution.optimal
    assert [solution.draws[('N', j)] for j in (4, 5, 6)] == ['2p', '1z', '1s']
    assert ('N', 6) not in model.facts.draws  # The answer remains external.
    decoded = copy.deepcopy(reference)
    decoded['haipai'] = solution.haipai
    decoded['draws'] = {f'{s}:{j}': tile for (s, j), tile in solution.draws.items()}
    for turn in decoded['turns']:
        key = turn['seat'], turn['j']
        turn['draw'] = solution.draws.get(key)
        turn['draw2'] = solution.draws2.get(key)
        turn['discard'] = solution.discards.get(key, model._discard_of(key))
        turn['tsumogiri'] = (turn['draw2'] or turn['draw']) == turn['discard']
    kyoku, _ = kyoku_from_decode(decoded, case['entry'], HandResult(**case['result']))
    assert replay_kyoku(kyoku.dump()) == []
