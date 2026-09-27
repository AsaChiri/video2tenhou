"""A later partial camera view must not erase an independently witnessed call."""

from tests.paths import DATA
import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from video2tenhou.engine.calls import CallAnchor
from video2tenhou.engine.dense import place_taken
from video2tenhou.engine.melds import Call, track_melds
from video2tenhou.engine.ponds import tail_runs, track_pond
from video2tenhou.train.data import CLASS_INDEX, CLASSES


def recorded():
    return json.loads(gzip.decompress((DATA / "week11_contradicted_chi.json.gz").read_bytes()))


def hypothesis(data):
    calls=track_melds('E',data['meld_observations'],include_contradicted=True)
    return next(c for c in calls if c.type=='chi' and c.t_first==9818)


def source_logs(data):
    logs={s:[] for s in 'ESWN'}
    logs['N']=track_pond(data['pond_observations'])
    runs=tail_runs(logs['N'],data['source_window']['readings'])
    source=next(r for r in runs if r['gone'] and r['tile']=='0p' and 9810<=r['t_first']<=9812)
    slot=place_taken('N',source,logs)
    return logs,slot


def test_recorded_chi_requires_independent_removed_discard():
    data=recorded()
    ordinary=track_melds('E',data['meld_observations'])
    assert not any(c.type=='chi' and c.t_first==9818 for c in ordinary)
    ev=hypothesis(data)
    assert ev.contradicted and ev.seen==3 and ev.absent==5
    assert sorted(ev.tiles)==['0p','3p','4p']
    assert ev.to_dict()['contradicted'] is True
    assert CallAnchor({s:[] for s in 'ESWN'},{},None,None,0,[]).anchor([ev],9500,9970,None)==[]
    logs,slot=source_logs(data)
    anchor=CallAnchor(logs,{},None,None,0,[])
    calls=anchor.anchor([ev,copy.deepcopy(ev)],9500,9970,None)
    assert len(calls)==1  # the competing hypothesis cannot claim the same tile
    call=calls[0]
    assert call.type=='chi' and call.source=='kamicha' and call.called_tile=='0p'
    assert sorted(call.tiles)==['0p','3p','4p']
    assert call.anchor=='discard' and call.contradicted and call.absent==5
    assert id(slot) in anchor.claimed


@pytest.mark.parametrize('mismatch',['not_removed','wrong_source','wrong_tile'])
def test_contradicted_camera_event_cannot_claim_unrelated_evidence(mismatch):
    data=recorded();ev=hypothesis(data);logs,slot=source_logs(data)
    if mismatch=='not_removed':slot.t_removed=None
    elif mismatch=='wrong_source':logs['W']=[slot];logs['N']=[]
    else:
        slot.p=np.eye(len(CLASSES))[CLASS_INDEX['7p']]
    assert CallAnchor(logs,{},None,None,0,[]).anchor([ev],9500,9970,None)==[]


def test_contradicted_external_kan_has_no_self_kan_fallback():
    p=np.eye(len(CLASSES))[CLASS_INDEX['2m']]
    ev=Call('E',50,(45,50),'kan',['2m']*4,0,'kamicha','2m',[p]*4,1,
            seen=2,absent=3,contradicted=True)
    anchor=CallAnchor({s:[] for s in 'ESWN'},{},None,None,0,[])
    assert anchor.anchor([ev],10,90,None)==[]
