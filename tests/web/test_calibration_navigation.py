"""Exercise the actual calibration editor across save/check/navigation boundaries."""

from tests.paths import ROOT
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


PAGE = ROOT / "src/video2tenhou/tool/static/index.html"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")


def run_calibration(tmp_path, scenario):
    script = re.search(r"<script>([\s\S]*)</script>", PAGE.read_text(encoding="utf-8"))[1]
    script = script.split("// ------------------------------------------------------------------ calibrate (stage 0)")[1]
    script = script.split("// ------------------------------------------------------------------ boot")[0]
    harness = """
const nodes=new Map(), handlers={}, messages=[], calls=[];
const $=selector=>{
  if(!nodes.has(selector))nodes.set(selector,{value:'',classList:{},
    addEventListener(type,fn){handlers[type]=fn;},getBoundingClientRect(){return {left:0,top:0,width:1000};}});
  return nodes.get(selector);
};
const EMBEDDED=true, location={origin:'http://localhost:8876'};
const window={parent:{postMessage(message,origin){messages.push({...message,origin});}},
  addEventListener(type,fn){handlers[type]=fn;}};
let saved={frame:[1000,1000],regions:{
  'hand:TL':{movable:true,rect:[100,100,100,100],quad:[[100,100],[200,100],[200,200],[100,200]],roll:3},
  'meld:TL':{movable:true,rect:[300,300,100,100],quad:[[300,300],[400,300],[400,400],[300,400]]}
},checks:{'hand:TL':{level:'ok'}}};
let api=async(path,body)=>{calls.push({path,body});return structuredClone(saved);};
"""
    setup = """
renderCalib=()=>{};
calibPoll=async key=>{calls.push({poll:key});};
C.data=structuredClone(saved); C.plate={};
function drag(){handlers.mousedown({clientX:150,clientY:150});handlers.mousemove({clientX:160,clientY:150});handlers.mouseup();}
"""
    path = tmp_path / "calibration.mjs"
    path.write_text(harness + script + setup + scenario, encoding="utf-8")
    result = subprocess.run(["node", str(path)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_check_saves_only_dragged_region_before_validation(tmp_path):
    result = run_calibration(tmp_path, """
drag(); await calibLoad();
const draft={rect:C.data.regions['hand:TL'].rect,dirty:[...C.dirty],calls:calls.length,check:C.data.checks['hand:TL']};
await calibCheck();
console.log(JSON.stringify({draft,calls,dirty:[...C.dirty],messages}));
""")
    assert result["draft"] == {"rect": [110, 100, 100, 100], "dirty": ["hand:TL"], "calls": 0}
    assert result["calls"] == [
        {"path": "api/calib", "body": {"hand": {"TL": {"rect": [110, 100, 100, 100], "roll": 3}}}},
        {"path": "api/calib"}, {"path": "api/calib/check", "body": {}}, {"poll": "check"},
    ]
    assert not result["dirty"]
    assert result["messages"][0]["dirty"] is True
    assert result["messages"][-1]["dirty"] is False


def test_unchanged_check_does_not_persist_geometry(tmp_path):
    result = run_calibration(tmp_path, "await calibCheck(); console.log(JSON.stringify(calls));")
    assert result == [{"path": "api/calib/check", "body": {}}, {"poll": "check"}]


def test_overhead_and_pending_regions_are_saved_atomically(tmp_path):
    result = run_calibration(tmp_path, """
drag(); $('#ohx').value='500'; $('#ohy').value='450'; $('#oha').value='1'; $('#ohs').value='1.2';
await applyOh(); console.log(JSON.stringify(calls));
""")
    assert result == [{"path": "api/calib", "body": {
        "hand": {"TL": {"rect": [110, 100, 100, 100], "roll": 3}},
        "overhead": {"center": [500, 450], "angle": 1, "scale": 1.2},
    }}, {"path": "api/calib"}]


def test_failed_save_preserves_draft_and_blocks_check_and_remeasure(tmp_path):
    result = run_calibration(tmp_path, """
drag(); await calibAuto(); const remeasureCalls=calls.length;
api=async(path,body)=>{calls.push({path,body});throw Error('Cannot save');};
await calibCheck();
console.log(JSON.stringify({remeasureCalls,calls,dirty:[...C.dirty],rect:C.data.regions['hand:TL'].rect,
  busy:C.busy,status:$('#cstatus').textContent,lastMessage:messages.at(-1)}));
""")
    assert result["remeasureCalls"] == 0
    assert len(result["calls"]) == 1 and result["calls"][0]["path"] == "api/calib"
    assert result["dirty"] == ["hand:TL"] and result["rect"] == [110, 100, 100, 100]
    assert result["busy"] is False and result["status"] == "Cannot save"
    assert result["lastMessage"]["dirty"] is True


def test_discard_reloads_without_writing_and_releases_navigation(tmp_path):
    result = run_calibration(tmp_path, """
drag(); await calibDiscard();
console.log(JSON.stringify({calls,dirty:[...C.dirty],rect:C.data.regions['hand:TL'].rect,lastMessage:messages.at(-1)}));
""")
    assert result["calls"] == [{"path": "api/calib"}]
    assert result["dirty"] == [] and result["rect"] == [100, 100, 100, 100]
    assert result["lastMessage"]["dirty"] is False


def test_delayed_calibration_load_cannot_replace_a_new_drag(tmp_path):
    result = run_calibration(tmp_path, """
let finish;
api=()=>new Promise(resolve=>finish=resolve);
const pending=calibLoad(); drag(); finish(structuredClone(saved)); await pending;
console.log(JSON.stringify({dirty:[...C.dirty],rect:C.data.regions['hand:TL'].rect}));
""")
    assert result == {"dirty": ["hand:TL"], "rect": [110, 100, 100, 100]}
