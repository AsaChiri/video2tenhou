"""Run the studio's actual event handlers across asynchronous project changes.

The DOM model tracks element replacement and iframe navigation, so a redraw
cannot silently pass as preservation of an edited form or playing review.
"""

from tests.paths import ROOT
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest


PAGE = ROOT / "src/video2tenhou/tool/static/studio.html"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")


def run_studio(tmp_path, scenario):
    html = PAGE.read_text(encoding="utf-8")
    script = re.search(r"<script>([\s\S]*)</script>", html)[1]
    # Startup polling is explicitly driven by each scenario; handlers are unmodified.
    script = script.rsplit("refresh(true); setInterval", 1)[0]
    ids = re.findall(r'\bid="([^"]+)"', html.split("<script>")[0])
    harness = """
const nodes = new Map();
class Element {
  constructor(id) { this.id=id; this.value=''; this.attrs={}; this.style={}; this.dataset={};
    this.writes=0; this.navigations=0; this.mediaTime=0;
    this.classList={add(){},remove(){},toggle(){}}; }
  get innerHTML() { return this.html||''; }
  set innerHTML(html) {
    this.html=html; this.writes++;
    for(const match of html.matchAll(/<[^>]+\\bid="([^"]+)"[^>]*>/g)) {
      const child=new Element(match[1]);
      child.value=/\\bvalue="([^"]*)"/.exec(match[0])?.[1]||'';
      child.open=/\\bopen(?:\\s|>)/.test(match[0]);
      nodes.set(child.id,child);
    }
  }
  set src(value) { this.attrs.src=value; this.navigations++; this.mediaTime=0; }
  get src() { return this.attrs.src; }
  getAttribute(name) { return this.attrs[name]??null; }
  removeAttribute(name) { if(name==='src'&&this.attrs.src){this.navigations++;this.mediaTime=0;} delete this.attrs[name]; }
  setAttribute(name,value) { this.attrs[name]=value; }
}
const document={getElementById:id=>nodes.get(id)||null,querySelectorAll:()=>[]};
const windowHandlers={};
const window={addEventListener(type,handler){windowHandlers[type]=handler;}};
const location={origin:'http://localhost:8876',hash:''};
const fetch=()=>{throw new Error('Unexpected real network call');};
const project=id=>({id,name:'Recording '+id,games:[id==='A'?1:2],layout:'pml',
  artifacts:['g0.json'],has_fit:true,stale_exports:false,open_items:1,results_revision:[1],job:{running:false}});
const A=project('A'), B=project('B');
"""
    initialization = f"for(const id of {json.dumps(ids)}) nodes.set(id,new Element(id));\n"
    path = tmp_path / "studio-navigation.mjs"
    path.write_text(harness + initialization + script + "\nprojects=[A,B];\n" + scenario, encoding="utf-8")
    result = subprocess.run(["node", str(path)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("operation", ["analyze", "settings"])
def test_delayed_response_cannot_replace_another_project(tmp_path, operation):
    scenario = """
openProject('A');
let resolveRequest, requested;
api=(path,body)=>{requested={path,body};return new Promise(resolve=>resolveRequest=resolve);};
"""
    if operation == "analyze":
        scenario += "const pending=action('analyze');\n"
    else:
        scenario += """
setView('settings');
$('settings-games').value='11,12';
$('settings-layout').value='custom-layout';
const pending=$('settings-form').onsubmit({preventDefault(){}});
"""
    scenario += """
openProject('B');
const frame=$('review-frame'), before={src:frame.src,navigations:frame.navigations,pane:$('project-pane').innerHTML};
resolveRequest({...A,games:[11,12],job:{running:true,stage:'Analyzing'}});
await pending;
console.log(JSON.stringify({selected,activeView,requested,before,
  after:{src:frame.src,navigations:frame.navigations,pane:$('project-pane').innerHTML},
  updatedA:projects.find(p=>p.id==='A').games,error:$('global-error').textContent||''}));
"""
    result = run_studio(tmp_path, scenario)
    assert result["selected"] == "B"
    assert result["activeView"] == "review"
    assert result["before"] == result["after"]
    assert result["updatedA"] == [11, 12]  # Response is retained for its own project.
    assert result["requested"]["path"] == f"/api/projects/A/{operation}"
    assert result["error"] == ""
    if operation == "settings":
        assert result["requested"]["body"] == {"games": [11, 12], "layout": "custom-layout"}


def test_polling_preserves_settings_draft_and_review_player(tmp_path):
    result = run_studio(tmp_path, """
openProject('A'); setView('settings');
const games=$('settings-games'), layout=$('settings-layout'), form=$('settings-form');
games.value='42,43'; layout.value='unsaved-layout';
const paneWrites=$('project-pane').writes;
api=async path=>{if(path!=='/api/workspace')throw new Error(path);
  return {projects:[{...A,results_revision:[2],job:{running:false,log:['New log line']}},B],setup:{ready:true},sources:[]};};
await refresh(); await refresh();
const settings={games:games.value,layout:layout.value,sameForm:form===$('settings-form'),
  sameInputs:games===$('settings-games')&&layout===$('settings-layout'),redraws:$('project-pane').writes-paneWrites};
setView('review');
const frame=$('review-frame'), navigations=frame.navigations;
frame.mediaTime=123.5;
await refresh(); await refresh();
console.log(JSON.stringify({settings,review:{sameFrame:frame===$('review-frame'),
  navigations:frame.navigations-navigations,mediaTime:frame.mediaTime,src:frame.src}}));
""")
    assert result["settings"] == {
        "games": "42,43", "layout": "unsaved-layout", "sameForm": True,
        "sameInputs": True, "redraws": 0,
    }
    assert result["review"] == {
        "sameFrame": True, "navigations": 0, "mediaTime": 123.5,
        "src": "/review/A/?embedded=1",
    }


def test_processing_log_keeps_its_expanded_state_while_polling(tmp_path):
    result = run_studio(tmp_path, """
A.artifacts=[]; A.job={running:true,stage:'Reading tiles',log:['first']};
openProject('A');
$('processing-log').open=true;
api=async()=>({projects:[{...A,job:{...A.job,stage:'Reconstructing',log:['first','second']}}],setup:{ready:true}});
await refresh();
const expanded=$('processing-log').open;
const updated=$('project-pane').innerHTML.includes('second');
$('processing-log').open=false;
await refresh();
console.log(JSON.stringify({expanded,updated,collapsed:!$('processing-log').open}));
""")
    assert result == {"expanded": True, "updated": True, "collapsed": True}


def test_slow_results_request_is_not_restarted_by_polling(tmp_path):
    result = run_studio(tmp_path, """
openProject('A'); activeView='results';
let requests=0, resolveResults;
api=(path)=>{
  if(path==='/api/workspace') return Promise.resolve({projects:[A,B],setup:{ready:true}});
  requests++; return new Promise(resolve=>resolveResults=resolve);
};
const pending=loadResults(A);
const paneWrites=$('project-pane').writes;
await refresh(); await refresh();
const during={requests,redraws:$('project-pane').writes-paneWrites};
resolveResults({games:[],revision:A.results_revision,pending_rebuilds:[]});
await pending;
console.log(JSON.stringify({during,loadingCleared:resultsLoading===null,rendered:$('project-pane').innerHTML}));
""")
    assert result["during"] == {"requests": 1, "redraws": 0}
    assert result["loadingCleared"] is True
    assert "No game logs yet" in result["rendered"]


def test_calibration_draft_blocks_actions_and_navigation_until_saved(tmp_path):
    result = run_studio(tmp_path, """
openProject('A');
const frame=$('review-frame'); frame.contentWindow={};
const signal=(origin,source,dirty)=>windowHandlers.message({origin,source,data:{type:'video2tenhou:calibration-dirty',dirty}});
signal('http://other-site',frame.contentWindow,true);
signal(location.origin,{},true);
const ignored=!calibrationDirty;
signal(location.origin,frame.contentWindow,true);
const navigations=frame.navigations;
openProject('B'); showNew(); setView('settings'); await action('analyze');
const blocked={selected,activeView,navigations:frame.navigations-navigations,message:$('global-error').textContent};
signal(location.origin,frame.contentWindow,false);
openProject('B');
console.log(JSON.stringify({ignored,blocked,after:selected,dirty:calibrationDirty}));
""")
    assert result["ignored"] is True
    assert result["blocked"]["selected"] == "A" and result["blocked"]["activeView"] == "review"
    assert result["blocked"]["navigations"] == 0
    assert "Save or discard" in result["blocked"]["message"]
    assert result["after"] == "B" and result["dirty"] is False
