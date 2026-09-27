"""The review page's script must parse: a syntax error leaves the page stuck at "Loading"."""

from tests.paths import ROOT
import re
import json
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = ROOT / "src" / "video2tenhou" / "tool" / "static" / "index.html"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
@pytest.mark.parametrize("page", ["index.html", "studio.html"])
def test_page_script_parses(tmp_path, page):
    html = (PAGE.parent / page).read_text(encoding="utf-8")
    m = re.search(r"<script>([\s\S]*)</script>", html)
    assert m, "no script block"
    js = tmp_path / "page.js"
    js.write_text(m.group(1), encoding="utf-8")
    r = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_incomplete_search_renders_retry_without_tile_answers(tmp_path):
    """Execute real review rendering with a processing item and a legacy note.

    Minimal DOM sinks collect generated markup; network/evidence creation is
    forbidden so this also verifies that a solver issue doesn't decode clips.
    """
    from html.parser import HTMLParser

    script = re.search(r"<script>([\s\S]*)</script>", PAGE.read_text(encoding="utf-8"))[1]
    script = script.split("// ------------------------------------------------------------------ label view")[0]
    harness = """
const nodes = new Map();
const document = {querySelectorAll(){return [];},querySelector(selector) {
  if (selector === '#tilelist') return null;
  if (!nodes.has(selector)) nodes.set(selector, {innerHTML:'',offsetWidth:0,children:[],querySelectorAll(){return [];},classList:{add(){},remove(){},toggle(){}}});
  return nodes.get(selector);
}};
const window = {scrollTo(){}};
const fetch = () => { throw new Error('Processing items must not fetch video evidence'); };
"""
    scenario = """
HANDS[19] = {hand:19,game:1,kyoku:3,honba:1,decoded_at:1,t_start:20,t_end:50,corner_wind:{}};
ITEMS = [{hand:19,idx:0,kind:'solver_incomplete',text:'Search stopped before proving the best result.'}];
FACTS[19] = [{kind:'note',text:ITEMS[0].text,ts:2,author:'tool'}];
NOTES[19] = [];
evidenceBlock = () => { throw new Error('Processing items must not render a video'); };
await showItem(19, 0);
console.log(JSON.stringify({answered:isAnswered(ITEMS[0]),card:nodes.get('#qcard').innerHTML,list:nodes.get('#qlist').innerHTML}));
"""
    path = tmp_path / "processing-review.mjs"
    path.write_text(harness + script + scenario, encoding="utf-8")
    result = subprocess.run(["node", str(path)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)

    class Controls(HTMLParser):
        def __init__(self):
            super().__init__()
            self.actions = []
            self.tags = []

        def handle_starttag(self, tag, attrs):
            self.tags.append(tag)
            if tag == "button":
                self.actions.append(dict(attrs).get("onclick", ""))

    controls = Controls()
    controls.feed(rendered["card"])
    assert rendered["answered"] is False  # A matching note must not dismiss an unfinished search.
    assert "redecode(19)" in controls.actions
    assert any("showHandDetail(19)" in action for action in controls.actions)
    assert not any("answer" in action.lower() for action in controls.actions)
    assert "video" not in controls.tags
    assert "Hand 20" in rendered["list"] and "Hand 191" not in rendered["list"]
    assert "Search incomplete" in rendered["card"] and "Rebuild this hand to retry" in rendered["card"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_uncertain_tile_group_answers_only_the_selected_choice(tmp_path):
    """Render the real group/editors and save distinct facts through their handlers.

    A starting-hand Can't tell must never answer draw zero, and an old generic
    note must not dismiss the group. Resolved choices also lose their turn badge.
    """
    script = re.search(r"<script>([\s\S]*)</script>", PAGE.read_text(encoding="utf-8"))[1]
    script = script.split("// ------------------------------------------------------------------ label view")[0]
    harness = """
const nodes = new Map();
const document = {querySelectorAll(){return [];},querySelector(selector) {
  if (selector === '#tilelist' && !nodes.get('#qcard')?.innerHTML.includes('id="tilelist"')) return null;
  if (!nodes.has(selector)) nodes.set(selector, {innerHTML:'',dataset:{},offsetWidth:0,children:[],
    querySelectorAll(){return [];},classList:{add(){},remove(){},toggle(){}}});
  return nodes.get(selector);
}};
const window = {scrollTo(){}};
const fetch = () => { throw new Error('Unexpected network call'); };
"""
    scenario = """
HANDS[0] = {hand:0,game:0,kyoku:0,honba:0,decoded_at:10,t_start:20,t_end:80,corner_wind:{top:'N'}};
ITEMS = [{hand:0,idx:0,kind:'uncertain_tiles',text:'uncertain',choices:[
  {field:'draw',seat:'N',j:0,value:'2p',margin:0.5,t:28},
  {field:'draw',seat:'N',j:1,value:'1z',margin:0.3,t:58},
  {field:'haipai',seat:'N',j:-1,value:Array(13).fill('1m'),margin:0.5,t:20}
]}];
FACTS[0] = [{kind:'note',text:'uncertain',ts:11}, {kind:'draw',seat:'N',j:0,tile:'2p',ts:9}];
NOTES[0] = [];
const saved = [];
api = async (url, fact) => { if(url !== 'api/facts') throw new Error(url); saved.push(fact); return {...fact,ts:12}; };
turnOf = async (hand,seat,j) => ({seat,j,t:j===0?30:60,t_prev:20,discard:'2p'});
handEvidence = () => '<p>draw evidence</p>';
contextHtml = async () => '';
let clips = 0;
evidenceBlock = () => { clips++; return '<video></video>'; };
await showItem(0,0);
const overview = nodes.get('#qcard').innerHTML;
const navigation = nodes.get('#review-nav').innerHTML;
const initiallyAnswered = isAnswered(ITEMS[0]);
const overviewClips = clips;
await showItem(0,0,2);
const haipaiEditor = nodes.get('#qcard').innerHTML;
const initialTiles = JSON.parse(nodes.get('#tilelist').dataset.tiles);
await answerHaipaiLost();
const afterHaipai = {answered:isAnswered(ITEMS[0]),drawBadge:drawNeedsReview(0,{seat:'N',j:0}),card:nodes.get('#qcard').innerHTML};
await showItem(0,0,0);
const drawEditor = nodes.get('#qcard').innerHTML;
await answerDraw('2p');
const afterDraw = {first:drawNeedsReview(0,{seat:'N',j:0}),second:drawNeedsReview(0,{seat:'N',j:1})};
await showItem(0,0,1);
await answerLost();
const activeVideo = {paused:false,ended:false};
document.querySelectorAll = () => [activeVideo];
REVIEW_REVISION = '["old"]'; REVIEW_DIRTY = false;
api = async () => ['new'];
let refreshCount = 0;
refreshReview = async () => {refreshCount++;};
await checkReviewRevision();
const whilePlaying = {refreshCount,revision:REVIEW_REVISION,banner:nodes.get('#banner').innerHTML};
activeVideo.paused = true;
REVIEW_DIRTY = true;
await checkReviewRevision();
const whileEditing = refreshCount;
REVIEW_DIRTY = false;
await checkReviewRevision();
console.log(JSON.stringify({overview,navigation,initiallyAnswered,overviewClips,haipaiEditor,initialTiles,
  afterHaipai,drawEditor,afterDraw,saved,whilePlaying,whileEditing,refreshCount,done:isAnswered(ITEMS[0]),finalCard:nodes.get('#qcard').innerHTML}));
"""
    path = tmp_path / "uncertain-review.mjs"
    path.write_text(harness + script + scenario, encoding="utf-8")
    result = subprocess.run(["node", str(path)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)
    assert rendered["initiallyAnswered"] is False
    assert rendered["overviewClips"] == 0  # Evidence is loaded only for the selected choice.
    assert 'class="workbench"' in rendered["overview"]
    assert 'id="primary-evidence"' in rendered["overview"]
    assert rendered["overview"].count('<video ') == 1
    assert 'data-src="api/frame?' in rendered["haipaiEditor"]  # Crop waits for an explicit request.
    assert ' src="api/frame?' not in rendered["haipaiEditor"]
    assert 'whole frame, uncropped' not in rendered["overview"].lower()
    assert 'Draw 2</h2>' in rendered["overview"]
    # Least-certified choice first; equal margins retain their original indices/order.
    assert rendered["navigation"].index('value="0:1"') < rendered["navigation"].index('value="0:0"') < rendered["navigation"].index('value="0:2"')
    assert "answerNote" not in rendered["overview"]
    assert 'onclick="answerHaipaiLost()"' in rendered["haipaiEditor"]
    rows = re.findall(r'<div class="palette-row">(.*?)</div>', rendered["haipaiEditor"])
    choices = [re.findall(r'data-tile="([0-9][mpsz])"', row) for row in rows]
    assert choices == [
        [f"{rank}{suit}" for rank in (1, 2, 3, 4, 5, 0, 6, 7, 8, 9)] for suit in "mps"
    ] + [[f"{rank}z" for rank in range(1, 8)]]
    assert rendered["initialTiles"] == ["1m"] * 13
    assert rendered["afterHaipai"]["answered"] is False
    assert rendered["afterHaipai"]["drawBadge"] is True
    assert 'Draw 2</h2>' in rendered["afterHaipai"]["card"]
    assert "draw evidence" in rendered["drawEditor"]
    assert rendered["afterDraw"] == {"first": False, "second": True}
    assert rendered["saved"] == [
        {"kind": "lost", "field": "haipai", "hand": 0, "seat": "N", "t": 20},
        {"kind": "draw", "hand": 0, "seat": "N", "j": 0, "t": 30, "t_discard": 30, "tile": "2p"},
        {"kind": "lost", "hand": 0, "seat": "N", "j": 1, "t": 60},
    ]
    assert rendered["done"] is True
    assert "Every choice has an answer" in rendered["finalCard"]
    assert rendered["whilePlaying"]["refreshCount"] == 0
    assert rendered["whilePlaying"]["revision"] == '["old"]'
    assert "Load updates" in rendered["whilePlaying"]["banner"]
    assert rendered["whileEditing"] == 0
    assert rendered["refreshCount"] == 1


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_review_queue_confidence_order_skip_answers_and_hand_filter(tmp_path):
    """Exercise real navigation across groups, hands and persisted answers."""
    script = re.search(r"<script>([\s\S]*)</script>", PAGE.read_text(encoding="utf-8"))[1]
    script = script.split("// ------------------------------------------------------------------ label view")[0]
    harness = """
const nodes=new Map();
const document={querySelectorAll(){return [];},querySelector(key){
  if(!nodes.has(key))nodes.set(key,{innerHTML:'',textContent:'',querySelectorAll(){return [];},classList:{add(){},remove(){},toggle(){}}});
  return nodes.get(key);
}};
const window={scrollTo(){}};
const fetch=()=>{throw new Error('Unexpected network');};
const calibBlocked=()=>false;
"""
    scenario = """
HANDS=[0,1].map(hand=>({hand,game:0,kyoku:hand,honba:0,decoded_at:10,corner_wind:{BR:'N'}}));
ITEMS=[
 {hand:0,idx:0,kind:'ura',margin:null},
 {hand:0,idx:1,kind:'uncertain_tiles',choices:[
   {field:'haipai',seat:'N',j:-1,margin:.4,value:[]},
   {field:'draw',seat:'N',j:0,margin:0,alternative_gap:90,value:'2p'},
   {field:'draw',seat:'N',j:1,margin:.2,alternative_gap:0,value:'1z'},
   {field:'draw',seat:'N',j:2,margin:.2,value:'9m'}]},
 {hand:0,idx:2,kind:'discard',seat:'N',j:9,margin:.1},
 {hand:1,idx:0,kind:'draw',seat:'N',j:3,margin:.5,lost:true},
 {hand:1,idx:1,kind:'discard',seat:'N',j:4,margin:0},
 {hand:1,idx:2,kind:'conflict'},
 {hand:1,idx:3,kind:'draw',seat:'N',j:5,margin:.15},
 {hand:1,idx:4,kind:'solver_incomplete',alternative_gap:0},
 {hand:1,idx:5,kind:'lost',seat:'N',j:6}
];
FACTS={0:[{kind:'draw',seat:'N',j:0,ts:9}],1:[{kind:'draw',seat:'N',j:5,ts:11}]};
ITEMS.forEach(item=>item.done=isAnswered(item));
const unchanged=JSON.stringify(ITEMS);
const keys=()=>reviewDecisions().map(({item,choice})=>`${item.hand}:${item.idx}:${choice??''}`);
const initial=keys();
renderNavigator(1);
const individualLabels=nodes.get('#review-nav').innerHTML;
const shown=[];
showItem=async(hand,idx,choice=null)=>{
  const item=ITEMS.find(x=>x.hand===hand&&x.idx===idx);
  CUR={...item,choiceIndex:choice};shown.push([hand,idx,choice]);
};
renderQuestions();
await nextItem(); // Skip structural conflict, then both explicit missing-evidence decisions.
await nextItem();
await nextItem();
await nextItem(); // Skip advances to a different hand with the same certified margin.
const skipped=shown.slice();
await selectReviewHand(0);
const filtered=keys();
await selectDecision('1:2');
const selected=shown.at(-1);
FILTER=null;
await showItem(0,1,1);
const saved=[];
api=async(path,fact)=>{saved.push({path,fact});return {...fact,ts:12};};
await saveFact({hand:0,kind:'draw',seat:'N',j:0,tile:'2p'});
await afterTileAnswer({...CUR,aggregate:true});
const afterAnswer=shown.at(-1), remaining=keys();
// A new answer removes only its original choice, leaving tied siblings in order.
FILTER=0;renderNavigator(0);
console.log(JSON.stringify({initial,individualLabels,skipped,filtered,selected,afterAnswer,remaining,saved,
  navigator:nodes.get('#review-nav').innerHTML,originalUnchanged:unchanged===JSON.stringify(ITEMS),
  choices:ITEMS[1].choices.map(c=>[c.field,c.j])}));
"""
    path = tmp_path / "review-order.mjs"
    path.write_text(harness + script + scenario, encoding="utf-8")
    result = subprocess.run(["node", str(path)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual["initial"] == ["1:2:", "1:0:", "1:5:", "0:1:1", "1:1:", "0:2:", "0:1:2", "0:1:3", "0:1:0", "0:0:", "1:4:"]
    assert 'value="0:" >Draw 4 — North</option>' in actual["individualLabels"]
    assert 'value="5:" >Draw 7 — North</option>' in actual["individualLabels"]
    assert 'value="1:" >Discard 5 — North</option>' in actual["individualLabels"]
    assert actual["skipped"] == [[1, 2, None], [1, 0, None], [1, 5, None], [0, 1, 1], [1, 1, None]]
    assert actual["filtered"] == ["0:1:1", "0:2:", "0:1:2", "0:1:3", "0:1:0", "0:0:"]
    assert actual["selected"] == [0, 1, 2]
    assert actual["afterAnswer"] == [1, 2, None]  # Resume at the first remaining decision, not the next raw index.
    assert "0:1:1" not in actual["remaining"] and "1:3:" not in actual["remaining"]
    assert actual["saved"] == [{"path": "api/facts", "fact": {"hand": 0, "kind": "draw", "seat": "N", "j": 0, "tile": "2p"}}]
    assert actual["originalUnchanged"]  # No item/choice reindexing or in-place sorting.
    assert actual["choices"] == [["haipai", -1], ["draw", 0], ["draw", 1], ["draw", 2]]
    assert actual["navigator"].index('value="2:"') < actual["navigator"].index('value="1:2"') < actual["navigator"].index('value="1:3"') < actual["navigator"].index('value="1:0"')


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_pending_rebuild_real_review_js_lifecycle(tmp_path):
    """Run real save/load/rebuild handlers, with HTTP responses and clock controlled."""
    script = re.search(r"<script>([\s\S]*)</script>", PAGE.read_text(encoding="utf-8"))[1]
    script = script.split("// ------------------------------------------------------------------ label view")[0]
    harness = """
const location={search:'?embedded=1'};
const nodes=new Map(), timers=[], calls=[];
function node() { const classes=new Set(); return {innerHTML:'',textContent:'',classes,
  classList:{add(...xs){xs.forEach(x=>classes.add(x));},remove(...xs){xs.forEach(x=>classes.delete(x));},
  toggle(x,on){if(on)classes.add(x);else classes.delete(x);}}, querySelectorAll(){return [];}}; }
const retry=node();
const document={querySelector(key){if(!nodes.has(key))nodes.set(key,node());return nodes.get(key);},
 querySelectorAll(key){return key==='[data-redecode]'?[retry]:[];}};
const window={scrollTo(){}};
const setTimeout=fn=>{timers.push(fn);};
let pending=[], running=false, fail=false, saved=[];
const hands=[0,1].map(hand=>({hand,game:0,kyoku:hand,honba:0,decoded_at:0,status:'review'}));
async function fetch(path,options={}) {
  calls.push({path,method:options.method||'GET',body:options.body?JSON.parse(options.body):null});
  let result;
  if(path==='api/facts'&&options.method==='POST') {
    result={...JSON.parse(options.body),ts:10};saved.push(result);pending=[...new Set([...pending,result.hand])];
  } else if(path==='api/facts') result=saved;
  else if(path==='api/hands') result=hands.map(h=>({...h,pending_rebuild:pending.includes(h.hand)}));
  else if(path==='api/items') result=[];
  else if(path==='api/decode_pending') {
    if(options.method==='POST') {running=true;result={running:true,hands:[...pending],hands_done:0,hands_total:pending.length,pending:[...pending],error:null};}
    else {if(running){running=false;if(!fail)pending=[];}result={running:false,pending:[...pending],error:fail?'Rebuild failed; Analyze recording':null};}
  } else throw new Error('Unexpected endpoint '+path);
  return {ok:true,status:200,json:async()=>result};
}
"""
    scenario = """
let rendered=0;renderQuestions=()=>{rendered++;};renderHands=()=>{};
await loadAll();
const initiallyHidden=nodes.get('#rebuild-controls').classes.has('hidden')&&!retry.classes.has('hidden');
await saveFact({hand:0,kind:'draw',tile:'2p',seat:'N',j:0});
const afterSave=nodes.get('#rebuild-controls').innerHTML;
pending.push(1); // A deletion from another tab exists only in the server ledger.
await loadAll();
const beforeJob={html:nodes.get('#rebuild-controls').innerHTML,retryHidden:retry.classes.has('hidden')};
await rebuildChanges();
const duringJob=nodes.get('#rebuild-controls').innerHTML;
await timers.shift()();
const completed={hidden:nodes.get('#rebuild-controls').classes.has('hidden'),retryHidden:retry.classes.has('hidden'),pending:pendingHands(),rendered};
const firstPosts=calls.filter(c=>c.method==='POST'&&c.path==='api/decode_pending');
await saveFact({hand:0,kind:'discard',tile:'1m',seat:'N',j:1});
fail=true;await rebuildChanges();await timers.shift()();
const failed={pending:pendingHands(),banner:nodes.get('#banner').textContent,html:nodes.get('#rebuild-controls').innerHTML};
REVIEW_DIRTY=true;const before=calls.length;await rebuildChanges();
const guarded=calls.length===before;REVIEW_DIRTY=false;
// Reconnecting follows the active job with GET, without posting another job.
fail=false;running=true;const postsBefore=calls.filter(c=>c.method==='POST'&&c.path==='api/decode_pending').length;
await rebuildChanges(true);
const postsAfter=calls.filter(c=>c.method==='POST'&&c.path==='api/decode_pending').length;
await saveFact({hand:0,kind:'draw',tile:'4z',seat:'N',j:2});
await rebuildChanges();
REVIEW_DIRTY=true;const renderCount=rendered;$('#qcard').innerHTML='unsaved answer';
await timers.shift()();
const held={rendered:rendered===renderCount,draft:nodes.get('#qcard').innerHTML,banner:nodes.get('#banner').innerHTML};
console.log(JSON.stringify({initiallyHidden,afterSave,beforeJob,duringJob,completed,firstPosts,failed,guarded,postsBefore,postsAfter,held}));
"""
    path = tmp_path / "pending-review.mjs"
    path.write_text(harness + script + scenario, encoding="utf-8")
    result = subprocess.run(["node", str(path)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)
    assert rendered["initiallyHidden"]
    assert "Rebuild changes (1 hand)" in rendered["afterSave"]
    assert "Rebuild changes (2 hands)" in rendered["beforeJob"]["html"] and rendered["beforeJob"]["retryHidden"]
    assert "disabled" in rendered["duringJob"] and "Rebuilding changes" in rendered["duringJob"]
    assert rendered["completed"] == {"hidden": True, "retryHidden": False, "pending": [], "rendered": 1}
    assert rendered["firstPosts"] == [{"path": "api/decode_pending", "method": "POST", "body": {}}]
    assert rendered["failed"]["pending"] == [0] and "Analyze recording" in rendered["failed"]["banner"]
    assert "Rebuild changes (1 hand)" in rendered["failed"]["html"] and "disabled" not in rendered["failed"]["html"]
    assert rendered["guarded"] and rendered["postsBefore"] == rendered["postsAfter"]
    assert rendered["held"]["rendered"] and rendered["held"]["draft"] == "unsaved answer"
    assert "Load updates" in rendered["held"]["banner"]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not on PATH")
def test_turn_evidence_full_frame_navigation_and_late_response(tmp_path):
    """Actual turn handlers keep one primary player and reject obsolete responses."""
    script = re.search(r"<script>([\s\S]*)</script>", PAGE.read_text(encoding="utf-8"))[1]
    script = script.split("// ------------------------------------------------------------------ label view")[0]
    harness = """
const nodes=new Map();
let pauses=0;
const playing={paused:false,ended:false,pause(){this.paused=true;pauses++;}};
const document={querySelector(key){if(!nodes.has(key))nodes.set(key,{innerHTML:'',outerHTML:'',dataset:{},
 classList:{add(){},remove(){},toggle(){}},scrollIntoView(){},querySelectorAll(){return [];}});return nodes.get(key);},
 querySelectorAll(key){return key==='video'?[playing]:[];}};
const window={scrollTo(){}};
const fetch=()=>{throw Error('Unexpected fetch');};
const calibBlocked=()=>false;
"""
    scenario = """
$('#questions').querySelectorAll=()=>[playing];
$('#qcard').innerHTML='unsaved answer';REVIEW_DIRTY=true;
go('hands');
const navigation={pauses,draft:$('#qcard').innerHTML,dirty:REVIEW_DIRTY};
DETAIL_HAND=0;
const entry={hand:0,game:0,kyoku:0,honba:0,t_start:10,t_end:100,corner_wind:{TL:'N'}};
const decoded={entry,decode:{dealer:'E',dora:['1m'],result:{outcome:'ryuukyoku'},haipai:{},problems:[],t_last:70,turns:[{i:0,seat:'N',t:30,t_prev:20,discard:'2p'},
 {i:1,seat:'N',t:70,t_prev:60,discard:'1z'}]}};
CUR_E=entry;
api=async()=>decoded;await showHandDetail(0);const detailMarkup=$('#handdetail').innerHTML;
const waiting=[];api=()=>new Promise(resolve=>waiting.push(resolve));
contextHtml=async()=>'';
const first=turnEvidence(0,0), second=turnEvidence(0,1);
waiting[1](decoded);await second;
const latest=$('#turnev').innerHTML;
waiting[0](decoded);await first;
const staleIgnored=$('#turnev').innerHTML===latest;
$('#primary-evidence').outerHTML='question player';
$('#turn-primary-evidence').dataset={t0:'54',t1:'100'};
extendPrimary(0,15,'turn-primary-evidence');
const extended=$('#turn-primary-evidence').outerHTML;
const questionUntouched=$('#primary-evidence').outerHTML;
playing.paused=false;REVIEW_DIRTY=false;REVIEW_REVISION='["old"]';
api=async()=>['new'];let refreshes=0;refreshReview=async()=>{refreshes++;};
await checkReviewRevision();
const poll={refreshes,html:$('#turnev').innerHTML,draft:$('#qcard').innerHTML};
$('#handdetail').querySelectorAll=()=>[playing];
showHandsList();
console.log(JSON.stringify({navigation,detailMarkup,latest,staleIgnored,extended,questionUntouched,poll,pauses,detail:DETAIL_HAND}));
"""
    path = tmp_path / "turn-evidence.mjs"
    path.write_text(harness + script + scenario, encoding="utf-8")
    result = subprocess.run(["node", str(path)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    rendered = json.loads(result.stdout)
    assert rendered["navigation"] == {"pauses": 1, "draft": "unsaved answer", "dirty": True}
    assert rendered["detailMarkup"].index('id="turnev"') < rendered["detailMarkup"].index('Starting hands')
    assert '(TL) (TL)' not in rendered["detailMarkup"]
    assert "Turn 2:" in rendered["latest"] and rendered["staleIgnored"]
    assert 'id="turn-primary-evidence"' in rendered["latest"]
    assert len(re.findall(r'<video\b[^>]*\ssrc=', rendered["latest"])) == 1
    assert 'region=frame' in rendered["latest"]
    assert '<details class="review-details"><summary>Additional evidence</summary>' in rendered["latest"]
    assert 'data-src="api/clip?' in rendered["latest"] and 'data-src="api/frame?' in rendered["latest"]
    assert 't1=115.0' in rendered["extended"] and 'id="turn-primary-evidence"' in rendered["extended"]
    assert rendered["questionUntouched"] == "question player"
    assert rendered["poll"] == {"refreshes": 0, "html": rendered["latest"], "draft": "unsaved answer"}
    assert rendered["pauses"] == 2 and rendered["detail"] is None
