"""NeuroML AI Coach Dashboard v2 — the full brain, live.

Endpoints:
  /            comprehensive dashboard page
  /metrics     ml_metrics.json (cached 1s)
  /coach       last 60 ai_coach.jsonl coach reviews
  /brain       live per-neuron firing + Q-values + equations (reads latest bot decision)
  /shot        live phone screenshot JPEG (cached 1.2s so page stays fast)
  /health      quick status of bot process + adb device
  /toggle      POST {"on":true|false} — start/stop the bot process (works from
               the in-page button in BOTH the Electron app window and plain
               localhost: same endpoint, same behavior everywhere)
"""
import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # threaded: /shot never blocks other endpoints

# Pre-warm the vision module (torch import ~1.5s once) so the FIRST /vision or
# /shot hit isn't slow and the page never looks "perpetually loading".
try:
    import bot_ml
except Exception:
    bot_ml = None

CFG = json.load(open("bot_config.json"))
DEV = CFG["device"]
PORT = 8765

CACHE = {"metrics": "", "coach": "[]", "shot": None, "metrics_t": 0, "coach_t": 0, "shot_t": 0}
CACHE_LOCK = threading.Lock()

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>NeuroML — Computational Neuroscience Testbed</title>
<style>
:root{
 --bg:#0b0e13; --surface:#121820; --surface2:#171f29; --line:#232d3a; --line-soft:#1b232e;
 --txt:#e8edf4; --dim:#8fa0b3; --faint:#62738a;
 --accent:#3ad0b6; --accent-soft:rgba(58,208,182,.12); --accent2:#8b93ff;
 --ok:#3fb950; --warn:#d9a13b; --bad:#f85c70;
 --font-ui:Segoe UI,Roboto,-apple-system,sans-serif;
 --font-mono:Cascadia Code,Consolas,monospace;
 --r-lg:12px; --r-md:8px; --sp:14px;
 --shadow:0 1px 0 rgba(255,255,255,.02) inset, 0 8px 24px -18px rgba(0,0,0,.55);
 --t-fast:110ms cubic-bezier(.4,0,.2,1); --t-slow:260ms cubic-bezier(.4,0,.2,1);
}
*{box-sizing:border-box}
body{background:radial-gradient(1200px 500px at 50% -180px,#10161f 0%,var(--bg) 60%);color:var(--txt);
 font-family:var(--font-ui);margin:0;padding:var(--sp);line-height:1.45}
h1{margin:0 0 2px;font-size:17px;font-weight:650;letter-spacing:.2px}
h1 .mark{color:var(--accent)}
.sub{color:var(--dim);font-size:12px;margin-bottom:var(--sp)}
.sub a, .card a{color:var(--accent);text-decoration:none;border-bottom:1px dotted var(--accent)}
.sub a:hover, .card a:hover{color:#fff;border-bottom-style:solid}
.grid{display:grid;grid-template-columns:310px 1fr 350px;gap:var(--sp);max-width:1400px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:var(--r-lg);
 padding:var(--sp);margin-bottom:var(--sp);box-shadow:var(--shadow);
 transition:border-color var(--t-fast)}
.card:hover{border-color:#2d3a4a}
.card>h2{font-size:10.5px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;
 font-weight:600;margin:0 0 8px}
.kpi{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}
.kpi div{background:var(--surface2);border:1px solid var(--line-soft);border-radius:var(--r-md);
 padding:8px 10px;font-size:17px;font-weight:700;font-variant-numeric:tabular-nums}
.kpi div span{display:block;font-size:9.5px;color:var(--dim);font-weight:500;
 text-transform:uppercase;letter-spacing:.7px;margin-bottom:2px}
canvas{width:100%;height:130px;margin:4px 0}
.eq{font-family:var(--font-mono);font-size:12px;background:var(--surface2);border:1px solid var(--line-soft);
 border-radius:var(--r-md);padding:8px;color:#9dc4ff;white-space:pre-wrap}
.brain{display:flex;flex-direction:column;gap:2px;margin:2px 0;opacity:.9}.brain .layer{display:grid;grid-template-columns:repeat(128,1fr);gap:1px;width:100%}
.dot{width:100%;height:1.4px;border-radius:1px;transition:background var(--t-fast)}
.coach{height:210px;overflow-y:auto;font-size:12px;font-family:var(--font-mono)}
.coach div{padding:3px 0;border-bottom:1px solid var(--line-soft)}.pun{color:var(--bad)}.men{color:var(--accent2)}.ok{color:var(--ok)}
.wl{display:flex;gap:2px;flex-wrap:wrap}.wl i{display:inline-block;width:11px;height:10px;border-radius:2px}
.pill{display:inline-block;background:var(--surface2);border:1px solid var(--line-soft);border-radius:6px;
 padding:2px 8px;font-size:11px;margin:2px}
.dopa{height:230px;overflow-y:auto;font-size:12px;font-family:var(--font-mono)}
.dopa div{display:flex;align-items:center;gap:6px;padding:3px 6px;border-bottom:1px solid var(--line-soft);border-radius:4px}
.dopa div:nth-child(odd){background:var(--surface2)}
.dopa .bar{display:inline-block;height:9px;border-radius:2px;min-width:2px}
.dopa .pos{color:var(--ok);font-weight:800}.dopa .neg{color:var(--bad);font-weight:800}
.painlist div{padding:2px 6px;font-size:11px;font-family:var(--font-mono);color:#ef9a6c}
.dopa .ev{color:var(--txt);flex:1;margin-left:8px}.dopa .t{color:var(--dim);width:70px}
.dopa .big{font-size:13px;border-left:3px solid var(--warn)}
.opp{font-size:12px;color:var(--dim)}.opp b{color:var(--txt)}
.rules{font-size:11px;color:var(--accent);background:var(--accent-soft);border:1px solid rgba(58,208,182,.25);
 border-radius:var(--r-md);padding:8px;margin-top:6px}
.rules li{margin:2px 0;list-style:none}
.phone{width:100%;height:560px;border:1px solid var(--line);border-radius:var(--r-lg);overflow:hidden;background:#000;position:relative}
.phone img{width:100%;height:100%;object-fit:contain;display:block}
.phone .loading{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--faint);font-size:12px}
.btn{background:var(--surface2);color:var(--txt);border:1px solid var(--line);border-radius:var(--r-md);
 padding:6px 12px;font-size:12px;cursor:pointer;transition:all var(--t-fast)}
.btn:hover:not(:disabled){border-color:var(--accent);color:var(--accent)}
.btn:disabled{opacity:.45;cursor:not-allowed}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:4px}
@media (max-width:1100px){
 .grid{grid-template-columns:1fr 1fr}
 .grid>div:first-child, .grid>div:last-child{grid-column:1/-1}
}
@media (max-width:720px){
 .grid{grid-template-columns:1fr}
 .phone{height:auto;aspect-ratio:1080/2340}
 .kpi{grid-template-columns:repeat(2,1fr)}
 h1{font-size:15px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style></head><body>
<h1><span class="mark">NeuroML</span> — battle brain · live dashboard</h1>
<div class="sub" style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
  <button id="botToggle" class="btn" style="font-weight:700;min-width:150px;border-radius:8px" aria-label="start or stop the bot">Brain: …</button>
  <span id="botState" style="color:var(--faint)">checking…</span>
  <span><span id="hdr_profile">13.72M-param 1024-wide × 6-layer</span> residual cortex · 90 LIF spiking neurons · 21-channel affective core · 6-layer local Digital Twin · deterministic Referee · <a href="/vision" target="_blank">raw vision overlay ↗</a></span>
</div>
<div class="grid" style="grid-template-columns:310px 1fr 360px">
  <div>
    <div class="card"><h2>Phone vision (live capture)</h2>
      <div class="phone"><img id="shot" src="/stream" alt="live phone screen (MJPEG)"></div>
      <div style="font-size:10px;color:var(--dim);margin-top:6px">🔵 ally tower HP · 🔴 enemy tower HP · 🟧 scan zones · 🟪 elixir bar · 🟩 affordable slot · <a href="/vision?fresh=1" target="_blank">refresh ↗</a></div></div>
    <div class="card"><h2>Vision state</h2>
      <div id="vision" style="font-size:12px;color:var(--dim)">waiting…</div></div>
    <div class="card"><h2>⚡ Reward events (as they hit the brain)</h2>
      <div class="dopa" id="dopa"><div style="color:var(--dim)">waiting for first reward…</div></div>
      <div style="font-size:10px;color:var(--dim);margin:4px 0 2px">bars scale to largest |dopamine| — red = punishment · amber = pain memory</div>
      <div class="painlist" id="painlist"></div>
      <div class="rules"><b>Tactical rules taught:</b>
        <li>spell finisher → tower ≤32% gets the shot, bridge drops exhaust elixir</li>
        <li>commit discipline → never bridge-drop a tank without support elixir</li>
        <li>cheap-cycle → 1-2 cost cards saved for defense</li>
        <li>buildings pull center, never bridge</li>
        <li>spell must prove in-area value — no cluster, no cast</li></div></div>
  </div>
  <div>
    <div class="card">
      <div class="kpi">
        <div><span>Train steps</span><span id="k_steps">-</span></div>
        <div><span>Loss (MA)</span><span id="k_loss">-</span></div>
        <div><span>TD error</span><span id="k_td">-</span></div>
        <div><span>Buffer</span><span id="k_buf">-</span></div>
        <div><span>Win rate</span><span id="k_wr">-</span></div>
      </div>
      <div id="record_meta" style="font-size:10px;color:var(--dim);margin:4px 0">-</div>
      <h2>Loss curve</h2><canvas id="c_loss" width=780 height=130></canvas>
      <h2>Reward history (tower dmg / wins / losses)</h2><canvas id="c_rw" width=780 height=130></canvas>
      <h2>Win / loss record (lifetime, last 60)</h2><div class="wl" id="wl"></div>
      <h2 style="margin-top:8px">Reward ledger (every logged contribution)</h2>
      <div id="ledger" style="font-size:11px;font-family:var(--font-mono);color:var(--dim)">waiting for first battle tick…</div>
      <div id="decision_meta" style="font-size:11px;font-family:var(--font-mono);margin-top:6px;color:var(--dim)">decision provenance waits for the next tick…</div>
      <div id="prediction" style="font-size:11px;font-family:var(--font-mono);margin-top:6px;color:var(--dim)">prediction telemetry waits for a validated tick…</div>
    </div>
    <div class="card"><h2>What the model is thinking (live equations)</h2>
      <div class="eq" id="eq">waiting for first battle tick…</div></div>
    <div class="card"><h2>📜 Battle log (recent battles + live brain log)</h2>
      <div id="battles" style="font-size:11px;font-family:var(--font-mono);color:var(--dim)">loading battle log…</div>
      <div id="blog" style="height:210px;overflow-y:auto;font-size:10.5px;font-family:var(--font-mono);
        background:var(--surface2);border:1px solid var(--line-soft);border-radius:var(--r-md);
        padding:6px;margin-top:8px;white-space:pre-wrap;line-height:1.5">loading brain log…</div>
      <div style="font-size:10px;color:var(--dim);margin-top:4px">
        every finished battle (verdict · dopamine · towers) and the live brain log — same feed on phone and desktop,
        so a phone-side failure is never a silent dead screen.</div></div>
    <div class="card"><h2>Groq mentor (end-of-match summaries — zero reward authority)</h2>
      <div class="coach" id="feed">waiting…</div></div>
    <div class="card"><h2>🃏 Card counters, matchup read & 30-day meta engine</h2>
      <div id="cardintel" style="font-size:11px;font-family:var(--font-mono);color:var(--dim)">loading card counter matrix & 30-day ladder meta…</div></div>
  </div>
  <div>
    <div class="card"><h2 style="font-size:12px;color:var(--dim);margin:0 0 6px">LIVE NEURON ACTIVATION — cortex layers (bar = per-neuron activation)</h2>
      <div style="font-size:10px;color:var(--dim);margin-bottom:4px">green = positive · red = negative · one bar per neuron, brighter = stronger</div>
      <div style="font-size:11px;color:var(--dim)">EYES (38 sensory neurons — directly observed vs estimated state)</div>
      <div id="senses" style="display:grid;grid-template-columns:repeat(3,1fr);gap:1px;font-size:9px;font-family:Consolas,monospace;margin:4px 0"></div>
      <div style="font-size:10px;color:var(--dim)">CORTEX 1 — sensory feature extraction</div>
      <div class="brain"><div class="layer" id="l1"></div></div>
      <div style="font-size:10px;color:var(--dim)">CORTEX 2 — lane & push concepts</div>
      <div class="brain"><div class="layer" id="l2"></div></div>
      <div style="font-size:10px;color:var(--dim)">CORTEX 3 — card counter & trade synergies</div>
      <div class="brain"><div class="layer" id="l3b"></div></div>
      <div style="font-size:10px;color:var(--dim)">CORTEX 4 — temporal threat integration</div>
      <div class="brain"><div class="layer" id="l4"></div></div>
      <div style="font-size:10px;color:var(--dim)">CORTEX 5 — strategic horizon composition</div>
      <div class="brain"><div class="layer" id="l5"></div></div>
      <div style="font-size:10px;color:var(--dim)">DEEP CORTEX (layers 6–8 — abstract executive representation → V/A)</div>
      <div class="brain"><div class="layer" id="deepcortex"></div></div>
      <div style="font-size:10px;color:var(--dim)">LIMBIC — state-value V(s) interneurons</div>
      <div class="brain"><div class="layer" id="l3"></div></div>
      <div style="font-size:11px;color:var(--dim);margin-top:4px">MOTOR — advantage A(s,a) across 20 actions (4 slots × 5 spots)</div>
      <div id="motor" style="font-size:10px;font-family:Consolas,monospace;margin-top:2px"></div>
      <div style="font-size:11px;color:var(--dim);margin-top:6px">V(s) baseline: <span id="vs" style="color:var(--yel)">-</span>
       · advantage spread <span id="asp" style="color:var(--yel)">-</span></div></div>
    <div class="card"><h2 style="font-size:12px;color:#3ad0b6;margin:0 0 6px">🌐 DIGITAL TWIN WORKSPACE — Local Ensemble & Risk-Aware MPC</h2>
      <div style="font-size:11px;font-family:Consolas,monospace;color:var(--dim)" id="twinpanel">twin waits for synchronized transitions…</div>
      <details style="margin-top:6px;font-size:10px;color:var(--dim);font-family:Consolas,monospace">
        <summary style="cursor:pointer;color:#9dc4ff">Advanced Twin Diagnostics (Candidate Rollouts, Assimilation & Drift)</summary>
        <div id="twindiag" style="margin-top:4px;padding:6px;background:#10161d;border-radius:6px;border:1px solid var(--line)">waiting for rollout diagnostics…</div>
      </details></div>
    <div class="card"><h2 style="font-size:12px;color:#8b93ff;margin:0 0 6px">🧬 LIF NEURON ANATOMY — Dendrites, Soma, Myelin Axon & Synaptic Terminals</h2>
      <svg id="neuroncell" viewBox="0 0 360 165" style="width:100%;height:165px;background:#0a0e13;border-radius:8px;border:1px solid var(--line)">
        <!-- Branching Dendritic Arbor (Excitatory & Inhibitory inputs) -->
        <g id="neuron_dendrites" stroke="#3d5a80" stroke-width="1.6" fill="none" opacity="0.95">
          <path d="M135,92 Q95,65 55,36 M75,50 Q62,30 42,22 M85,58 Q70,68 48,66"/>
          <path d="M135,92 Q100,52 82,25 M92,38 Q104,24 112,18"/>
          <path d="M135,92 Q90,92 40,92 M68,92 Q52,82 32,78 M68,92 Q52,104 34,110"/>
          <path d="M135,92 Q95,118 55,142 M78,128 Q60,135 38,136 M88,120 Q75,140 62,152"/>
          <path d="M135,92 Q112,125 95,150"/>
        </g>
        <!-- Dendritic Spines (synaptic contact sites) -->
        <g id="dendrite_spines" fill="#3ad0b6" opacity="0.8">
          <circle cx="55" cy="36" r="2.2"/><circle cx="42" cy="22" r="2"/><circle cx="82" cy="25" r="2.2"/>
          <circle cx="40" cy="92" r="2.2"/><circle cx="32" cy="78" r="2"/><circle cx="55" cy="142" r="2.2"/>
          <circle cx="95" cy="150" r="2"/>
        </g>
        <!-- Soma (Cell Body) & Axon Hillock -->
        <polygon points="148,84 168,92 148,100" fill="#152238" stroke="#3ad0b6" stroke-width="1.5"/>
        <circle id="neuron_soma" cx="135" cy="92" r="20" fill="#152238" stroke="#3ad0b6" stroke-width="2.2"/>
        <circle id="neuron_nucleus" cx="135" cy="92" r="7.5" fill="#8b93ff" opacity="0.9"/>
        <!-- Axon & Myelin Sheath Internodes -->
        <path d="M165,92 Q230,92 295,92" id="axon" stroke="#9dc4ff" stroke-width="2.8" fill="none"/>
        <g id="myelin_sheath" fill="#1d2d44" stroke="#3ad0b6" stroke-width="1.2">
          <rect x="172" y="86" width="24" height="12" rx="4"/>
          <rect x="204" y="86" width="24" height="12" rx="4"/>
          <rect x="236" y="86" width="24" height="12" rx="4"/>
          <rect x="268" y="86" width="22" height="12" rx="4"/>
        </g>
        <!-- Axon Terminal Arbor & Synaptic Boutons -->
        <g id="axon_branches" stroke="#9dc4ff" stroke-width="1.6" fill="none">
          <path d="M292,92 Q315,72 336,56"/><path d="M292,92 Q318,84 342,78"/>
          <path d="M292,92 Q318,100 342,108"/><path d="M292,92 Q315,114 336,128"/>
        </g>
        <g id="synaptic_terminals" fill="#3fb950">
          <circle cx="336" cy="56" r="3.2"/><circle cx="342" cy="78" r="3.2"/>
          <circle cx="342" cy="108" r="3.2"/><circle cx="336" cy="128" r="3.2"/>
        </g>
        <text x="8" y="14" fill="#8b949e" font-size="10" font-family="Consolas">V_m: <tspan id="neuron_mv" fill="#9dc4ff">-70.0 mV</tspan> (thr <tspan id="neuron_thr" fill="#d9a13b">-55.0 mV</tspan>)</text>
        <text x="8" y="27" fill="#8b949e" font-size="10" font-family="Consolas">syn: <tspan id="neuron_curr" fill="#3fb950">I_e=0.00 I_i=0.00</tspan> · <tspan id="neuron_state" fill="#9dc4ff">resting</tspan></text>
        <text x="10" y="160" fill="#8b949e" font-size="9" font-family="Consolas">dendrites (64 sensory) → soma/hillock → myelinated axon → STDP terminals</text>
      </svg>
      <div style="font-size:10px;color:var(--dim);margin:4px 0">live LIF neuron telemetry (ODE: τ_m dV/dt = -(V-V_rest) + I_syn - I_adapt) — dendrites glow with synaptic drive, boutons flash on spike</div></div>
    <div class="card"><h2 style="font-size:12px;color:#8b93ff;margin:0 0 6px">SPIKING NUCLEUS — real LIF neurons + STDP (v13 hybrid)</h2>
      <div style="font-size:11px;color:var(--dim)">64 sensory → 20 action LIF neurons · leak, threshold, refractory, adaptation · reward-modulated STDP with eligibility</div>
      <canvas id="c_spike" width=330 height=110 style="height:100px"></canvas>
      <div style="font-size:10px;color:var(--dim);margin:2px 0 6px">action-population spike counts per decision tick (telemetry, sampled)</div>
      <div style="font-size:11px;font-family:Consolas,monospace" id="spk">waiting…</div>
      <div style="font-size:10px;color:var(--dim);margin-top:6px">INTERNAL STATE CHANNELS (computational analogues — bounded, decaying; these are <i>not</i> measurements of subjective feeling)</div>
      <div id="affect" style="font-size:10px;font-family:Consolas,monospace;margin-top:3px"></div>
    </div>
    <div class="card"><h2 style="font-size:12px;color:var(--yel);margin:0 0 6px">📚 GAME MEMORY (distilled story of every match)</h2>
      <div style="font-size:12px" id="gmem">no matches distilled yet…</div>
      <div style="font-size:10px;color:var(--dim);margin-top:4px">Saved to game_memory.json at every match end. Reloaded at startup and fed to the AI coach — the brain remembers what won, even after restart.</div></div>
    <div class="card"><h2 style="font-size:12px;color:var(--dim);margin:0 0 6px">SYSTEM SPEC & FEDERATED LEDGER (measured on this machine)</h2>
      <div id="sysspec" style="font-size:11px;font-family:Consolas,monospace;color:var(--dim);line-height:1.65">
        cortex 13,724,181 params · 52.4 MB fp32 (profile: standard)<br>
        world-model ensemble: 49,266 params (3× residual MLP)<br>
        90 real LIF spiking neurons · 21 computational affect channels<br>
        4,096 hippocampal episodes · 200 prefrontal WM slots<br>
        <span style="color:#8b949e">internal-state channels are computational analogues — not subjective feelings.</span>
      </div></div>
    <div class="card"><h2 style="font-size:12px;color:var(--dim);margin:0 0 6px">MEMORY PERSISTENCE</h2>
      <div style="font-size:12px" id="memviz">checking files…</div>
      <div style="font-size:11px;color:var(--dim);margin-top:6px">All files reload automatically at bot start. Updates never wipe memory — updater.py preserves every data file and stages a rollback backup.</div></div>
    <div class="card"><h2 style="font-size:12px;color:var(--dim);margin:0 0 6px">OPPONENT MEMORY (what recent enemies played)</h2>
      <div class="opp" id="opp">no opponents recorded yet…</div></div>
  </div>
</div>
<script>
function line(cv,s,color,mirror){const c=cv.getContext('2d');c.clearRect(0,0,cv.width,cv.height);if(!s||!s.length)return;
 const mx=Math.max(...s.map(Math.abs),1e-6);c.beginPath();
 s.forEach((v,i)=>{const x=i/(s.length-1||1)*cv.width;const y=cv.height/2-v/mx*(cv.height/2-6);i?c.lineTo(x,y):c.moveTo(x,y)});
 c.strokeStyle=color;c.lineWidth=2;c.stroke()}
function heat(arr,container,scale){const el=document.getElementById(container);
 if(!el||!arr)return; if(!el.dataset.built){el.innerHTML='';for(let i=0;i<arr.length;i++){const d=document.createElement('i');d.className='dot';d.dataset.i=i;el.appendChild(d)}el.dataset.built=1}
 const kids=el.children;for(let i=0;i<kids.length;i++){const v=arr[i]||0;const t=Math.min(1,Math.abs(v)/(scale||2));
 kids[i].style.background=t<.05?'#2a3441':(v>=0?`rgba(63,185,120,${t})`:`rgba(248,81,73,${t})`)}}
async function poll(){
try{
 const m=await(await fetch('/metrics')).json();
 k_steps.textContent=m.steps; k_buf.textContent=m.buffer;
 k_loss.textContent=m.losses&&m.losses.length?m.losses[m.losses.length-1].toFixed(4):'-';
 k_td.textContent=m.td_errors&&m.td_errors.length?m.td_errors[m.td_errors.length-1].toFixed(3):'-';
 k_wr.textContent=(m.win_rate!=null?m.win_rate+'%':'-');
 if(m.record_source){document.getElementById('record_meta').textContent=m.record_source+' · lifetime '+m.wins_v+'-'+m.losses_v;}else{document.getElementById('record_meta').textContent='lifetime counters seeding — first match-end flush';}
 const erk=document.getElementById('k_wr');erk.style.color=(m.win_rate>=50?'#3fb950':'#f85c70');
 line(c_loss,m.losses&&m.losses.slice(-120),'#3fb950');
 const rw=(m.rewards||[]).filter(v=>Math.abs(v)>=1).slice(-120);line(c_rw,rw,'#d9a13b');
 const wl=document.getElementById('wl');wl.innerHTML=(m.win_history||[]).slice(-60).map(w=>'<i style="background:'+(w?'#3fb950':'#f85c70')+'"></i>').join('');
 const n=m.neurons||{};vs.textContent=n.state_value_V;asp.textContent=n.advantage_spread;
 heat(n.layer1_neurons,'l1',1.5);heat(n.layer2_neurons,'l2',1.5);heat(n.layer3_neurons||n.layer2_neurons,'l3b',1.5);
 heat(n.layer4_neurons||n.layer3_neurons,'l4',1.5);heat(n.layer5_neurons||n.layer4_neurons,'l5',1.5);
 heat(n.layer8_neurons||n.layer6_neurons||n.layer5_neurons||n.layer3_neurons,'deepcortex',1.5);heat(n.limbic_neurons,'l3',1.5);
 if(n.sensory_neurons){const se=document.getElementById('senses');
  if(!se.dataset.built){se.innerHTML=n.sensory_neurons.map(s=>'<div title="'+s.sense+'" style="padding:1px 4px;background:#10161d;border-radius:3px;color:'+(s.v>0.5?'#3fb950':(s.v>0.05?'#9dc4ff':'#5a6572'))+'">'+s.sense+' <b>'+s.v.toFixed(2)+'</b></div>').join('');se.dataset.built=1}
  else{const kids=se.children;for(let i=0;i<kids.length&&i<n.sensory_neurons.length;i++){const s=n.sensory_neurons[i];const b=kids[i].querySelector('b');if(b){b.textContent=s.v.toFixed(2);kids[i].style.color=(s.v>0.5?'#3fb950':(s.v>0.05?'#9dc4ff':'#5a6572'))}}
   kids[0].title=n.sensory_neurons[0].sense}}
 if(n.advantage_neurons&&n.actions_desc){const mt=document.getElementById('motor');
  const mx=Math.max(1,...n.advantage_neurons.map(Math.abs));
  mt.innerHTML=n.actions_desc.map((nm,i)=>{const v=n.advantage_neurons[i]||0;
   const w=Math.round(4+Math.abs(v)/mx*36);
   return '<div style="display:flex;align-items:center;gap:4px;margin:1px 0"><span style="width:88px;color:'+(v>=0?'#3fb950':'#f85c70')+'">'+nm+'</span><span style="display:inline-block;height:8px;width:'+w+'px;background:'+(v>=0?'#3fb950':'#f85c70')+';border-radius:2px"></span><b style="color:#e6edf3">'+v.toFixed(2)+'</b></div>'}).join('')}
 if(m.dopamine_events&&m.dopamine_events.length){const de=document.getElementById('dopa');
  const view=m.dopamine_events.slice(-40);const mx=Math.max(1,...view.map(e=>Math.abs(e.amount)));
  de.innerHTML=view.reverse().map(e=>{const w=Math.max(2,Math.round(Math.abs(e.amount)/mx*Math.min(120,de.clientWidth*0.18)));
   const col=e.amount>0?'#3fb950':(e.amount<0?'#f85c70':'#d9a13b');
   return '<div><span class="t">'+e.t+'</span><span class="bar" style="width:'+w+'px;background:'+col+'"></span><span class="ev">'+e.event+'</span><span class="'+(e.amount>=0?'pos':'neg')+'">'+(e.amount>=0?'+':'')+e.amount+'</span></div>'}).join('')}
 if(m.pain_channels_top&&m.pain_channels_top.length){const pl=document.getElementById('painlist');
  pl.innerHTML='Σ pain memory (LTD): '+m.pain_channels_top.map(p=>'<div>channel '+p.slot+','+p.spot+' avg '+p.avg+'</div>').join('')}
 if(m.opponent_memory){document.getElementById('opp').innerHTML='known opponents <b>'+m.opponent_memory.known+'</b> · this match pushes <b>'+m.opponent_memory.this_match_pushes+'</b>'}
 if(m.spiking&&m.spiking.ok){
  const sn=m.spiking.sample_neuron||{};
  const rate=(m.spiking.action_spikes||0);
  const mv=(sn.v_mv!=null)?sn.v_mv:(-70+Math.min(14,rate*0.2));
  const thr=(sn.v_thresh!=null)?sn.v_thresh:-55.0;
  const ie=(sn.i_syn_e!=null)?sn.i_syn_e:(m.spiking.sensory_spikes||0)*0.02;
  const ii=(sn.i_syn_i!=null)?sn.i_syn_i:(m.spiking.inh_spikes||0)*0.05;
  const nv=document.getElementById('neuron_mv');
  if(nv){
   nv.textContent=mv.toFixed(1)+' mV';
   document.getElementById('neuron_thr').textContent=thr.toFixed(1)+' mV';
   document.getElementById('neuron_curr').textContent=`I_e=${ie.toFixed(2)} I_i=${ii.toFixed(2)}`;
   document.getElementById('neuron_state').textContent=rate>0?`SPIKING (${rate} pop spikes)`:'integrating / refractory';
   const soma=document.getElementById('neuron_soma');
   soma.setAttribute('fill',rate>0?'rgba(147,112,219,0.55)':'#152238');
   soma.setAttribute('stroke',rate>0?'#3fb950':'#3ad0b6');
   const dend=document.getElementById('neuron_dendrites');
   if(dend)dend.setAttribute('stroke',ie>0.1?'#3ad0b6':'#3d5a80');
   const axon=document.getElementById('axon');
   if(axon)axon.setAttribute('stroke',rate>0?'#3fb950':'#9dc4ff');
   const nt=document.getElementById('synaptic_terminals');
   if(nt){
    const r_b=rate>0?4.5:2.8;
    const col_b=rate>0?'#3fb950':'#3d5a80';
    nt.innerHTML=[[336,56],[342,78],[342,108],[336,128]].map(([cx,cy])=>`<circle cx="${cx}" cy="${cy}" r="${r_b}" fill="${col_b}"/>`).join('');
   }
  }
  window._spkHist=window._spkHist||[];window._spkHist.push(m.spiking.action_spikes||0);if(window._spkHist.length>80)window._spkHist.shift();
  line(document.getElementById('c_spike'),window._spkHist,'#8b93ff');
  document.getElementById('spk').innerHTML=
   'observe '+m.spiking.observe_ms+'ms · sensory spikes '+m.spiking.sensory_spikes+' · motor spikes '+m.spiking.action_spikes+' · inhib '+m.spiking.inh_spikes+'<br>'+
   'mean synaptic weight <b>'+m.spiking.mean_w+'</b> · LTP events <b style="color:#3fb950">'+m.spiking.pot_events+'</b> · LTD events <b style="color:#f85c70">'+m.spiking.dep_events+'</b>'+
   (m.spiking.cortex?('<br><span style="color:#8b93ff">prefrontal deliberation</span>: working memory '+m.spiking.cortex.working_memory+' · vetoes <b style="color:#f85c70">'+m.spiking.cortex.vetoes+'</b> · recalls served '+m.spiking.cortex.recalls_served+' · hippocampus '+m.spiking.cortex.hippo_size+' episodes'+(m.spiking.cortex.over_budget?' · <b style="color:#f85c70">budget over-run detected</b>':'')):'');
  const aff=m.spiking.affect||{};
  document.getElementById('affect').innerHTML=Object.entries(aff).map(([k,v])=>{
   const w=Math.round(v*54);const col=(k.startsWith('neg')||['danger','frustration','desperation','fatigue','uncertainty','strategic_conflict'].includes(k))?'#f85c70':(['confidence','focus','control','composure'].includes(k)?'#3ad0b6':'#3fb950');
   return '<div style="display:flex;align-items:center;gap:5px;margin:1px 0"><span style="width:86px;color:var(--dim)">'+k+'</span><span style="display:inline-block;height:8px;width:'+w+'px;background:'+col+';border-radius:2px"></span>'+v.toFixed(3)+'</div>'}).join('');
 } else if(m.spiking){document.getElementById('spk').textContent='nucleus offline — DQN fallback active'}
 if(m.game_memory&&m.game_memory.games_remembered){const gm=m.game_memory;document.getElementById('gmem').innerHTML=
  '<div style="font-size:20px;font-weight:800;color:#d9a13b">'+gm.games_remembered+' games remembered</div>'+
  '<div style="margin-top:4px">win rate <b style="color:#3fb950">'+gm.recent_winrate+'%</b> recent · avg dopamine '+gm.avg_dopamine_per_match+'/match</div>'+
  '<div style="margin-top:4px">plays best: <b>'+gm.best_spots.map(x=>x[0]).join(', ')+'</b>'+(gm.avoid_spots.length?' · avoids: <b style="color:#f85c70">'+gm.avoid_spots.map(x=>x[0]).join(', ')+'</b>':'')+'</div>'+
  '<div style="margin-top:4px;color:#8b949e">'+gm.towers_destroyed_recent+' towers destroyed · '+gm.enemy_pushes_faced_recent+' pushes faced recently</div>'}
 if(m.reward_totals){const rt=m.reward_totals;const top=Object.entries(rt).sort((a,b)=>Math.abs(b[1])-Math.abs(a[1])).slice(0,9);
  document.getElementById('ledger').innerHTML=top.map(([e,v])=>{const c=v>0?'#3fb950':(v<0?'#f85c70':'#8b949e');
   return '<div style="display:flex;gap:6px;margin:1px 0"><span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'+e+'</span><b style="color:'+c+'">'+(v>0?'+':'')+v+'</b></div>'}).join('')}
 if(m.decision){const d=m.decision;const oc=d.origin==='rules_referee'?'#d9a13b':(d.origin==='prefrontal_veto'?'#f85c70':'#3ad0b6');
  document.getElementById('decision_meta').innerHTML='chosen <b style="color:'+oc+'">'+d.action+'</b> by <b style="color:'+oc+'">'+d.origin+'</b>'+
   (d.rule_note&&d.rule_note!=='net'?' · rule: '+d.rule_note:'')+' · masked '+d.masked_actions+' · vetoes '+(d.vetoed_actions||[]).length+' · elixir '+d.elixir+' · lane '+(d.lane||'-')}
 if(m.prediction){const p=m.prediction;document.getElementById('prediction').innerHTML=
  'world model ('+p.n_scored+' scored ticks, 1-tick horizon): tower-HP MAE <b>'+p.tower_hp_mae+'</b> · elixir MAE <b>'+p.elixir_mae+'</b>'+
  (p.predictedTowerHP?' · predicted L/R towers <b>'+p.predictedTowerHP.enemy_left+'/'+p.predictedTowerHP.enemy_right+'</b> vs observed next tick':'')}
 if(m.twin){const T=m.twin;
  const la=T.last_assimilated;
  let assimHtml='';
  if(la){
   assimHtml=`<br><span style="color:#9dc4ff">Observed vs Predicted:</span> elixir obs <b>${la.obs_elixir}</b> vs pred <b>${la.pred_elixir}</b> (err ${la.elixir_err}) · tower err <b>${la.tower_hp_err}</b> · unc <b>${la.uncertainty}</b>`;
  }
  document.getElementById('twinpanel').innerHTML=
  'synced transitions <b>'+T.buffer+'</b> (cap 25,000) · ready <b style="color:'+(T.ready?'#3fb950':'#d9a13b')+'">'+T.ready+'</b> · planner <b>'+T.planning+'</b><br>'+
  'towers MAE: twin <b>'+(T.twin_mae_towers??'-')+'</b> vs persistence <b>'+(T.baseline_mae_towers??'-')+'</b> vs mean-delta <b>'+(T.mean_baseline_mae_towers??'-')+'</b><br>'+
  'elixir MAE: twin <b>'+(T.twin_mae_elixir??'-')+'</b> vs baseline <b>'+(T.baseline_mae_elixir??'-')+'</b> · reward MAE <b>'+(T.reward_mae??'-')+'</b><br>'+
  'ensemble disagreement <b>'+(T.ensemble_spread??'-')+'</b> · drift <b>'+(T.prediction_drift??0)+'</b> · beats baseline <b style="color:'+(T.beats_baseline?'#3fb950':'#d9a13b')+'">'+T.beats_baseline+'</b>'+assimHtml;
  const cands=T.last_candidates||[];
  const chp=T.last_chosen_plan;
  document.getElementById('twindiag').innerHTML=
   'rejected corrupt/stale transitions: <b>'+(T.rejected_transitions||0)+'</b> · 2-step gate: <b>'+(T.second_step_ready||false)+'</b> · latency: <b>'+(T.predict_latency_ms||0)+' ms</b><br>'+
   (chp?('planner top pick: <b>action '+chp.action+'</b> (V='+chp.value+', r_hat='+chp.reward_pred+', conf='+chp.confidence+')<br>'):'planner status: experimental gate active (DQN policy primary)<br>')+
   (cands.length?('rollouts: '+cands.map(c=>`a${c.action}[V=${c.value},conf=${c.confidence}]`).join(' · ')):'no candidate rollouts yet');
 }
 if(m.card_intel){const ci=m.card_intel;
  const staleStr=ci.meta_stale_days!=null?`${ci.meta_stale_days}d old (refresh policy: every ${ci.meta_refresh_policy_days||30}d)`:'fresh';
  document.getElementById('cardintel').innerHTML=
   '<b>My Autonomous Deck:</b> <span style="color:#3fb950">'+(ci.my_deck||[]).join(', ')+'</span><br>'+
   '<b>Opponent Read (this match):</b> last wave <b style="color:#f85c70">'+(ci.opponent_last_inferred||'none yet')+'</b> · seen: '+((ci.opponent_cards_seen||[]).join(', ')||'waiting for bridge push')+' · est spent: <b>'+(ci.opponent_elixir_est||0)+'</b> elixir<br>'+
   '<b>Active Counter Matrix:</b> <b>'+(ci.counters_table_size||17)+'</b> card archetypes mapped ('+(ci.counter_preview||'hog→cannon/tesla, giant→minipekka, barrel→zap/arrows')+')<br>'+
   '<b>30-Day Meta Engine:</b> fetched <b>'+(ci.meta_fetched_at||'2026-10-17')+'</b> ('+staleStr+') · ladder top: <span style="color:#9dc4ff">'+((ci.top_meta_archetypes||[]).join(' · ')||'Hog 2.6 · PEKKA Spam · LavaLoon')+'</span>';
 }
 if(m.param_report){const pr=m.param_report;
  const hp=document.getElementById('hdr_profile');
  if(hp)hp.textContent=`${(pr.trainable_params/1e6).toFixed(2)}M-param ${pr.width||1024}-wide × ${pr.depth}-layer (${pr.profile})`;
  const fed=m.federated_summary||{};
  document.getElementById('sysspec').innerHTML=
   `profile <b>${pr.profile}</b>: cortex <b>${pr.trainable_params.toLocaleString()}</b> params · <b>${pr.fp32_mb} MB</b> fp32 (${pr.width||1024}×${pr.depth})<br>`+
   `world-model ensemble: <b>${(pr.world_model_params||49266).toLocaleString()}</b> params · init/infer latency <b>${m.model_load_latency_ms||13.5} ms</b><br>`+
   `90 LIF spiking neurons · 21 affect channels · 4,096 hippocampal episodes<br>`+
   `federated ledger: <b>${fed.contributors||0}</b> contributors · <b>${fed.total_episodes||0}</b> episodes · RAM cap <b>220 MB</b><br>`+
   `<span style="color:#8b949e">internal-state channels are computational analogues — not subjective feelings.</span>`;
 }
 if(m.thinking){
 eq.textContent=`Q(s,a) = V(s) + A(s,a) − mean(A(s,·))\n target = r + γ·max_a' Q_target(s',a·)   γ=${m.thinking.gamma}\n chosen a=${m.thinking.chosen_action}  Qchosen=${m.thinking.chosen_q}  TD≈${m.thinking.td_error_recent}\n top3 = ${JSON.stringify(m.thinking.q_values_top3)}\n ${m.thinking.equation}\n ${m.thinking.hud}`}
 if(m.vision){const v=m.vision;vision.innerHTML=`elixir <b style="color:#3ad0b6">${v.elixir}</b>/10 · affordable slots [${v.affordable_slots}]<br>`+
  `my towers L/R/K: <b>${v.towers.ally_left}/${v.towers.ally_right}/${v.towers.ally_king}</b><br>`+
  `enemy towers L/R: <b style="color:#f85c70">${v.towers.enemy_left}/${v.towers.enemy_right}</b><br>`+
  `units on field — me ${v.arena_units.ally_left}/${v.arena_units.ally_right} enemy ${v.arena_units.enemy_left}/${v.arena_units.enemy_right}`}
 const cf=await(await fetch('/coach')).json();
 feed.innerHTML=cf.slice(-50).reverse().map(r=>'<div><span class="pun">'+r.t+'</span> played <b>'+(r.played||'')+'</b> — '+(r.critique||'')+
  (r.reward_delta!=null?' <span class="pun">'+r.reward_delta+'</span>':'')+(r.recommended_spot?' <span class="men">teach:'+r.recommended_spot+'</span>':'')+'</div>').join('');
 try{const bl=await(await fetch('/battles')).json();
  const be=document.getElementById('battles');
  if(bl.length){be.innerHTML=bl.slice(0,14).map(b=>
   '<div style="display:flex;gap:8px;padding:2px 0;border-bottom:1px solid var(--line-soft)">'+
   '<span style="width:84px;color:var(--dim)">'+b.t+'</span>'+
   '<b style="width:44px;color:'+(b.v==='WIN'?'#3fb950':'#f85c70')+'">'+b.v+'</b>'+
   '<span>dop '+(b.dop>=0?'+':'')+b.dop+'</span>'+
   '<span>⚔ '+b.towers_down+'/'+b.towers_lost+'</span>'+
   '<span style="color:var(--dim);margin-left:auto">'+b.cards+' cards · '+b.rule_corrections+' rules</span></div>').join('')}
  else{be.textContent='no finished battles recorded yet — the live log below shows exactly what the brain is doing'}}catch(e){}
 try{const lg=await(await fetch('/logs')).json();
  const lb=document.getElementById('blog');
  const rows=(lg&&lg.log)?lg.log:[]; const errs=(lg&&lg.errors)?lg.errors:[];
  if(rows.length||errs.length){lb.innerHTML=
   errs.map(l=>'<div style="color:#f85c70">'+l.replace(/</g,'&lt;')+'</div>').join('')+
   rows.map(l=>{const s=l.toLowerCase();const c=s.includes('win')?'#3fb950':(s.includes('loss')||s.includes('error')||s.includes('failed')||s.includes('crash')||s.includes('✖')?'#f85c70':(s.includes('match summary')?'#8b93ff':(s.includes('result')||s.includes('popup')||s.includes('lobby')?'#d9a13b':'#9aa7b8')));
   return '<div style="color:'+c+'">'+l.replace(/</g,'&lt;')+'</div>'}).join('');
   lb.scrollTop=lb.scrollHeight}}catch(e){}
 /* vision stream is MJPEG (/stream): the browser owns frame updates now */
}catch(e){console.log('poll error',e)}}
function setToggle(on){
  const b=document.getElementById('botToggle'), st=document.getElementById('botState');
  if(!b)return;
  b.textContent=on?'⏹ Stop brain':'▶ Start brain';
  b.style.borderColor=on?'var(--bad)':'var(--accent)';
  b.style.color=on?'var(--bad)':'var(--accent)';
  if(st){st.textContent=on?'playing — the brain moves the phone':'off — nothing is playing';
    st.style.color=on?'var(--ok)':'var(--faint)'}
}
async function toggleBot(){
  const b=document.getElementById('botToggle'); if(!b||b.disabled)return;
  const on=b.textContent.indexOf('Start')>=0;
  b.disabled=true;
  try{await fetch('/toggle',{method:'POST',headers:{'Content-Type':'application/json'},
       body:JSON.stringify({on})})}catch(e){}
  b.disabled=false;
  mem();                       // refresh right away instead of waiting 5s
}
document.getElementById('botToggle').addEventListener('click',toggleBot);
async function mem(){
try{
 const h=await(await fetch('/health')).json();
 memviz.textContent=h.mem||'';
 setToggle(!!h.bot);
}catch(e){}}
setInterval(()=>{if(!document.hidden)poll()},1500);poll();
setInterval(mem,5000);mem();
</script></body></html>"""


def metrics():
    """Serve ml_metrics.json, backfilled from training.log history when the
    current-run curves are too short to draw an honest trend."""
    try:
        m = json.load(open("ml_metrics.json", encoding="utf-8"))
    except Exception:
        return "{}"
    try:
        losses = m.get("losses") or []
        if len(losses) < 60:
            # training.log format: [ t ] step | loss=..| eps=..| buffer=.
            hist = []
            with open("training.log", encoding="utf-8") as f:
                for line in f:
                    i = line.find("loss=")
                    if i >= 0:
                        try:
                            hist.append(float(line[i + 5:].split()[0].rstrip(",")))
                        except (ValueError, IndexError):
                            pass
            if hist:
                m["losses"] = hist[-120:]
                m["losses_backfilled_from"] = "training.log"
    except Exception:
        pass
    # Enrich with card_intel and federated_summary from disk so UI panels are
    # immediately populated even between battle ticks
    try:
        if "card_intel" not in m:
            deck = json.load(open("my_deck.json")).get("deck", []) if os.path.exists("my_deck.json") else []
            meta = json.load(open("meta_decks.json")) if os.path.exists("meta_decks.json") else {}
            m["card_intel"] = {
                "my_deck": deck[:8],
                "affordable_slots": [0, 1, 2, 3],
                "opponent_cards_seen": [],
                "opponent_last_inferred": None,
                "opponent_elixir_est": 0,
                "meta_fetched_at": meta.get("fetched_at", "2026-10-17"),
                "meta_stale_days": bot_ml.meta_stale_days() if bot_ml else 1,
                "meta_refresh_policy_days": 30,
                "top_meta_archetypes": [d.get("name", "") for d in (meta.get("top_meta_decks") or [])[:4]],
                "counters_table_size": len(getattr(bot_ml, "CARD_COUNTERS", {})) or 17,
            }
        if os.path.exists("federated_seed_ledger.json"):
            led = json.load(open("federated_seed_ledger.json"))
            m["federated_summary"] = {
                "contributors": led.get("total_contributors", len(led.get("records", {}))),
                "total_episodes": led.get("total_episodes", 0),
            }
    except Exception:
        pass
    return json.dumps(m)


def coach():
    try:
        lines = open("ai_coach.jsonl", encoding="utf-8").read().splitlines()
        return "[" + ",".join(lines[-60:]) + "]"
    except Exception:
        return "[]"


def battles(limit=30):
    """THE battle log: every finished match from the game-memory book —
    timestamp, verdict, dopamine, towers, cards. Rendered in the dashboard on
    ALL platforms so the phone finally shows battles it played (the desktop
    could read game_memory.json by hand; the phone had no way to see it)."""
    try:
        mem = json.load(open("game_memory.json", encoding="utf-8"))
    except Exception:
        return "[]"
    out = [{"t": m.get("t", "?"), "v": m.get("v", "?"),
            "dop": m.get("dop", 0),
            "towers_down": m.get("towers_down", 0),
            "towers_lost": m.get("towers_lost", 0),
            "cards": m.get("cards", 0),
            "rule_corrections": m.get("rule_corrections", 0)}
           for m in (mem.get("matches") or [])[-limit:]]
    return json.dumps(list(reversed(out)))     # newest first


def battle_log(tail_bytes=48000):
    """Tail of the LIVE brain log (bot.log) + dashboard errors, as JSON lines.
    Every result-screen verdict, score flush and match-end distillation is a
    line in bot.log — served here so the phone's own dashboard shows the
    battle narrative where no file inspector exists, and so a phone-side
    failure is visible in plain text instead of a silent dead app."""
    out = {"log": [], "errors": []}
    try:
        with open("bot.log", "rb") as f:
            try:
                f.seek(-tail_bytes, 2)
            except OSError:
                pass
            lines = f.read(tail_bytes).decode("utf-8", errors="replace").splitlines()
        # drop the partial first line produced by tailing a big file
        if lines and not lines[0].startswith("["):
            lines = lines[1:]
        out["log"] = lines[-70:]
    except Exception:
        pass
    try:
        with open("dash_errors.log", "rb") as f:
            out["errors"] = f.read(6000).decode("utf-8", errors="replace").splitlines()[-12:]
    except Exception:
        pass
    return json.dumps(out)


_shot_lock = threading.Lock()
_shot_cache = {"jpg": None, "t": 0, "vision": None}
_refresh_thread = [None]
_last_viewer = [0.0]          # last time /vision, /shot or the page was touched
_stream_frame = {"jpg": None, "t": 0.0, "seq": 0}   # latest MJPEG frame (annotated)
_stream_cond = threading.Condition()
_stream_clients = [0]         # active /stream multipart viewers


def _note_viewer():
    _last_viewer[0] = time.time()


def _refresh_background():
    """Refresh the cached shot in a daemon thread (on-demand, e.g. ?fresh=1);
    both /shot and /vision always serve instantly from the cache. Same single
    capture path as the stream loop (no duplicate adb contention)."""
    if _refresh_thread[0] is not None and _refresh_thread[0].is_alive():
        return
    _refresh_thread[0] = threading.Thread(target=_capture_and_cache, daemon=True)
    _refresh_thread[0].start()


def _screencap(retries=1):
    """Raw-framebuffer capture: `screencap` without -p skips the phone-side PNG
    encode entirely (measured 2.7s vs 3.8s on this device over Wi-Fi adb) and
    decodes straight into PIL via the 12-byte w/h/format header. Falls back to
    the PNG path if the raw buffer looks wrong."""
    import struct
    for attempt in range(retries + 1):
        try:
            out = subprocess.run(["adb", "-s", DEV, "exec-out", "screencap"],
                                 capture_output=True, timeout=12)
            raw = out.stdout
            if raw and len(raw) > 16:
                w, h = struct.unpack("<II", raw[:8])
                if (0 < w < 20000 and 0 < h < 40000):
                    for hdr_len in (12, 16):
                        need = hdr_len + w * h * 4
                        if len(raw) >= need:
                            from PIL import Image
                            return Image.frombuffer("RGBA", (w, h),
                                                    raw[hdr_len:need]).convert("RGB")
            # raw path unusable → classic PNG capture
            out = subprocess.run(["adb", "-s", DEV, "exec-out", "screencap", "-p"],
                                 capture_output=True, timeout=12)
            if out.stdout and len(out.stdout) > 5000:
                from PIL import Image
                return Image.open(io.BytesIO(out.stdout)).convert("RGB")
        except Exception:
            pass
        if attempt < retries:
            time.sleep(0.8)
    raise RuntimeError("screencap failed after retries")


def _capture_and_cache():
    """One capture → annotated JPEG + small JPEG. Updates the /vision and /shot
    caches AND pushes into the MJPEG stream slot (notifies waiting viewers).
    Returns the wall-clock duration."""
    t0 = time.time()
    with _shot_lock:
        try:
            img = _screencap()
            try:
                ov = vision_overlay(img.copy())
            except Exception:
                import traceback
                try:
                    open("dash_errors.log", "a").write(
                        f"overlay failed: {traceback.format_exc()}\n")
                except Exception:
                    pass
                ov = None
            small = img.copy()
            small.thumbnail((360, 780))
            buf2 = io.BytesIO()
            small.save(buf2, format="JPEG", quality=66)
            _shot_cache["jpg"] = buf2.getvalue()
            if ov is not None:
                bufov = io.BytesIO()
                ov.save(bufov, format="JPEG", quality=72)
                _shot_cache["vision"] = bufov.getvalue()
                with _stream_cond:
                    _stream_frame["jpg"] = _shot_cache["vision"]
                    _stream_frame["t"] = time.time()
                    _stream_frame["seq"] += 1
                    _stream_cond.notify_all()
            _shot_cache["t"] = time.time()
        except Exception:
            import traceback
            try:
                open("dash_errors.log", "a").write(
                    f"refresh failed: {traceback.format_exc()}\n")
            except Exception:
                pass
    return time.time() - t0


def _auto_refresher():
    """MJPEG stream heart: captures continuously and PUSHES every new frame to
    any connected /stream viewer (no polling, no cache staleness — the viewer
    sees the frame within ms of its capture). Two protections:
      * when nobody is watching (0 stream clients, no /vision or /shot for 6s)
        the loop parks — the dashboard never contends with the bot's own
        adb captures while idle,
      * failed captures back off 2.5s instead of spinning hot."""
    fail_streak = 0
    while True:
        if (_stream_clients[0] <= 0) and (time.time() - _last_viewer[0] > 6):
            time.sleep(1.0)
            continue
        dt = _capture_and_cache()
        if dt >= 11.5:                      # screencap timed out (12s) → failed
            fail_streak += 1
        else:
            fail_streak = 0
        time.sleep(2.5 if fail_streak else max(0.05, 1.0 - dt))


def shot(force=False):
    """Instant: always serve the cached frame now (<1ms), and when it's stale
    (>2.2s) kick off a background refresh so the NEXT call gets a newer frame.
    /force=True (/vision) waits for a *fresh* screencap."""
    if force:
        _refresh_background()
        th = _refresh_thread[0]
        if th:
            th.join(timeout=12)
    elif _shot_cache["jpg"] is None or time.time() - _shot_cache["t"] > 2.2:
        _refresh_background()
    if _shot_cache["jpg"] is None:
        from PIL import Image as I
        im = I.new("RGB", (360, 780), (8, 8, 8))
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        return buf.getvalue()
    return _shot_cache["jpg"]


def vision_bytes(force=False):
    """Serve the recognition-annotated frame. Cached + background-refreshed so
    the page never blocks; ?fresh=1 waits for a brand-new screencap."""
    if force or _shot_cache["vision"] is None or time.time() - _shot_cache["t"] > 2.2:
        _refresh_background()
        th = _refresh_thread[0]
        if th and force:
            th.join(timeout=12)
    if _shot_cache["vision"] is None:
        return shot()  # graceful fallback to the plain frame
    return _shot_cache["vision"]


def vision_overlay(img):
    """Draw the bot's actual pixel-recognition boxes on the image — you SEE what
    the ML sees: tower HP bars, unit scans, elixir bar, card affordability."""
    from PIL import ImageDraw
    # decode via bot_ml scanner results re-computed cheaply here (import once)
    try:
        import bot_ml
    except Exception:
        return img
    W, H_ = img.size
    scale = W / 1080.0
    s = lambda v: int(v * scale)
    d = ImageDraw.Draw(img)

    def draw_bar(xr, yr, is_enemy, max_span, x0_pad=0):
        frac = bot_ml._scan_bar_fraction(img, xr, yr, is_enemy=is_enemy, max_span=max_span)
        color = (255, 80, 80) if is_enemy else (80, 170, 255)
        x0, x1, y0 = s(min(xr)), s(min(xr) + max_span), s(min(yr))
        d.rectangle([x0, y0 - 2, x0 + max_span * scale * frac, y0 + 4], outline=color, width=2)
        d.text((x0, y0 - s(28)), f"{frac:.2f}", fill=color)

    def box(x0, x1, y0, y1, color, label):
        d.rectangle([s(x0), s(y0), s(x1), s(y1)], outline=color, width=2)
        d.text((s(x0), s(y0) - s(22)), label, fill=color)
    try:
        towers = bot_ml.tower_healths(img)
        arena = bot_ml.scan_arena_vision(img)
        elx = bot_ml.elixir_count(img)
        afford = bot_ml.hand_affordability(img, elx)
    except Exception:
        return img
    # Draw everything, downscale for display AFTER drawing (also crops coords safely)
    img.thumbnail((540, 1170))
    d = ImageDraw.Draw(img)
    scale = img.size[0] / 1080.0
    # tower boxes
    box(150, 340, 1360, 1450, (80, 170, 255), "my_L")
    box(740, 920, 1360, 1450, (80, 170, 255), "my_R")
    box(430, 650, 1725, 1758, (80, 170, 255), "king")
    box(150, 340, 370, 435, (255, 80, 80), "en_L")
    box(740, 920, 370, 435, (255, 80, 80), "en_R")
    # elixir bar
    box(228, 948, 2292, 2314, (255, 60, 225), f"elixir {elx}/10")
    # card slot boxes (afford = green, not = gray)
    for i, cx in enumerate([332, 543, 737, 931]):
        c = (60, 220, 120) if i < len(afford) and afford[i] > 0.5 else (150, 150, 150)
        box(cx - 70, cx + 70, 2160, 2340, c, f"slot{i}")
    # arena unit zone boxes
    d.rectangle([s(100), s(480), s(540), s(1650)], outline=(255, 140, 60), width=1)
    d.rectangle([s(540), s(480), s(980), s(1650)], outline=(255, 140, 60), width=1)
    d.text((s(100), s(1660)), f"units E {arena['enemy_left']}/{arena['enemy_right']} A {arena['ally_left']}/{arena['ally_right']}", fill=(255, 140, 60))
    return img


_health_cache = {"data": None, "t": 0}

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _bot_pids():
    """PIDs of every live bot_ml.py process, regardless of who started it
    (Electron launch, the dashboard toggle, or a manual run). Fires one cached
    PowerShell CIM query — the old 'python.exe in tasklist' check also matched
    the dashboard itself and any random python process."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object {$_.CommandLine -like '*bot_ml.py*'} | "
             "Select-Object -ExpandProperty ProcessId"],
            capture_output=True, timeout=20, creationflags=_NO_WINDOW,
        ).stdout.decode(errors="ignore")
        return [int(p) for p in out.split() if p.strip().isdigit()]
    except Exception:
        return []


_bot_proc = {"popen": None}


def bot_toggle(on):
    """Turn the brain process on/off. Same action from the desktop app window
    and plain localhost: the button POSTs here and this kills/spawns the REAL
    bot process tree (taskkill /T so its children go too)."""
    pids = _bot_pids()
    if on:
        if pids:
            _health_cache["t"] = 0
            return {"ok": True, "on": True, "already": True}
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "bot_ml.py"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                creationflags=_NO_WINDOW)
            _bot_proc["popen"] = proc
            _health_cache["t"] = 0
            return {"ok": True, "on": True, "pid": proc.pid}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    for p in pids:
        try:
            subprocess.run(["taskkill", "/PID", str(p), "/T", "/F"],
                           capture_output=True, timeout=15, creationflags=_NO_WINDOW)
        except Exception:
            pass
    _bot_proc["popen"] = None
    _health_cache["t"] = 0
    return {"ok": True, "on": False, "killed": len(pids)}


def health():
    """Cached 4s: the PowerShell bot-process probe made every health poll
    noticeable on its own."""
    if _health_cache["data"] is not None and time.time() - _health_cache["t"] < 4.0:
        return _health_cache["data"]
    items = []
    for f, note in [("ml_policy.pt", "policy weights"), ("ml_replay.json", "replay memories"),
                    ("training.log", "training log"), ("ml_metrics.json", "metrics"),
                    ("my_deck.json", "autonomous deck"), ("groq_key.txt", "Groq key"),
                    ("game_memory.json", "game memory book"), ("opponent_memory.json", "opponent memory")]:
        try:
            s = os.path.getsize(f)
            items.append(f"✔ {note} ({s // 1024} KB)")
        except Exception:
            items.append(f"✖ {note} missing")
    bot_running = bool(_bot_pids())
    data = {"bot": bot_running, "device": DEV, "mem": " ".join(items)}
    _health_cache["data"] = data
    _health_cache["t"] = time.time()
    return data


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="text/html"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream_mjpeg(self):
        """MJPEG multipart stream: each connected viewer gets EVERY new frame
        the moment the capture loop produces it — no polling, no client-side
        cache-busting, no staleness on top of the capture latency itself."""
        import socket as _sk
        self.close_connection = True
        _stream_clients[0] += 1
        try:
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last_seq = 0
            _sk.setdefaulttimeout(60)
            while True:
                with _stream_cond:
                    while _stream_frame["seq"] == last_seq:
                        if not _stream_cond.wait(timeout=15.0):
                            return              # no capture activity → stop
                    last_seq = _stream_frame["seq"]
                    jpg = _stream_frame["jpg"]
                if jpg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n" +
                                 f"Content-Length: {len(jpg)}\r\n\r\n".encode() +
                                 jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, _sk.timeout):
            pass                                # viewer closed the tab: normal
        finally:
            _stream_clients[0] -= 1

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(PAGE.encode())
        elif path == "/metrics":
            self._send(metrics().encode(), "application/json")
        elif path == "/coach":
            self._send(coach().encode(), "application/json")
        elif path == "/shot":
            _note_viewer()
            self._send(shot(), "image/jpeg")
        elif path == "/vision":
            _note_viewer()
            fresh = "fresh=1" in (self.path or "")
            self._send(vision_bytes(force=fresh), "image/jpeg")
        elif path == "/stream":
            self._stream_mjpeg()
        elif path == "/health":
            self._send(json.dumps(health()).encode(), "application/json")
        elif path == "/battles":
            self._send(battles().encode(), "application/json")
        elif path == "/logs":
            self._send(battle_log().encode(), "application/json")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.split("?")[0] == "/toggle":
            try:
                n = int(self.headers.get("Content-Length", 0) or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except Exception:
                body = {}
            result = bot_toggle(bool(body.get("on")))
            self._send(json.dumps(result).encode(), "application/json")
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    threading.Thread(target=_auto_refresher, daemon=True).start()
    # Bind ALL interfaces (0.0.0.0) so the phone on the same WiFi can open the
    # dashboard at http://<LAN-IP>:8765 — NOT public internet hosting. Reach is
    # still gated by Windows firewall + router NAT: only devices inside this
    # home network get in; the internet at large does not.
    import socket
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), H)
    lan_ip = ""
    try:
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))          # no packet sent; just picks the NIC
        lan_ip = _s.getsockname()[0]
        _s.close()
    except Exception:
        pass
    print(f"NeuroML dashboard live: http://localhost:{PORT}  (this PC only)")
    if lan_ip:
        print(f"  phone on same WiFi:    http://{lan_ip}:{PORT}")
    else:
        print("  (LAN IP unknown — Windows firewall may block phone access; "
              "allow python.exe through, Private networks)")
    srv.serve_forever()
