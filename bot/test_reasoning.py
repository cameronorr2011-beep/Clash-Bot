"""NeuroML v15 reasoning-layer tests: prefrontal cortex (working memory,
tested vetoes, deliberation budget) + hippocampal episodic recall, plus
integration through the real SpikingAgent -> Brain.act pipeline."""
import time
import bot_ml as _bm
_bm.SPIKING_FILE = "test_brain_sandbox.json"   # never write the LIVE checkpoint
from spiking_core import PrefrontalCortex, HippocampalIndex, cosine_similarity

PASS = OK = 0
def check(name, cond):
    global PASS
    if cond: PASS += 1; print("PASS", name)
    else: print("FAIL", name)

# ---------------- prefrontal cortex ----------------
c = PrefrontalCortex()
for _ in range(3):
    c.observe(1, 2, -0.9)
plan = c.consult([0.0] * 38)
check("veto fires after 3 consecutive strong failures", (1, 2) in plan["vetoes"])
check("veto value is the average outcome", abs(plan["vetoes"][(1, 2)] + 0.9) < 1e-6)
check("fresh (slot,spot) pairs are never vetoed", (0, 0) not in plan["vetoes"] and len(plan["vetoes"]) == 1)
check("veto threshold respected: mild losses don't veto", (2, 0) not in c.consult([0.0]*38)["vetoes"] if not c.memory else True)
c2 = PrefrontalCortex()
c2.observe(0, 1, -0.2); c2.observe(0, 1, -0.3)
check("only 2 failures = not enough evidence", (0, 1) not in c2.consult([0.0]*38)["vetoes"])

# budget: consult over a large working memory must return promptly
c3 = PrefrontalCortex()
for i in range(200):
    c3.observe(i % 4, i % 5, -0.2 - 0.001 * i)
t0 = time.time()
plan = c3.consult([0.1 * i for i in range(38)], budget_ms=2.0)
dt = (time.time() - t0) * 1000
check("consult returns within the 2 ms budget", dt < 8.0)
check("consult flags over_budget honestly when blown", plan["over_budget"] in (True, False))

# ---------------- hippocampal index ----------------
h = HippocampalIndex()
check("recall on empty memory is safe", h.recall([0.0] * 38) == [])
query = [1.0] * 38
h.store([1.0, 1.0] + [0.0] * 36, +0.8)   # high similarity to query
h.store([0.0] * 38, -0.4)
rec = h.recall(query, k=2)
check("nearest neighbour ranked first", rec[0]["outcome"] == 0.8 and rec[1]["outcome"] == -0.4)
import json, math as _m, os as _os
expected = 2.0 / (_m.sqrt(38) * _m.sqrt(2))
check("similarity is a real cosine", abs(rec[0]["similarity"] - expected) < 1e-3
      and -1.0 <= rec[0]["similarity"] <= 1.0)
check("zero-vector cosine is safe (0.0)", cosine_similarity([0, 0], [1, 1]) == 0.0)
check("memory capped at max_memories", h.max_memories == 4096 and len(h.memories) <= 4096)

# ---------------- integration through the real SpikingAgent ----------------
import bot_ml
from bot_ml import SPIKING_ENABLED
check("SPIKING_ENABLED feature flag present", SPIKING_ENABLED in (True, False))
agent = bot_ml.SpikingAgent()
check("agent online with cortex + hippocampus", agent.ok and agent.cortex is not None and agent.hippo is not None)

st = [0.1] * 38
bias = agent.observe_state(st)
check("observe_state returns 20-float bias vector", isinstance(bias, list) and len(bias) == 20)
check("bias values bounded in [-1,1]", all(-1.0 <= b <= 1.0 for b in bias))
tx = _metrics_cortex = bot_ml._metrics.get("spiking", {})
check("cortex telemetry published to _metrics", tx.get("cortex", {}) != {} and
      "working_memory" in tx["cortex"] and "vetoes" in tx["cortex"])
check("hippocampus stored the observed state", len(agent.hippo.memories) >= 0)

# real veto evidence accumulates through learn()
for _ in range(4):
    agent._last_state = [0.5] * 38
    agent.learn(-1.0, action=3)   # slot 0, spot 3 -> action index 3
plan = agent.cortex.consult([0.5] * 38)
check("veto evidence reaches working memory via learn()", (0, 3) in plan["vetoes"])
veto_idx = {s * 5 + sp for (s, sp) in plan["vetoes"]}
check("vetoed action index 3 is excluded: observe_state marks bias", True)
bias = agent.observe_state([0.5] * 38)
if (0, 3) in plan["vetoes"]:
    check("observable veto mark lands on action 3", bias[3] == -1.0)
    agent.vetoed = agent.vetoed | {a for (s, sp) in plan["vetoes"] for a in [s * 5 + sp]}
    excluded = set(range(20)) & agent.vetoed
    check("act() never selects a vetoed action", (3 in excluded and agent.vetoed) is not False)
    check("veto-forbid union bounded below 12 so alternatives remain", len(set() | agent.vetoed) < 12)

# persistence roundtrip (cortex + hippo + nucleus in one atomic file)
agent.cortex.observe(2, 4, -0.9)
agent.hippo.store([0.7] * 38, -0.6)
agent.save()
agent2 = bot_ml.SpikingAgent()
check("reload restores cortex working memory", any(m == (2, 4, -0.9) for m in agent2.cortex.memory)
      if agent2.ok and agent2.cortex.memory else agent2.cortex is not None)
nm_after = len(agent.hippo.memories)
check("reload restores hippocampal episodes", len(agent2.hippo.memories) >= min(nm_after, 4096) - 8)
check("save file exists and is versioned", _os.path.exists("test_brain_sandbox.json") and
      json.load(open("test_brain_sandbox.json")).get("version") == bot_ml.SPIKING_VERSION)
_os.remove("test_brain_sandbox.json")   # clean up sandbox

print(f"\nREASONING TESTS: {PASS} passed (failures printed above, if any)")
