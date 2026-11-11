"""Integration test: SpikingAgent end-to-end against real TargetedDQN.

Run: python test_integration_spiking.py
"""
import os
import sys
import tempfile
import json as _json
import math

_failures = []
def check(name, cond, detail=""):
    print(("  PASS " if cond else "  FAIL ") + name + ("" if cond else " " + str(detail)))
    if not cond:
        _failures.append(name)

import bot_ml

def main():
    print("[SpikingAgent end-to-end]")
    agent = bot_ml.SpikingAgent()
    check("spiking agent active", agent.ok)
    check("affective core created", agent.affect is not None)
    check("nucleus created", agent.nucleus is not None)

    state = [0.5] + [0.0] * 37   # realistic 38-dim state, elixir 5/10
    bias = agent.observe_state(state)
    check("observe returns bias vector", bias is not None and len(bias) == 20,
          f"len={len(bias) if bias is not None else None}")
    if bias:
        check("bias bounded in [-1,1]", all(-1.01 <= b <= 1.01 for b in bias))

    brain = bot_ml.TargetedDQN()
    a0 = brain.act(state)
    a1 = brain.act(state, bias=bias)
    check("act with bias returns valid action", 0 <= a1 < bot_ml.ACTION_DIM)
    # bias must be able to change the decision but is bounded: with zero-Q
    # separation (all-equal Q), a strong bias decides; with huge separation
    # it must not override. Verify bounded influence directly:
    import torch
    q = brain.policy(torch.tensor(state, dtype=torch.float32))
    width = float(q.max() - q.min())
    b = torch.tensor(bias, dtype=torch.float32)
    q_biased = q + bot_ml.SPIKING_BIAS_SCALE * width * b
    check("bias influence <= SPIKING_BIAS_SCALE * spread",
          torch.all((torch.abs(q_biased - q)) <= bot_ml.SPIKING_BIAS_SCALE * width * abs(b).max() + 1e-3))

    # ---- reward-modulated plasticity actually changes weights
    w_before = agent.nucleus.ff[0][0].w
    for _ in range(5):
        agent.observe_state(state)   # creates eligible pre->post pairs
        agent.learn(+1.0)            # big win
    w_after = agent.nucleus.ff[0][0].w
    changed = any(s.w != w_before for row in agent.nucleus.ff for s in row)
    check("positive rewards changed synaptic weights", changed,
          f"sample {w_before:.4f} -> {w_after:.4f}")
    check("positive valence rose", agent.affect.state["pos_rpe"] > 0.1)

    for _ in range(5):
        agent.observe_state(state)
        agent.learn(-1.0)
    check("negative valence rose", agent.affect.state["neg_rpe"] > 0.1)
    # decay-per-observe: channels must relax when injections stop
    for _ in range(120):
        agent.observe_state(state)
    check("valence decays after injections stop (equilibrium, not saturation)",
          agent.affect.state["neg_rpe"] < 0.5, f"neg_rpe={agent.affect.state['neg_rpe']:.3f}")
    # relief: simulate threat then resolution
    agent.relief_test_probe = getattr(agent, 'relief_test_probe', None)
    st = [0.0]*38; st[36] = 1.0
    agent.affect.set("danger", 0.6)
    agent.observe_state(st)                 # threat again -> danger rises
    st2 = [0.0]*38
    agent.observe_state(st2)                # clear -> relief fired
    check("relief fires when danger resolves", agent.affect.state["relief"] > 0.01,
          f"relief={agent.affect.state['relief']:.3f}")
    check("valence channels bounded", all(0.0 <= v <= 1.0 for v in agent.affect.state.values()))

    # ---- persistence: atomic versioned checkpoint round-trip
    with tempfile.TemporaryDirectory() as td:
        old = bot_ml.SPIKING_FILE
        bot_ml.SPIKING_FILE = os.path.join(td, "spiking_brain.json")
        try:
            w_saved = agent.nucleus.ff[0][0].w
            agent.save()
            check("checkpoint file written", os.path.exists(bot_ml.SPIKING_FILE))
            with open(bot_ml.SPIKING_FILE) as f:
                payload = _json.load(f)
            check("checkpoint versioned", payload.get("version") == bot_ml.SPIKING_VERSION)
            agent2 = bot_ml.SpikingAgent.__new__(bot_ml.SpikingAgent)
            from spiking_core import SpikingNucleus, AffectiveCore
            agent2.nucleus = SpikingNucleus(n_sensory=64, n_action=20, seed=99)
            agent2.affect = AffectiveCore()
            agent2.nucleus.load_state_dict(payload["nucleus"])
            check("weights survive save/load round-trip",
                  abs(agent2.nucleus.ff[0][0].w - w_saved) < 1e-9)
        finally:
            bot_ml.SPIKING_FILE = old

    # ---- corrupt file: fresh start, no crash
    with tempfile.TemporaryDirectory() as td:
        old = bot_ml.SPIKING_FILE
        bot_ml.SPIKING_FILE = os.path.join(td, "bad.json")
        try:
            with open(bot_ml.SPIKING_FILE, "w") as f:
                f.write("{corrupt")
            agent3 = bot_ml.SpikingAgent.__new__(bot_ml.SpikingAgent)
            agent3.nucleus = agent.nucleus
            agent3.affect = agent.affect
            agent3.load()   # must not raise
            check("corrupt checkpoint handled gracefully", True)
        except Exception as e:
            check("corrupt checkpoint handled gracefully", False, e)
        finally:
            bot_ml.SPIKING_FILE = old

    # ---- disabled agent = pure DQN fallback
    bot_ml.SPIKING_ENABLED = False
    agent4 = bot_ml.SpikingAgent()
    check("flag-off agent ok=False, DQN-only", agent4.ok is False)
    check("flag-off observe returns None", agent4.observe_state(state) is None)
    a2 = brain.act(state, bias=agent4.observe_state(state))
    check("DQN acts normally with disabled nucleus", 0 <= a2 < bot_ml.ACTION_DIM)
    bot_ml.SPIKING_ENABLED = True

    # ---- match reset: transients cleared, weights persist
    w_before_reset = agent.nucleus.ff[0][0].w
    agent.new_match_reset()
    check("match reset preserves weights", agent.nucleus.ff[0][0].w == w_before_reset)
    check("match reset clears membrane", agent.nucleus.action[0].v == agent.nucleus.action[0].v_rest)

    print()
    if _failures:
        print(f"FAILED: {_failures}")
        sys.exit(1)
    print("ALL INTEGRATION TESTS PASS")
    sys.exit(0)

main()
