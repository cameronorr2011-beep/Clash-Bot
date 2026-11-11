"""Unit tests for the v13/v14 spiking core + affective channels.

Run: python test_spiking.py
"""
import math
import sys

import spiking_core as sc

_failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS {name}")
    else:
        print(f"  FAIL {name} {detail}")
        _failures.append(name)


# ------------------------------------------------ LIF resting behavior
def test_lif_rest():
    print("[LIF resting]")
    n = sc.LIFNeuron(seed=1)
    for _ in range(400):
        n.step()
    check("resting potential hovers at v_rest", abs(n.v - sc.V_REST) < 3.0,
          f"v={n.v:.2f}")
    check("no spontaneous spikes at rest", n.spike_count == 0)


# ------------------------------------------------ LIF threshold / spike
def test_lif_spike():
    print("[LIF spike]")
    n = sc.LIFNeuron(seed=2)
    # sustained drive, like real synaptic bombardment re-injected each step
    spiked = False
    for _ in range(200):
        n.receive_step(6.0)
        if n.step():
            spiked = True
            break
    check("sustained input causes a spike", spiked)
    check("membrane resets after spike", n.v == sc.V_RESET)
    # refractory: immediately after spike it must not fire again for REF ms
    spiked2 = []
    for _ in range(int(sc.REFRACTORY_MS)):
        spiked2.append(n.step())
    check("refractory: no double-fire within refractory window",
          not any(spiked2))
    # without more input it should settle back to rest
    for _ in range(500):
        n.step()
    check("membrane returns toward rest after spike", n.v < sc.V_THRESH)


# ------------------------------------------------ LIF adaptation
def test_lif_adaptation():
    print("[LIF adaptation]")
    counts = []
    for burst in range(3):
        n = sc.LIFNeuron(seed=3)          # fresh membrane per burst
        cnt = 0
        for _ in range(100):
            n.receive_step(7.0)           # same sustained drive each burst
            if n.step():
                cnt += 1
        counts.append(cnt)
    # same drive on a fresh neuron truncation: adaptation lives on the neuron;
    # to test adaptation on ONE neuron, use one and compare early vs late ms
    check("deterministic check setup", counts[-1] > 0, f"counts={counts}")
    n2 = sc.LIFNeuron(seed=4)
    early = 0
    late = 0
    for t in range(1000):
        n2.receive_step(3.0)
        sp = n2.step()
        if sp and t < 200:
            early += 1
        elif sp and 600 <= t < 800:
            late += 1
    check("adaptation reduces firing rate over time", early > late,
          f"early={early} late={late}")


# ------------------------------------------------ STDP directionality
def test_stdp_directionality():
    print("[STDP directionality]")
    s = sc.STDPSynapse()
    w0 = s.w
    # pre fires first, post fires 10 ms later -> potentiation
    s.pre_spike()
    for _ in range(10):
        s.step_traces()
    s.post_spike()
    check("pre-before-post potentiates", s.w > w0, f"w {w0} -> {s.w:.4f}")

    # reversed order: post first, pre 10 ms later -> depression
    s2 = sc.STDPSynapse()
    w0b = s2.w
    s2.post_spike()
    for _ in range(10):
        s2.step_traces()
    s2.pre_spike()
    check("post-before-pre depresses", s2.w < w0b, f"w {w0b} -> {s2.w:.4f}")


# ------------------------------------------------ STDP bounds
def test_stdp_bounds():
    print("[STDP bounds]")
    s = sc.STDPSynapse(w0=1.0)
    # try to potentiate a saturated synapse
    for _ in range(50):
        s.pre_spike()
        s.step_traces()
        s.post_spike()
    check("weights bounded above", s.w <= sc.W_MAX, f"w={s.w}")
    s2 = sc.STDPSynapse(w0=0.0)
    for _ in range(50):
        s2.post_spike()
        s2.step_traces()
        s2.pre_spike()
    check("weights bounded below", s2.w >= sc.W_MIN, f"w={s2.w}")


# ------------------------------------------------ reward-modulated STDP
def test_reward_modulation():
    print("[Reward-modulated STDP]")
    s = sc.STDPSynapse(w0=0.5)
    s.pre_spike()
    s.step_traces()
    s.post_spike()
    w_before = s.w
    # accumulate eligibility: same -> weight change
    delta = s.apply_reward(+1.0)
    check("positive reward potentiates eligible synapse",
          s.w > w_before or delta > 0,
          f"w before {w_before:.4f} -> {s.w:.4f}, delta={delta:.5f}")
    # reward with no eligibility should be a no-op
    s.apply_reward(+1.0)  # eligibility cleared above -> should be no-op
    check("reward with no eligibility changes nothing", s.eligibility == 0)


# ------------------------------------------------ affective valence bounds
def test_valence_bounds():
    print("[Affective valence]")
    a = sc.AffectiveCore()
    a.inject("neg_rpe", 5.0)
    check("negative valence clamped to <= 1", a.state["neg_rpe"] <= 1.0)
    a.inject("pos_rpe", 9.0)
    check("positive valence clamped to <= 1", a.state["pos_rpe"] <= 1.0)
    for _ in range(300):
        a.decay_step()
    check("negative valence decays away", a.state["neg_rpe"] < 0.05,
          f"neg_rpe={a.state['neg_rpe']:.4f}")
    check("positive valence decays away", a.state["pos_rpe"] < 0.05,
          f"pos_rpe={a.state['pos_rpe']:.4f}")
    before = a.state["frustration"]
    a.inject("frustration", 0.4)
    check("frustration rises on repeated failure", a.state["frustration"] > before)
    a2 = sc.AffectiveCore()
    a2.set("confidence", 0.9)
    for _ in range(500):
        a2.decay_step()
    check("confidence relaxes toward neutral 0.5",
          abs(a2.state["confidence"] - 0.5) < 0.05)


# ------------------------------------------------ reset semantics
def test_reset_semantics():
    print("[Reset semantics]")
    n = sc.LIFNeuron(seed=5)
    n.receive(30.0)
    for _ in range(200):
        n.step()
    n.reset()
    check("reset returns membrane to rest", n.v == n.v_rest)
    check("reset clears spike count", n.spike_count == 0)

    net = sc.SpikingNucleus(n_sensory=8, n_action=3, seed=7)
    net.encode_sensory([1.0] * 8)
    net.step(30)
    net.apply_reward(1.0)
    w_before = net.state_dict()
    net.reset_transient()
    w_after = net.state_dict()
    check("reset_transient preserves weights",
          w_before["ff"][0][0]["w"] == w_after["ff"][0][0]["w"])
    check("reset_transient clears spike log", len(net.spike_log) == 0)


# ------------------------------------------------ nucleus runs stable
def test_nucleus_stability():
    print("[Nucleus stability]")
    net = sc.SpikingNucleus(n_sensory=16, n_action=5, seed=11)
    for rep in range(5):
        net.encode_sensory([0.6] * 16)
        net.step(10)
    t = net.telemetry()
    check("network produces spikes", sum(t["action_activity"]) > 0)
    check("weights remain within bounds", 0 <= t["mean_w"] <= 1.0,
          f"mean_w={t['mean_w']:.4f}")
    finite_membranes = all(math.isfinite(n.v) for n in net.sensory + net.action)
    check("membrane potentials stay finite", finite_membranes)


def main():
    test_lif_rest()
    test_lif_spike()
    test_lif_adaptation()
    test_stdp_directionality()
    test_stdp_bounds()
    test_reward_modulation()
    test_valence_bounds()
    test_reset_semantics()
    test_nucleus_stability()
    print()
    if _failures:
        print(f"FAILED: {len(_failures)} -> {_failures}")
        sys.exit(1)
    print("ALL TESTS PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
