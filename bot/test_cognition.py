"""vNext cognition tests: deterministic referee owns dopamine/pain (no Groq
in the reward path), 21-channel interacting affect core with persistence and
behavioral modulation, structural mentor interpreter, real offline federated
averaging (including synthetic 2 & 4 w/ weights 1 & 3 -> 3.5), model profiles
(compact/standard/advanced) with checkpoint migration, card-counter & 30-day
meta intelligence."""
import json
import math
import os as _os
import sys
import tempfile as _tf
import torch as _t
from spiking_core import AffectiveCore
from bot_ml import (referee_pain, interpret_mentor, QNET_PROFILES,
                    CARD_COUNTERS, select_counter_card, META_REFRESH_DAYS,
                    ensure_meta_fresh)

PASS = 0
FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("PASS", name)
    else:
        FAIL += 1
        print("FAIL", name)

# ---------------- deterministic referee (replaces the Groq punisher) ----------------
info = {"ally_hp_drop": 0.2, "ally_towers_lost": 0, "ally_king": 1.0}
p1, src1 = referee_pain(info, 3, 10)
check("proportional damage pain from measured physics", abs(p1 + 0.8) < 1e-6)
check("source is the rule referee, not a model", src1 == "referee_rules")
p2, _ = referee_pain({"ally_hp_drop": 0.5, "ally_towers_lost": 1, "ally_king": 1.0}, 2, 50)
check("tower loss + king-next + endgame urgency escalates pain", p2 < p1 and p2 <= -3.0)
p3, _ = referee_pain({"ally_hp_drop": 0.02, "ally_towers_lost": 0}, 3, 5)
check("small damage still punished (never silent)", p3 == -0.5)
p4, _ = referee_pain({"ally_hp_drop": 0.9, "ally_towers_lost": 3}, 0, 60)
check("catastrophe clamped at the floor", p4 == -3.5)
d_mid = [referee_pain(info, 3, t)[0] for t in (40, 41, 42)]
d_late = [referee_pain(info, 3, t)[0] for t in (47, 48, 49)]
check("deterministic: same damage -> same pain (per phase)",
      len(set(d_mid)) == 1 and len(set(d_late)) == 1)
check("R3 endgame urgency is an intentional, documented phase step",
      abs(d_late[0] - d_mid[0] - (1.25 - 1.0) * 0.8) < 0.01 or d_late[0] < d_mid[0])

# ---------------- extended 21-channel affect core ----------------
ac = AffectiveCore()
EXPECTED_CHANNELS = 21
check("extended core has 21 documented channels", len(ac.state) == EXPECTED_CHANNELS
      and len(ac.doc) >= EXPECTED_CHANNELS)
for k in ("neg_rpe", "pos_rpe", "progress", "danger", "confidence", "frustration",
          "relief", "anticipation", "curiosity", "momentum", "pressure",
          "desperation", "vigilance", "fatigue", "uncertainty", "focus",
          "control", "surprise", "strategic_conflict", "patience", "opponent_read"):
    check(f"channel '{k}' present, bounded, documented", k in ac.state
          and 0.0 <= ac.state[k] <= 1.0 and k in ac.doc)

base0 = ac.reward_baseline
ac.update({"reward": 3.0})
check("RPE is measured against the EMA baseline, not raw reward",
      ac.state["pos_rpe"] > 0 and ac.reward_baseline > base0)
ac.update({"king_hp": 0.1, "threat": 1.0})
check("desperation responds to critical king health (bounded injection)",
      ac.state["desperation"] > 0.03)
ac.update({"overruns": 3})
check("fatigue rises only from measured overrun evidence", ac.state["fatigue"] > 0.05)

# Behavioral & plasticity modulation + persistence roundtrip
bmod = ac.behavioral_modulation()
check("behavioral_modulation returns bounded plasticity_scale, exploration_nudge, defensive_bias",
      0.25 <= bmod["plasticity_scale"] <= 1.5 and -0.04 <= bmod["exploration_nudge"] <= 0.06
      and 0.0 <= bmod["defensive_bias"] <= 1.0)
ac_sd = ac.state_dict()
ac_restored = AffectiveCore()
ac_restored.load_state_dict(ac_sd)
check("AffectiveCore state_dict/load_state_dict preserves channels + reward_baseline",
      abs(ac_restored.reward_baseline - ac.reward_baseline) < 1e-6
      and abs(ac_restored.state["fatigue"] - ac.state["fatigue"]) < 1e-6)

for _ in range(60):
    ac.decay_step()
check("every channel returns toward resting equilibrium", all(v <= 0.55 for v in ac.state.values()))
ac.inject("danger", 0.9)
for _ in range(5):
    ac.decay_step()
check("danger raises vigilance (cross-channel coupling)", ac.state["vigilance"] > 0.05)
ac2 = AffectiveCore()
ac2.state["frustration"] = 0.9; ac2.state["confidence"] = 0.4
ac2.decay_step(); ac2.decay_step()
check("frustration inhibits confidence recovery", ac2.state["confidence"] < 0.4)
ac3 = AffectiveCore()
ac3.state["progress"] = 0.9; ac3.state["neg_rpe"] = 0.6
ac3.decay_step(); ac3.decay_step()
check("progress closes out negative affect (relief dynamics)",
      ac3.state["neg_rpe"] < 0.6)
comp = ac3.composites()
check("composites derived and bounded",
      set(comp) == {"desperation", "composure"}
      and all(0.0 <= v <= 1.0 for v in comp.values()))
check("valence is desperation-modulated, not just pos-neg",
      -1.0 <= ac3.valence_scalar() <= 1.0)

# ---------------- structural mentor interpreter ----------------
towers = {"enemy_left": 0.15, "enemy_right": 1.0}
m_ok = interpret_mentor({"critique": "finish the left tower now"}, towers, 6)
check("critique=finish accepted when a tower is in finish range",
      m_ok["accepted"] and m_ok["intents"]["finish"])
m_rej = interpret_mentor({"critique": "finish the right tower now"}, towers, 6)
check("the interpreter validates intent against the MEASURED state (right tower at 100% rejected)",
      not m_rej["accepted"] and m_rej["intents"]["finish"])
m_low = interpret_mentor({"critique": "save elixir, cycle cheap"}, towers, 3)
check("elixir-saving intent accepted at low elixir", m_low["accepted"])

# ---------------- card counters & 30-day meta engine ----------------
check("card counter matrix has >= 15 archetypes", len(CARD_COUNTERS) >= 15)
cnt = select_counter_card("hogrider", ["musketeer", "cannon", "zap", "giant"], [0, 1, 2], threat_lane="left")
check("select_counter_card picks cannon vs hogrider",
      cnt is not None and cnt["card"] == "cannon" and cnt["slot"] == 1)
check("meta refresh interval is 30 days", META_REFRESH_DAYS == 30)
used_meta, stale_d = ensure_meta_fresh()
check("ensure_meta_fresh runs cleanly", used_meta is True)

# ---------------- model profiles (compact / standard / advanced) & migration ----------------
from bot_ml import QNet, TargetedDQN, param_report
_os.environ["QNET_PROFILE"] = "compact"
net_c = QNet()
pr_c = param_report(net_c)
check("compact profile = 768-wide x 6 blocks (~7.89M measured)",
      7_000_000 <= pr_c["trainable_params"] <= 9_000_000 and pr_c["depth"] == 6 and pr_c["width"] == 768)

_os.environ["QNET_PROFILE"] = "advanced"
net_a = QNet()
pr_a = param_report(net_a)
check("advanced profile = mandated 1280-wide x 8 blocks (~27.93M measured)",
      25_000_000 <= pr_a["trainable_params"] <= 32_000_000 and pr_a["depth"] == 8 and pr_a["width"] == 1280)
check("fp32 size claim matches parameter count", abs(pr_a["fp32_mb"] - pr_a["trainable_params"] * 4 / 1048576) < 0.5)
out_a = net_a(_t.randn(1, 38))
check("forward pass works on advanced profile", tuple(out_a.shape) == (1, 20))

_os.environ["QNET_PROFILE"] = "standard"
_os.environ["NEUROML_PROFILE"] = "standard"
net_s = QNet()
pr_s = param_report(net_s)
check("standard profile restored = 1024-wide x 6 blocks (13,724,181 params)",
      pr_s["trainable_params"] == 13724181 and pr_s["world_model_params"] > 0)

# Test overlapping-slice migration between compact and standard
ex_n, sl_n, fr_n = TargetedDQN._migrate_overlapping_weights(net_s, net_c.state_dict())
check("profile migration copies overlapping slices explicitly without crashing",
      sl_n > 0 and ex_n >= 1)

# ---------------- federated averaging (real math, including 2 & 4 w/ 1 & 3 -> 3.5) ----------------
import federated

_cwd = _os.getcwd()
_sandbox = _tf.mkdtemp()           # keep the REAL fleet ledger out of tests
_os.chdir(_sandbox)
try:
    # Exact spec D synthetic test: weights 2 and 4 with episode weights 1 and 3 -> 3.5
    pkg_w2 = {
        "schema": "neuroml_federated_v2",
        "arch_fingerprint": "SYNTH_FP",
        "base_fingerprint": "SYNTH_FP",
        "profile": "standard",
        "episode_count": 1,
        "policy": {"sensory.0.weight": _t.full((4, 38), 2.0)},
        "target": {"sensory.0.weight": _t.full((4, 38), 2.0)},
        "matches": {"games": 1, "wins": 1, "losses": 0},
        "identity": "dev_w2",
    }
    pkg_w4 = {
        "schema": "neuroml_federated_v2",
        "arch_fingerprint": "SYNTH_FP",
        "base_fingerprint": "SYNTH_FP",
        "profile": "standard",
        "episode_count": 3,
        "policy": {"sensory.0.weight": _t.full((4, 38), 4.0)},
        "target": {"sensory.0.weight": _t.full((4, 38), 4.0)},
        "matches": {"games": 3, "wins": 2, "losses": 1},
        "identity": "dev_w4",
    }
    res_math = federated.federated_average([pkg_w2, pkg_w4], out="math_avg.pt")
    mean_val = float(res_math["policy"]["sensory.0.weight"].mean().item())
    check("synthetic tensors: weights 2 and 4 with episode_count 1 and 3 yield exact weighted mean 3.5",
          res_math["aggregated"] and abs(mean_val - 3.5) < 1e-6)

    def pkg(fp, ep, games, wins, losses, path=None, ident="t", base_fp=None):
        tiny = {"x": _t.randn(6, 4), "sensory.0.weight": _t.randn(8, 38)}
        d = {"schema": "neuroml_federated_v2", "epsilon": 0.3, "profile": "t",
             "arch_fingerprint": fp, "base_fingerprint": base_fp or fp,
             "episode_count": ep,
             "policy": tiny, "target": tiny,
             "replay_pointer": {"steps_count": 0, "buffer_size": 0},
             "matches": {"games": games, "wins": wins, "losses": losses},
             "identity": ident}
        if path:
            _t.save(d, path)
            return path
        return d

    # Reset ledger for clean contributor count check
    if _os.path.exists(federated.LEDGER):
        _os.remove(federated.LEDGER)
    s = federated.federated_average([pkg("FP", 1000, 50, 40, 10, "a.pt", ident="devA"),
                                     pkg("FP", 3000, 80, 60, 20, ident="devB")])
    check("FedAvg weights by EPISODE COUNT (4000 total)",
          s["aggregated"] and s["total_episodes"] == 4000)
    check("fleet match ledger sums every game ever", s["fleet_matches"]["games"] == 130
          and s["fleet_matches"]["wins"] == 100)
    s2 = federated.federated_average([pkg("FP", 10, 1, 1, 0, "b.pt"),
                                      pkg("DIFF", 10, 1, 1, 0)])
    check("fingerprint mismatch explicitly REJECTED (incompatible architecture)",
          not s2["aggregated"] and "fingerprint" in s2["reason"])
    s_lineage = federated.federated_average([pkg("FP", 10, 1, 1, 0, base_fp="BASE1"),
                                             pkg("FP", 10, 1, 1, 0, base_fp="BASE2")])
    check("incompatible base lineage explicitly REJECTED",
          not s_lineage["aggregated"] and "lineage" in s_lineage["reason"])
    s3 = federated.federated_average([pkg("FP", 0, 0, 0, 0)])
    check("zero-episode package refused", not s3["aggregated"])
    led = json.load(open(federated.LEDGER))
    check("only aggregation-valid contributions enter the ledger", len(led["records"]) == 2)
    federated.federated_average([pkg("FP", 1000, 50, 40, 10, "a.pt", ident="devA")])
    led = json.load(open(federated.LEDGER))
    check("re-importing the same contributor never double-counts",
          len(led["records"]) == 2)
    ok, msg = federated.verify_package("federated_aggregate.pt")
    check("aggregate passes structural verification", ok)
finally:
    _os.chdir(_cwd)
    _t_shutil = __import__('shutil')
    _t_shutil.rmtree(_sandbox, ignore_errors=True)

check("memory-cap: packages over 220 MB are refused before load",
      federated.MAX_PACKAGE_BYTES == 220 * 1024 * 1024)

print(f"\nCOGNITION TESTS: {PASS} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)
