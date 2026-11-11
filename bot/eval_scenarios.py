"""NeuroML offline evaluation harness (big-prompt §7).

Runs 10 controlled scenarios with fixed seeds against the ACTUAL shipped
decision layers (rules gates, hand tracker, spell ledger, tower targeting),
ablates the legality gates to measure their effect on the illegal-action
rate, and measures real model latency / memory. Nothing here invents match
win rates — there is no offline opponent yet; match metrics still come from live play.

Usage:
    python eval_scenarios.py            # prints table + writes eval_report.json
"""
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(HERE)

import bot_ml  # noqa: E402
from bot_ml import (card_play_policy, physics_finisher, tower_finish_check,  # noqa: E402
                    validate_final_action, spell_cast_value_gate)
import spell_ledger  # noqa: E402
import hand_tracker  # noqa: E402
import card_index  # noqa: E402

LIVE_DECK = ["hogrider", "musketeer", "ice_golem", "cannon",
             "skeletons", "ice_spirit", "fireball", "the_log"]
SPOT = {"bridge_l": 0, "bridge_r": 1, "left_defend": 2, "right_defend": 3, "center_back": 4}
RESULTS = []


def scenario(n, name, fn, seed):
    random.seed(seed)
    ok, detail, metrics = fn()
    RESULTS.append({"n": n, "scenario": name, "ok": bool(ok), "detail": str(detail)[:300],
                    **({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))})})
    print(("PASS " if ok else "FAIL ") + f"{n:02d} {name}  {'' if ok else detail}")


# ---------------------------------------------------------------- scenarios
def s1_no_target_cast():
    """Offensive spell at bridge with NO enemy evidence: legality gate must veto."""
    a = {"enemy_left": 0, "enemy_right": 0, "enemy_bridge_l": 0, "enemy_bridge_r": 0, "air_threat": 0}
    g = spell_cast_value_gate("fireball", a, {}, "bridge_left", None)
    return not g["ok"], g, {"vetoed": 1 if not g["ok"] else 0}


def s2_multi_target_cast():
    a = {"enemy_left": 12, "enemy_right": 3, "enemy_bridge_l": 1, "enemy_bridge_r": 0, "air_threat": 4}
    g = spell_cast_value_gate("arrows", a, {}, "left_defend", None)
    return g["ok"], g, {"expected_value_ok": 1 if g["ok"] else 0}


def s3_defensive_spell():
    """Dangerous push + spell in hand: gate allows it as defense (cluster evidence)."""
    a = {"enemy_left": 18, "enemy_right": 0, "enemy_bridge_l": 2, "enemy_bridge_r": 0, "air_threat": 0}
    g = spell_cast_value_gate("fireball", a, {}, "left_defend", None)
    return g["ok"], g, {"defensive_cast": 1 if g["ok"] else 0}


def s4_low_elixir():
    """Insufficient elixir: physics finisher refuses; policy no correction."""
    k = physics_finisher({"enemy_left": 0.2, "enemy_right": 1.0},
                         ["fireball", None, None, None], 2, afford=[1.0, None, None, None])
    return k is None, k, {}


def s5_lr_targeting():
    """Asymmetric towers: weakest LIVING tower targeted; destroyed ignored."""
    k = physics_finisher({"enemy_left": 0.0, "enemy_right": 0.3},
                         ["fireball", None, None, None], 10, afford=[1.0, None, None, None])
    return (k is not None and k["side"] == "right"
            and k["target_xy"] == bot_ml.ENEMY_TOWER_XY["right"]), k, {}


def s6_resolution_mapping():
    """Tower coordinates must derive from screen geometry, not fixed constants."""
    w, h = 1280, 2400
    left = bot_ml.enemy_tower_xy("left", screen_w=w, screen_h=h)
    right = bot_ml.enemy_tower_xy("right", screen_w=w, screen_h=h)
    frac_left = left[0] / w
    ok = (abs(frac_left - bot_ml.ENEMY_TOWER_XY["left"][0] / 1080) < 1e-3
          and left[0] < right[0])
    return ok, f"left={left} right={right}", {"left_x_frac": round(frac_left, 4)}


def s7_bad_hand_obs():
    """Unknown slot identity + unknown elixir NEVER authorize a tap."""
    tr = hand_tracker.make_tracker(LIVE_DECK)
    tr.observe_hand([None, "musketeer", None, None], [0, 1.0, 0, 0])
    c, why = tr.authorize_play(0)
    okf, whyf = validate_final_action(0 * 5 + 2, [None, "musketeer", None, None],
                                      [0.5, 1.0, 0.0, 1.0], tr)
    return (c is None and not okf), whyf, {}


def s8_deck_rotation():
    hud = hand_tracker.HandTracker(LIVE_DECK)
    hud.verify_initial_hand(["fireball", "hogrider", "ice_golem", "skeletons"], [1.0] * 4)
    hud.commit_play(0)
    lived = hud.live_hand()
    return ("fireball" not in lived and len(lived) == 4
            and all(c in LIVE_DECK for c in lived)), lived, {}


def s9_emergency_finisher():
    """Big push defers the finisher at low elixir; affordable elixir takes it."""
    towers = {"enemy_left": 0.2, "enemy_right": 0.95}
    fin = tower_finish_check(towers, ["fireball", None, None, None], 10)
    diff = {"enemy_left": 6, "enemy_bridge_l": 4, "enemy_right": 0, "enemy_bridge_r": 0}
    act_defer, note_defer = card_play_policy(3 * 5 + 4, 4, "left", diff, towers,
                                             [0, 0, 0, 0.9], ["m", "s", "z", "fireball"],
                                             finishers=fin)
    act_take, note_take = card_play_policy(3 * 5 + 4, 10, "left", diff, towers,
                                           [0, 0, 0, 1.0], ["m", "s", "z", "fireball"],
                                           finishers=fin)
    return ("defer finisher" in note_defer and act_defer == 3 * 5 + 4
            and act_take // 5 == 3 and "finisher" in note_take), \
        f"defer={note_defer} take={note_take}", {}


def s10_unknown_spell_outcome():
    Hm = {"left": (500, 1100)}
    pt = [(500, 1100 + 8 * i) for i in range(6)]
    bars = [(x, y, 220, 30, 60) for (x, y) in pt]
    led = spell_ledger.SpellLedger()
    led.record(card="arrows", hand_slot=1, spot_at_cast=2, cast_xy=(500, 1100),
               pre_frame_stats=bars, pre_towers={"enemy_left": 1.0, "enemy_right": 1.0},
               card_info_spell=card_index.spell_profile("arrows"),
               screen_w=1080, screen_h=2340, lane="left", tick=1)
    time.sleep = time.time  # noqa: no-op
    v = led.evaluate(bars, {"enemy_left": 1.0, "enemy_right": 1.0},
                     now=time.time() + 999)
    ok = v and v[0]["outcome"] == "UNKNOWN" \
        and spell_ledger.SpellLedger.pain_for("UNKNOWN", 3) == 0.0 \
        and spell_ledger.SpellLedger.credit_for("UNKNOWN", 3, False) == 0.0
    return ok, str(v), {"outcome": v[0]["outcome"] if v else "none"}


# ---------------------------------------------------------------- ablation
def ablation_illegal_rate(trials=400):
    """Policy ablation with seeds: run the ACTUAL policy net (greedy DQN)
    on randomized states, with vs without the live legality mask (verified
    slots only). Counts how often the chosen action would tap an
    unverified/unaffordable slot."""
    brain = bot_ml.TargetedDQN()
    brain.policy.eval()
    tr = hand_tracker.make_tracker(LIVE_DECK)
    tr.observe_hand([None, "musketeer", None, None], [0, 1.0, 0, 0])
    rng = random.Random(7)
    raw_bad = gated_bad = 0
    for _ in range(trials):
        state = [rng.random() for _ in range(38)]
        forbidden = [s * 5 + sp for s in range(4) for sp in range(5)
                     if tr.authorize_play(s)[0] is None]
        brain.epsilon = 0.0
        a_raw = brain.act(state)               # no mask: net picks anything
        a_gated = brain.act(state, forbidden=forbidden)
        raw_bad += 1 if a_raw // 5 in (0, 2, 3) else 0
        gated_bad += 1 if a_gated // 5 in (0, 2, 3) else 0
    return {"illegal_raw": round(raw_bad / trials, 4),
            "illegal_final": round(gated_bad / trials, 4)}


# ---------------------------------------------------------------- model metrics
def model_metrics():
    import torch
    from bot_ml import QNet
    torch.set_num_threads(2)
    n_cls = QNet(profile="standard")
    x = torch.tensor([0.5] * 38)
    for _ in range(10):
        n_cls(x)
    t = []
    for _ in range(100):
        t0 = time.perf_counter()
        n_cls(x)
        t.append((time.perf_counter() - t0) * 1000)
    params = sum(p.numel() for p in n_cls.parameters() if p.requires_grad)
    pmc_peak_kb = 0
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [("_", ctypes.c_long * 2), ("WorkingSetSize", ctypes.c_size_t),
                        ("_2", ctypes.c_size_t * 7)]
        pmc = PMC()
        ctypes.windll.psapi.GetProcessMemoryInfo(wt.HANDLE(-1), ctypes.byref(pmc), ctypes.sizeof(pmc))
        pmc_peak_kb = pmc.WorkingSetSize // 1024
    except Exception:
        pass
    return {"params": params, "fp32_mb": round(params * 4 / 1048576, 1),
            "infer_ms_avg": round(statistics.mean(t), 1),
            "infer_ms_p95": round(statistics.quantiles(t, n=20)[18], 1),
            "peak_rss_mb": round(pmc_peak_kb / 1024, 1)}


# ---------------------------------------------------------------- runner
if __name__ == "__main__":
    SEED = 20261019
    scenario(1, "offensive spell, no valuable target -> vetoed", s1_no_target_cast, SEED)
    scenario(2, "offensive spell, multiple valid targets -> cast", s2_multi_target_cast, SEED)
    scenario(3, "defensive spell vs dangerous push -> cast", s3_defensive_spell, SEED)
    scenario(4, "low elixir -> no impossible action", s4_low_elixir, SEED)
    scenario(5, "L vs R tower: weakest LIVING tower targeted", s5_lr_targeting, SEED)
    scenario(6, "resolution-aware tower coordinates", s6_resolution_mapping, SEED)
    scenario(7, "bad/missing hand observation -> no tap", s7_bad_hand_obs, SEED)
    scenario(8, "deck rotation keeps the tracker honest", s8_deck_rotation, SEED)
    scenario(9, "emergency defense defers finisher; affordable takes it", s9_emergency_finisher, SEED)
    scenario(10, "unknown spell outcome -> UNKNOWN, zero consequence", s10_unknown_spell_outcome, SEED)
    abl = ablation_illegal_rate()
    mm = model_metrics()
    report = {"seed": SEED, "generated": time.strftime("%Y-%m-%d %H:%M"),
              "scenarios": RESULTS, "policy_ablation": abl, "model": mm,
              "limitations": [
                  "no offline scripted-opponent win-rate yet: match metrics need live play",
                  "temporal branch (QNET_TEMPORAL=1) is trained only by live episodes",
                  "mirror/camera-flip reversal not detected yet (resolution scaling is)"]}
    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    print(f"\nablation: illegal rate raw={abl['illegal_raw']:.2%} -> final gated={abl['illegal_final']:.2%}")
    print(f"model: {mm['params']:,} params ({mm['fp32_mb']} MiB fp32) · infer avg {mm['infer_ms_avg']}ms p95 {mm['infer_ms_p95']}ms · peak RSS {mm['peak_rss_mb']} MiB")
    fails = [r for r in RESULTS if not r["ok"]]
    print(f"scenarios {len(RESULTS)-len(fails)}/{len(RESULTS)} passed")
    sys.exit(1 if fails else 0)
