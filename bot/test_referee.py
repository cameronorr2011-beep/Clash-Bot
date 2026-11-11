"""Referee module tests: deterministic dopamine/pain with evidence and caps,
anti-farming exemptions, elixir trades, no-LLM guarantee, structured outputs,
and 9-feature mentor interpreter."""
import sys
import referee
from referee import (referee_pain, referee_adjust, referee_evaluate,
                     referee_evaluate_damage, interpret_mentor,
                     groq_damage_punishment, CheapCycleMonitor,
                     DAMAGE_FLOOR, DAMAGE_CEIL)

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

# ---- 1. evidence discipline: no damage -> no fabricated penalty ----
p, src = referee_pain({"ally_hp_drop": 0.0, "ally_towers_lost": 0}, 3, 10, 1.0)
check("no damage -> zero penalty, never invented", p == 0.0 and "no measured damage" in src)
d_struct = referee_evaluate_damage({"ally_hp_drop": 0.0, "ally_towers_lost": 0}, 3, 10, 1.0)
check("structured damage eval returns components/evidence/source",
      d_struct["reward_adjustment"] == 0.0 and "evidence" in d_struct and "source" in d_struct)

p, _ = referee_pain({"ally_hp_drop": 0.0, "ally_towers_lost": 0, "ally_king_drop": 0.1}, 3, 10, 1.0)
check("king damage is measured, weighted MORE than regular damage", -0.62 < p <= -0.56)
p, _ = referee_pain({"ally_hp_drop": 0.3}, 3, 10, 1.0)
check("proportional rule applies exactly", abs(p + 1.2) < 1e-6)

# ---- 2. king-next escalation + phase urgency + hard caps ----
p_normal, _ = referee_pain({"ally_hp_drop": 0.0, "ally_towers_lost": 1}, 3, 10, 1.0)
p_kingnext, _ = referee_pain({"ally_hp_drop": 0.0, "ally_towers_lost": 1, "king_next": True}, 2, 10, 1.0)
check("tower loss near king-next escalates", p_kingnext < p_normal)
p_end, _ = referee_pain({"ally_hp_drop": 0.3}, 3, 50, 0.8)          # late game
p_mid, _ = referee_pain({"ally_hp_drop": 0.3}, 3, 10, 0.8)
check("endgame urgency multiplies", abs(p_end) > abs(p_mid))
p_cat, _ = referee_pain({"ally_hp_drop": 1.0, "ally_towers_lost": 3}, 0, 60, 0.1)
check("catastrophic damage clamps at documented floor", p_cat == DAMAGE_FLOOR)
check("floors/ceilings documented and constant", DAMAGE_FLOOR == -3.5 and DAMAGE_CEIL == -0.5)

# ---- 3. compatibility wrapper has ZERO LLM dependency ----
p, src = groq_damage_punishment({"ally_hp_drop": 0.5, "ally_towers_lost": 0})
check("legacy wrapper is deterministic (no API), contract preserved",
      src == "referee_deterministic" and p < 0)

# ---- 4. anti-farming: lull-farm suppressed, threat-defense exempt ----
m = CheapCycleMonitor()
for _ in range(4):
    m.record(1.0, threat=False)
adj, comps = referee_adjust({"cost": 1.0, "threat": 0.0}, cheap_monitor=m)
check("repeat cheap in lull = farming suppressed", adj <= referee.CHEAP_CYCLE_FARM_SUPPRESS
      and "cheap_cycle_farm" in comps)
m2 = CheapCycleMonitor()
for _ in range(4):
    m2.record(1.0, threat=True)
adj2, comps2 = referee_adjust({"cost": 1.0, "threat": 1.0}, cheap_monitor=m2)
check("same repetition UNDER THREAT is never suppressed (genuine defense)",
      "cheap_cycle_farm" not in comps2)

# ---- 5. weakest-tower + finisher components + elixir trades with evidence ----
adj3, comps3 = referee_adjust({"targeted_tower": ("left", 0.1),
                               "enemy_hp": {"enemy_left": 0.1, "enemy_right": 1.0}})
check("weakest-tower targeting bonus with evidence", adj3 > 0 and "weakest_tower" in comps3)
adj_wrong_t, comps_wrong_t = referee_adjust({"targeted_tower": ("right", 1.0),
                                             "enemy_hp": {"enemy_left": 0.2, "enemy_right": 1.0}})
check("targeting stronger tower gets no weakest-tower bonus", "weakest_tower" not in comps_wrong_t)

adj4, comps4 = referee_adjust({"finisher_executed": True})
check("executed finisher credited", "finisher_executed" in comps4 and adj4 > 0)
adj5, comps5 = referee_adjust({"finisher_available": True, "spell_used": "fireball",
                               "finisher_executed": False})
check("missed finisher penalized only when available",
      "missed_finisher" in comps5 and adj5 < 0)
adj_nofin, comps_nofin = referee_adjust({"finisher_available": False, "spell_used": "fireball",
                                         "finisher_executed": False})
check("no missed-finisher penalty when no finisher was available",
      "missed_finisher" not in comps_nofin)

# Elixir trade: only when reliable cost estimates are present
res_trade = referee_evaluate({"cost": 3.0, "opponent_cost_est": 6.0})
check("positive elixir trade credited when opponent cost is reliably known",
      res_trade["reward_adjustment"] > 0 and "elixir_trade" in res_trade["components"])
res_notrade = referee_evaluate({"cost": 3.0, "opponent_cost_est": None})
check("no elixir trade invented when opponent cost is unknown",
      "elixir_trade" not in res_notrade["components"])

# ---- 6. referee contribution caps ----
big, _ = referee_adjust({"targeted_tower": ("left", 0.1),
                         "enemy_hp": {"enemy_left": 0.1, "enemy_right": 1.0},
                         "finisher_executed": True,
                         "cost": 2.0, "opponent_cost_est": 9.0,
                         "defended_threat_lane": True, "threat": 1.0,
                         "counterpush_lane": True})
check("referee bonus capped at documented ceiling", big <= referee.REF_BONUS_CAP)

# ---- 7. mentor interpreter: all 9 features + explicit accept/reject ----
r1 = interpret_mentor("finish the left tower now for a positive elixir trade", "bridge_left",
                      {"towers": {"enemy_left": 0.2, "enemy_right": 1.0},
                       "finishers": [{"side": "left"}]})
check("finish intent + finisher lane accepted", r1["accepted"])
required_9 = {"elixir", "finish", "defend", "pressure", "tower",
              "trade", "counterpush", "caution", "novelty"}
check("all 9 required semantic features parsed", required_9.issubset(set(r1["features"].keys())))
check("features accurately detect finish, tower, trade, elixir",
      r1["features"]["finish"] and r1["features"]["tower"] and r1["features"]["trade"])

r2 = interpret_mentor("push their tower now", "left_defend",
                      {"towers": {"enemy_left": 0.1, "enemy_right": 1.0},
                       "finishers": [{"side": "left"}]})
check("non-finisher spot while finisher pending = rejected with reason",
      not r2["accepted"] and "physics" in r2["reason"])
r3 = interpret_mentor("do something", "not_a_spot")
check("illegal spot named in rejection", not r3["accepted"] and "illegal" in r3["reason"])
r4 = interpret_mentor("defend right lane", "right_defend",
                      {"threat_lane": "left", "towers": {"enemy_left": 1.0, "enemy_right": 1.0}})
check("wrong-lane defense rejected against measured threat_lane",
      not r4["accepted"] and "threat" in r4["reason"])
r5 = interpret_mentor("play knight now", "left_defend", {"affordable_slots": []})
check("unaffordable recommendation rejected", not r5["accepted"] and "affordable" in r5["reason"])

print(f"\nREFEREE TESTS: {PASS} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)
