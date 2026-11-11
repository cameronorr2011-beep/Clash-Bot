"""NeuroML v16 spec-compliance tests: reward-ledger honesty (no censoring,
no double-counting), configurable reward weights, lifetime W/L record
seeding correctness."""
import bot_ml
from bot_ml import REWARD_WEIGHTS, reward_totals_update, seed_lifetime_record

PASS = 0
def check(name, cond):
    global PASS
    if cond: PASS += 1; print("PASS", name)
    else: print("FAIL", name)

# ---------------- reward ledger: additive accounting, no censoring ----------------
tot = {}
reward_totals_update(tot, [("✅ correct lane defense", 0.8), ("🛑 my tower took damage", -1.6)], dmg_punish=None if False else -1.6)
check("simultaneous good+bad both recorded", tot.get("✅ correct lane defense") == 0.8
      and tot.get("ALL TOWER DAMAGE") == -1.6)
check("no censoring: the positive survives the tower hit", "✅ correct lane defense" in tot)
reward_totals_update(tot, [("✅ correct lane defense", 0.8)], 0.0)
check("repeat event accumulates (sum not overwrite)", tot.get("✅ correct lane defense") == 1.6)
reward_totals_update(tot, [], None)
check("empty tick changes nothing", tot.get("ALL TOWER DAMAGE") == -1.6)
tot2 = {}
reward_totals_update(tot2, [("only event", 0.5)], dmg_punish=0.0)
check("zero damage is not logged as a component", "ALL TOWER DAMAGE" not in tot2)

# ---------------- configurable reward weights ----------------
check("reward weights are a real module-level dict", isinstance(REWARD_WEIGHTS, dict))
required = ["correct_lane_defense", "tower_advantage_held", "game_rule_followed",
            "enemy_troop_kill_per_px", "spell_wasted", "wrong_lane_defense",
            "tower_lost_per_tower"]
check("all major shaping terms configurable", all(k in REWARD_WEIGHTS for k in required))
check("tower loss is always strongly negative",
      REWARD_WEIGHTS["tower_lost_per_tower"] <= -2.0)
check("weights can be re-tuned at runtime",
      (REWARD_WEIGHTS.__setitem__("correct_lane_defense", 0.9) or True)
      and REWARD_WEIGHTS["correct_lane_defense"] == 0.9)

# ---------------- lifetime W/L record seeding ----------------
fake_mem = {"matches": [
    {"v": "WIN"}, {"v": "WIN"}, {"v": "LOSS"}, {"v": "WIN"},
    {"v": "LOSS"}, {"v": "LOSS"},
]}
m = {}
n = seed_lifetime_record(m, fake_mem)
check("counts every verdict-bearing match", n == 6 and m["wins_v"] == 3 and m["losses_v"] == 3)
check("win rate is mathematically correct over the lifetime", abs(m["win_rate"] - 50.0) < 0.01)
check("history window keeps last 60", len(m["win_history"]) == 6
      and m["win_history"][-1] == 0 and m["win_history"][0] == 1)
check("record provenance is labeled for the dashboard",
      m.get("record_source", "").startswith("game_memory.json"))
check("lifetime game count exposed", m.get("lifetime_games") == 6)

mixed_mem = {"matches": [{"v": "WIN"}, {"t": "garbage"}, {"v": "bogus"}, {"v": "LOSS"}]}
m2 = {}
n2 = seed_lifetime_record(m2, mixed_mem)
check("corrupt/unlabeled entries are skipped, not miscounted", n2 == 2 and m2["wins_v"] == 1)
m3 = {}
n3 = seed_lifetime_record(m3, {"matches": []})
check("empty memory seeds to 0-0 safely", n3 == 0 and m3["wins_v"] == 0 and m3["losses_v"] == 0
      and m3["win_history"] == [])

# real dashboard consistency: ml_metrics record fields, once seeded live
try:
    mm = bot_ml._metrics
    seed_lifetime_record(mm)
    check("live _metrics accepts the seed", "wins_v" in mm and mm["wins_v"] >= 0)
except Exception as e:
    check("live _metrics accepts the seed", False)

print(f"\nREWARD/RECORD TESTS: {PASS} passed (failures printed above, if any)")
