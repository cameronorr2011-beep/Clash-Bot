"""Tactics-layer tests: the placement laws that stop the repeat blunders.

Run from the bot directory: py -3.12 test_tactics.py

Spot indices come from the bot's own list, and one check pins that list to the
real one — the first version of this layer assumed "right_defend" was index 4
(it is center_back), which silently sent live redirects to the wrong row.
"""
import tactics

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


# The live bot's order, copied from bot_ml.SPOT_NAMES.
SPOTS = ["bridge_left", "bridge_right", "left_defend", "right_defend", "center_back"]
check("bot SPOT_NAMES order is the one we resolve against",
      list(tactics.SPOT_NAMES) == SPOTS and tactics.spot_index(SPOTS, "center_back") == 4)

CARDS = {
    "hogrider": {"cost": 4, "role": "win_condition"},
    "cannon": {"cost": 3, "role": "building"},
    "skeletons": {"cost": 1, "role": "support"},
    "the_log": {"cost": 2, "role": "spell"},
}
HAND = ["hogrider", "cannon", "skeletons", "the_log"]
ALL_AFFORD = [1.0, 1.0, 1.0, 1.0]
TOWERS = {"enemy_left": 0.42, "enemy_right": 0.91, "ally_left": 1.0,
          "ally_right": 1.0, "ally_king": 1.0}
EMPTY_ARENA = {"enemy_left": 0, "enemy_right": 0, "enemy_bridge_l": 0, "enemy_bridge_r": 0}


def A(slot, spot_name):
    return slot * 5 + tactics.spot_index(SPOTS, spot_name)


def adv(action, elixir, arena=EMPTY_ARENA, towers=TOWERS, afford=ALL_AFFORD, hand=HAND):
    return tactics.advise(action, elixir, hand, afford, towers, arena, CARDS, spot_names=SPOTS)


# ---------- the index law this bug came from ----------
act, note = adv(A(0, "center_back"), 9)
check("no-threat center_back redirects to the LEFT bridge row",
      act == A(0, "bridge_left") and "pressure_left" in note)
check("...and never to the center_back row", act % 5 != 4)

# ---------- threat detection ----------
check("empty arena reads as no threat", tactics.threat_lanes(EMPTY_ARENA) == (0.0, 0.0))
left, right = tactics.threat_lanes({"enemy_left": 5, "enemy_bridge_l": 3,
                                    "enemy_right": 1, "enemy_bridge_r": 0})
check("lanes sum lane + bridge columns", (left, right) == (8.0, 1.0))

# ---------- affordability ----------
act, note = adv(A(1, "center_back"), 2, afford=[1.0, 0.0, 1.0, 1.0])
check("unaffordable slot is replaced", act // 5 != 1 and "afford" in note)
check("replacement really is affordable", act // 5 in (0, 2, 3))
act, note = adv(A(1, "bridge_left"), 5)
check("sound affordable play is left alone", (act, note) == (A(1, "bridge_left"), ""))
act, note = adv(A(0, "center_back"), 5, hand=[], afford=[])
check("empty hand never crashes or redirects", (act, note) == (A(0, "center_back"), ""))

# ---------- rule 1: defend the lane that is under attack ----------
push_left = dict(EMPTY_ARENA, enemy_left=9)
act, note = adv(A(2, "right_defend"), 7, push_left)
check("wrong-lane defence is corrected to the threatened lane",
      act == A(2, "left_defend") and note.startswith("defend_left"))
act, note = adv(A(0, "center_back"), 7, push_left)
check("center_back under a live push becomes the threatened-lane defence",
      act == A(0, "left_defend") and note.startswith("defend_left"))
act, note = adv(A(2, "left_defend"), 7, push_left)
check("correct defence is not touched", (act, note) == (A(2, "left_defend"), ""))
act, note = adv(A(2, "bridge_left"), 7, push_left)
check("same-side bridge counter-push is allowed", (act, note) == (A(2, "bridge_left"), ""))
act, note = adv(A(2, "bridge_right"), 7, push_left)
check("opposite bridge is NOT a defence", act == A(2, "left_defend"))

push_right = dict(EMPTY_ARENA, enemy_right=12, enemy_left=2)
act, note = adv(A(1, "center_back"), 8, push_right)
check("right push pulls the answer to right_defend (row 3, not center_back)",
      act == A(1, "right_defend") and note.startswith("defend_right"))

# ---------- rule 2: no threat + high elixir => pressure ----------
act, note = adv(A(0, "center_back"), 10,
                towers={"enemy_left": 1.0, "enemy_right": 0.30})
check("the *weaker* tower is the target",
      act == A(0, "bridge_right") and "pressure_right" in note)
act, note = adv(A(0, "left_defend"), 4)
check("low elixir defends nothing and is left alone", (act, note) == (A(0, "left_defend"), ""))
act, note = adv(A(1, "bridge_left"), 9)
check("an existing bridge push at high elixir is left alone",
      (act, note) == (A(1, "bridge_left"), ""))
act, note = adv(A(0, "center_back"), 9, towers={"enemy_left": 0.0, "enemy_right": 0.0})
check("no standing tower => no invented target", (act, note) == (A(0, "center_back"), ""))
act, note = adv(A(0, "center_back"), 9, arena={"enemy_right": 7})
check("with a live push, pressure rule never overrides the defence",
      act == A(0, "right_defend"))
act, note = adv(A(1, "center_back"), 5)
check("note never claims an override it did not make", note == "")
check("action always stays inside the 20-action space",
      all(0 <= adv(a, e, ar)[0] < 20 for a in range(20) for e in (0, 5, 10)
          for ar in (EMPTY_ARENA, push_left, push_right)))

print(f"\nTACTICS TESTS: {PASS} passed")
print(f"failures: {FAIL}")
raise SystemExit(0 if FAIL == 0 else 1)
