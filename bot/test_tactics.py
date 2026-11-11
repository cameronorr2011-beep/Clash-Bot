"""Tactics-layer tests: placement laws + card choice.

Run from the bot directory: py -3.12 test_tactics.py
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
    "musketeer": {"cost": 4, "role": "support"},
    "ice_golem": {"cost": 3, "role": "support"},
    "cannon": {"cost": 3, "role": "building"},
    "skeletons": {"cost": 1, "role": "support"},
    "the_log": {"cost": 2, "role": "spell"},
}
HAND = ["hogrider", "musketeer", "ice_golem", "the_log"]
ALL = [1.0, 1.0, 1.0, 1.0]
TOWERS = {"enemy_left": 0.42, "enemy_right": 0.91, "ally_left": 1.0,
          "ally_right": 1.0, "ally_king": 1.0}
CALM = {"enemy_left": 0, "enemy_right": 0, "enemy_bridge_l": 0, "enemy_bridge_r": 0,
        "ally_left": 0, "ally_right": 0}


def A(slot, spot_name):
    return slot * 5 + tactics.spot_index(SPOTS, spot_name)


def spot_of(action):
    return SPOTS[action % 5]


def adv(action, elixir, arena=CALM, towers=TOWERS, afford=ALL, hand=HAND):
    return tactics.advise(action, elixir, hand, afford, towers, arena, CARDS, spot_names=SPOTS)


# ---------- placement law (the bug this layer shipped with) ----------
act, note = adv(A(0, "center_back"), 9)
check("no-threat center_back redirects to the LEFT bridge row",
      act == A(0, "bridge_left") and "pressure_left" in note)
check("...and never to the center_back row", act % 5 != 4)

# ---------- rule 1: cheap troops answer a push, on the right lane ----------
push_left = dict(CALM, enemy_left=9)
act, note = adv(A(1, "right_defend"), 7, push_left)          # musketeer, wrong lane
check("wrong-lane defence is moved to the threatened lane",
      act == A(1, "left_defend") and note.startswith("defend_left"))
act, note = adv(A(1, "left_defend"), 7, push_left)
check("correct defence is not touched", (act, note) == (A(1, "left_defend"), ""))
act, note = adv(A(0, "center_back"), 9, push_left)            # hogrider as a defender
check("center_back under a push becomes a real defence", spot_of(act) == "left_defend")

# ---------- rule 1b: never a spell at a small push ----------
act, note = adv(A(3, "center_back"), 8, dict(CALM, enemy_left=6))   # the_log
check("a spell is swapped for a cheap troop on a small push",
      act // 5 == 2 and "no spell" in note)
check("...and the replacement IS the cheapest defender", act // 5 == 2)
act, note = adv(A(3, "center_back"), 8, dict(CALM, enemy_left=15))  # real swarm
check("a real swarm lets the spell defend in place",
      act // 5 == 3 and spot_of(act) == "left_defend")

# ---------- rule 2: convert the bank ----------
act, note = adv(A(0, "center_back"), 10, towers={"enemy_left": 1.0, "enemy_right": 0.30})
check("the *weaker* tower is the target",
      act == A(0, "bridge_right") and "pressure_right" in note)
act, note = adv(A(1, "center_back"), 9, dict(CALM, ally_left=14))
check("surviving ally troops are pushed behind, not abandoned",
      act == A(1, "bridge_left") and "support_left" in note)
act, note = adv(A(1, "center_back"), 5)
check("low elixir is not forced into a push", (act, note) == (A(1, "center_back"), ""))
act, note = adv(A(1, "bridge_left"), 9)
check("an existing bridge push at high elixir is left alone",
      (act, note) == (A(1, "bridge_left"), ""))

# ---------- affordability ----------
act, note = adv(A(1, "center_back"), 2, afford=[1.0, 0.0, 1.0, 0.0])
check("unaffordable slot is replaced", act // 5 != 1 and "afford" in note)
check("replacement is affordable", act // 5 in (0, 2))
act, note = adv(A(0, "center_back"), 5, hand=[], afford=[])
check("empty hand never crashes or redirects", (act, note) == (A(0, "center_back"), ""))
act, note = adv(A(0, "center_back"), 9, towers={"enemy_left": 0.0, "enemy_right": 0.0})
check("no standing tower => no invented target", (act, note) == (A(0, "center_back"), ""))
check("action always stays inside the 20-action space",
      all(0 <= adv(a, e, ar)[0] < 20 for a in range(20) for e in (0, 5, 10)
          for ar in (CALM, push_left, dict(CALM, enemy_right=20))))
check("notes are only emitted with a real change",
      all((adv(a, e)[1] == "") == (adv(a, e)[0] == a) for a in range(20) for e in (3, 9)))

print(f"\nTACTICS TESTS: {PASS} passed, {FAIL} failed")
raise SystemExit(0 if FAIL == 0 else 1)
