"""Tactics-layer tests: the placement laws that stop the repeat blunders.

Run from the bot directory: py -3.12 test_tactics.py
"""
import tactics

PASS = 0


def check(name, cond):
    global PASS
    if cond:
        PASS += 1
        print("PASS", name)
    else:
        print("FAIL", name)


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


def A(slot, spot_i):
    return slot * 5 + spot_i


# ---------- threat detection ----------
check("empty arena reads as no threat", tactics.threat_lanes(EMPTY_ARENA) == (0.0, 0.0))
left, right = tactics.threat_lanes({"enemy_left": 5, "enemy_bridge_l": 3,
                                    "enemy_right": 1, "enemy_bridge_r": 0})
check("lanes sum lane + bridge columns", (left, right) == (8.0, 1.0))

# ---------- affordability ----------
act, note = tactics.advise(A(1, 0), 2, HAND, [1.0, 0.0, 1.0, 1.0], TOWERS, EMPTY_ARENA, CARDS)
check("unaffordable slot is replaced", act // 5 in (0, 2, 3) and "afford" in note)
check("replacement really is affordable", act // 5 != 1)
act, note = tactics.advise(A(1, 1), 5, HAND, ALL_AFFORD, TOWERS, EMPTY_ARENA, CARDS)
check("sound affordable play is left alone", (act, note) == (A(1, 1), ""))
act, note = tactics.advise(A(0, 0), 5, [], [], TOWERS, EMPTY_ARENA, CARDS)
check("empty hand never crashes or redirects", (act, note) == (A(0, 0), ""))

# ---------- rule 1: defend the lane that is under attack ----------
push_left = dict(EMPTY_ARENA, enemy_left=9)
act, note = tactics.advise(A(2, 4), 7, HAND, ALL_AFFORD, TOWERS, push_left, CARDS)  # right_defend
check("wrong-lane defence is corrected to the threatened lane",
      act == A(2, 2) and note.startswith("defend_left"))
act, note = tactics.advise(A(0, 0), 7, HAND, ALL_AFFORD, TOWERS, push_left, CARDS)  # center_back
check("center_back under a live push becomes a lane defence",
      act == A(0, 2) and note.startswith("defend_left"))
act, note = tactics.advise(A(2, 2), 7, HAND, ALL_AFFORD, TOWERS, push_left, CARDS)
check("correct defence is not touched", (act, note) == (A(2, 2), ""))
act, note = tactics.advise(A(2, 1), 7, HAND, ALL_AFFORD, TOWERS, push_left, CARDS)
check("same-side bridge counter-push is allowed", (act, note) == (A(2, 1), ""))
act, note = tactics.advise(A(2, 1), 7, HAND, ALL_AFFORD, TOWERS, push_left, CARDS)
check("opposite bridge is not a defence", act == A(2, 1))

push_right = dict(EMPTY_ARENA, enemy_right=12, enemy_left=2)
act, note = tactics.advise(A(1, 0), 8, HAND, ALL_AFFORD, TOWERS, push_right, CARDS)
check("right push pulls the answer right", act == A(1, 4) and note.startswith("defend_right"))

# ---------- rule 2: no threat + high elixir => pressure ----------
act, note = tactics.advise(A(0, 0), 9, HAND, ALL_AFFORD, TOWERS, EMPTY_ARENA, CARDS)
check("no-threat center_back at 9 elixir becomes a push on the weak tower",
      act == A(0, 1) and "pressure_left" in note)
act, note = tactics.advise(A(0, 2), 10, HAND, ALL_AFFORD,
                           {"enemy_left": 1.0, "enemy_right": 0.30}, EMPTY_ARENA, CARDS)
check("the *weaker* tower is the target", act == A(0, 3) and "pressure_right" in note)
act, note = tactics.advise(A(0, 0), 4, HAND, ALL_AFFORD, TOWERS, EMPTY_ARENA, CARDS)
check("low elixir defends nothing and is left alone", (act, note) == (A(0, 0), ""))
act, note = tactics.advise(A(1, 1), 9, HAND, ALL_AFFORD, TOWERS, EMPTY_ARENA, CARDS)
check("an existing bridge push at high elixir is left alone", (act, note) == (A(1, 1), ""))
act, note = tactics.advise(A(0, 0), 9, HAND, ALL_AFFORD,
                           {"enemy_left": 0.0, "enemy_right": 0.0}, EMPTY_ARENA, CARDS)
check("no standing tower => no invented target", (act, note) == (A(0, 0), ""))

# ---------- bounds ----------
check("note never claims an override it did not make",
      tactics.advise(A(1, 3), 5, HAND, ALL_AFFORD, TOWERS, EMPTY_ARENA, CARDS)[1] == "")
check("action always stays inside the 20-action space",
      all(0 <= tactics.advise(a, e, HAND, ALL_AFFORD, TOWERS, ar, CARDS)[0] < 20
          for a in range(20) for e in (0, 5, 10) for ar in (EMPTY_ARENA, push_left)))

print(f"\nTACTICS TESTS: {PASS} passed")
raise SystemExit(0 if PASS == 19 else 1)
