"""Opponent Read regression tests (big-prompt §1 audit fix).

Reproduces the REAL failure the user saw on the dashboard:
  "last wave giant · seen: giant, giant, giant, giant, giant · est spent: 27"
One actual giant push spanning several vision ticks was logged as five
pushes. After the fix the same push counts ONCE; a genuinely new push counts
again.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from bot_ml import opponent_observe, opponent_new_match  # noqa: E402

P, F = 0, 0


def check(name, cond, detail=""):
    global P, F
    if cond:
        P += 1
        print("PASS", name)
    else:
        F += 1
        print("FAIL", name, detail)


opp = opponent_new_match()

T_SAME = {"enemy_left": 1.0, "enemy_right": 1.0}


def A(left, bridge_l=0, right=0):
    return {"enemy_left": left, "enemy_bridge_l": bridge_l,
            "enemy_right": right, "enemy_bridge_r": 0}


# --- one giant push present for 6 consecutive vision ticks -------------------
# (a real giant beatdown wave crosses the start line already dense: ≥16px)
prev = A(0)
cur = A(18)
opponent_observe(opp, prev, cur, T_SAME, T_SAME)
prev = cur
for left in (19, 20, 20, 21, 22, 25):        # push keeps existing tick by tick
    cur = A(left)
    opponent_observe(opp, prev, cur, T_SAME, T_SAME)
    prev = cur
check("1a one sustained push = ONE archetype event",
      opp["cards_played"].count("giant") == 1, str(opp["cards_played"]))
check("1b est elixir counted once for that push",
      opp["elixir_spent"] == 5, str(opp["elixir_spent"]))

# --- push continues more ticks (no re-log) ----------------------------------
for _ in range(4):
    opponent_observe(opp, prev, A(18), T_SAME, T_SAME)
check("1c continuing push never re-logs", opp["cards_played"].count("giant") == 1)
check("1d est elixir still 5 (no per-tick inflation)", opp["elixir_spent"] == 5)

# --- push decays, then a NEW push arrives → logged again --------------------
prev = A(25)
for left in (10, 4, 2):
    cur = A(left)
    opponent_observe(opp, prev, cur, T_SAME, T_SAME)
    prev = cur
opponent_observe(opp, A(2), A(9), T_SAME, T_SAME)    # second push (smaller wave → knight)
check("2a second push logs exactly ONE new event",
      opp["cards_played"].count("knight") == 1
      and len(opp["cards_played"]) == 2, str(opp["cards_played"]))
check("2b est elixir = 5 (giant) + 3 (knight) = 8", opp["elixir_spent"] == 8,
      str(opp["elixir_spent"]))

# --- logbait-style: same countdown for right lane independently -------------
opponent_observe(opp, A(9, right=0), A(14, right=18), T_SAME, T_SAME)
check("3a right-lane giant push logged independently once",
      opp["cards_played"].count("giant") == 2
      and opp["cards_played"].count("hogrider") == 0, str(opp["cards_played"]))

# --- re-cast suppression: repeated musketeer fallback within its window -----
opp2 = opponent_new_match()
p0 = A(0)
for left in (7, 9, 11, 13, 15, 14, 13, 12):
    c = A(left)
    opponent_observe(opp2, p0, c, T_SAME, T_SAME)
    p0 = c
check("4 archetypes remain single-event (no musketeer spam on residuals)",
      opp2["cards_played"].count("musketeer") <= 1, str(opp2["cards_played"]))

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
