"""Hotfix test: deck-aware counter selection and opponent adaptation.

Reproduces the reported failure — the counter matrix recommends minipekka vs
giant and zap/arrows vs barrel while those cards are NOT in the active deck —
and proves the fixed pipeline can never select or play them:
  * an unavailable Mini P.E.K.K.A is never selected (active-deck filter),
  * an absent Zap or Arrows can never be played,
  * available cards are scored correctly vs repeated Giants,
  * the LIVE action-execution path (card_play_policy RULE 6 → final legality)
    is tested, not just the counter helper.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import bot_ml
from bot_ml import (score_counter_candidates, select_counter_card,  # noqa: E402
                    card_play_policy, validate_final_action, opponent_new_match,
                    opponent_observe)

ACTIVE = ["hogrider", "musketeer", "ice_golem", "cannon",
          "skeletons", "ice_spirit", "fireball", "the_log"]

P, F = 0, 0


def check(name, cond, detail=""):
    global P, F
    if cond:
        P += 1
        print("PASS", name)
    else:
        F += 1
        print("FAIL", name, detail)


T_SAME = {"ally_left": 1.0, "ally_right": 1.0, "ally_king": 1.0}


def A(left, bridge_l=0, right=0):
    return {"enemy_left": left, "enemy_bridge_l": bridge_l,
            "enemy_right": right, "enemy_bridge_r": 0, "air_threat": 0}


# --- 1. mini-pekka: in hand, in the counter matrix's idea list, NOT in deck --
HAND = ["minipekka", "cannon", "skeletons", "ice_golem"]
cands, rej = score_counter_candidates("giant", HAND, [0.9] * 4, active_deck=ACTIVE,
                                      towers=T_SAME, mlxir=10)
check("1a minipekka rejected NOT_IN_ACTIVE_DECK",
      any(r["card"] == "minipekka" and r["reason"] == "NOT_IN_ACTIVE_DECK" for r in rej),
      str(rej))
check("1b select_counter_card NEVER returns minipekka vs giant",
      select_counter_card("giant", HAND, [0.9] * 4, active_deck=ACTIVE) is None
      or select_counter_card("giant", HAND, [0.9] * 4, active_deck=ACTIVE)["card"] != "minipekka")
check("1c with deck cards conflicting: best playable candidate is legal",
      all(str(c["card"]) in ACTIVE for c in cands), str(cands))

# --- 2. zap / arrows vs goblin_barrel: absent from deck and hand -------------
HAND2 = ["hogrider", "musketeer", "cannon", "ice_golem"]
cands2, rej2 = score_counter_candidates("goblin_barrel", HAND2, [0.9] * 4,
                                        active_deck=ACTIVE)
check("2a zap/arrows absent -> no zap/arrows candidate",
      all(c["card"] not in ("zap", "arrows") for c in cands2))
cheap = select_counter_card("goblin_barrel", HAND2, [0.9] * 4, active_deck=ACTIVE)
check("2b barrel push still met with a LEGAL chip (or honest None), never zap/arrows",
      cheap is None or cheap["card"] not in ("zap", "arrows"), str(cheap))

# --- 3. NOT_IN_HAND / INSUFFICIENT_ELIXIR never authorize --------------------
HAND3 = [None, "musketeer", "cannon", "ice_golem"]
c3, r3 = score_counter_candidates("giant", HAND3, [0.9, 0.9, 0.9, 0.9], active_deck=ACTIVE)
check("3a unverified slot rejected NOT_IN_HAND",
      any(r["reason"] == "NOT_IN_HAND" for r in r3), str(r3))
HAND3b = ["hogrider", "cannon", "musketeer", "ice_golem"]
c3b, r3b = score_counter_candidates("giant", HAND3b, [0.5, 0.9, 0.9, 0.9], active_deck=ACTIVE)
check("3b unverified-elixir slot rejected INSUFFICIENT_ELIXIR",
      any(r["card"] == "hogrider" and r["reason"] == "INSUFFICIENT_ELIXIR" for r in r3b),
      str(r3b))
check("3c candidates never reference the unreadable slot 0",
      all(c["slot"] != 0 for c in c3))

# --- 4. repeated Giants: runnable scoring on the live deck ------------------
c4, r4 = score_counter_candidates("giant", ["musketeer", "cannon", "skeletons", "ice_golem"],
                                  [0.9, 0.9, 0.9, 0.9], active_deck=ACTIVE,
                                  threat_lane="left", towers=T_SAME, mlxir=9)
top = c4[0] if c4 else None
check("4a cannon ranks top vs repeated giant (building + prior)",
      top and top["card"] == "cannon", str([(c['card'], c['score']) for c in c4]))
check("4b every candidate carries evidence + cost (verified, not invented)",
      top and top.get("evidence") and top.get("cost") == 3, str(top))

# --- 5. LIVE action path: opponent pushes giants 5 times, then RULE 6 -------
opp = opponent_new_match()
prev = A(0)
cur = A(18)
opponent_observe(opp, prev, cur, T_SAME, T_SAME)
prev = cur
for left in (19, 20, 21, 22):
    opponent_observe(opp, prev, A(left), T_SAME, T_SAME)
    prev = A(left)
check("5a five giant-push ticks -> events deduped by lane quiescence",
      len(opp["pushes"]) <= 2, str(len(opp["pushes"])))
check("5b unique event ids", len({p["uid"] for p in opp["pushes"]}) == len(opp["pushes"]))
check("5c events honestly labeled inferred (not observed)",
      all(p["source"] == "inferred" for p in opp["pushes"]))
check("5d elixir from VERIFIED card costs (giant=5) with known/est split",
      opp["elixir_spent_est"] == 5.0 * len(opp["pushes"])
      and opp["elixir_spent_known"] == 0, str(opp["elixir_spent_est"]))

# live RULE 6: net proposed playing musketeer (slot 0), giant threat on left
opp["last_event_t"] = time.time()      # FRESH threat
act5, note5 = card_play_policy(0 * 5 + 4, 9, "left", A(18), T_SAME,
                               [0.9, 0.9, 0.9, 0.9],
                               ["musketeer", "cannon", "skeletons", "ice_golem"],
                               opponent=opp)
check("5e live path redirects to the DECK counter (cannon at center_back)",
      act5 == 1 * 5 + 4 and "counter cannon vs giant" in note5, f"{act5} {note5}")
okf, _ = validate_final_action(act5, ["musketeer", "cannon", "skeletons", "ice_golem"],
                               [0.9, 0.9, 0.9, 0.9])
check("5f the executed action passes the final legality gate", okf)

# stale archetype must NOT override the live situation
opp["last_event_t"] = time.time() - 20.0
_ = bot_ml._metrics.setdefault("counter_rejections", []).clear()
act6, note6 = card_play_policy(0 * 5 + 4, 9, "left", A(0), T_SAME,
                               [0.9, 0.9, 0.9, 0.9],
                               ["musketeer", "cannon", "skeletons", "ice_golem"],
                               opponent=opp)
check("6 stale archetype label (>8s) never redirects the play",
      act6 == 0 * 5 + 4
      and any(r.get("reason") == "STALE_THREAT"
              for r in bot_ml._metrics.get("counter_rejections", [])), note6)

# dominance feed present and honest (repeated strategy shapes the prior)
check("7 repeated-strategy dominance signal computed",
      opp.get("dominant_archetype") == "giant" and opp.get("dominant_ratio", 0) > 0.5,
      str(opp.get("dominant_archetype")), )

print(f"\n{P} passed, {F} failed")
sys.exit(1 if F else 0)
