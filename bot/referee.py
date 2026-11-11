"""NeuroML Referee — the DETERMINISTIC authority on dopamine and pain.

Every rule consumes measured battlefield signals only; no LLM is ever consulted
for a reward or punishment, so the reward path costs zero tokens and can never
be argued into a wrong verdict. Components are named, bounded, evidence-backed,
and each one lands in the reward ledger. This is a rules engine about a game,
NOT a model of feeling.

Compatibility: bot/bot_ml.py imports referee_pain/interpret_mentor/referee_adjust
from here. The legacy groq_damage_punishment() wrapper is kept for import
compatibility but NEVER contacts Groq — it forwards to the deterministic rules.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import bot_ml
except Exception:                       # standalone import (public_gui / tests)
    bot_ml = None

# deterministic weights (tunable constants, never model outputs)
TOWER_DAMAGE_PER_FRACTION = 4.0    # per full HP bar
KING_DAMAGE_MULT = 1.4             # king-tower damage weighs more
TOWER_LOST_PENALTY = 1.6
KING_NEXT_MULT = 2.0               # losing a tower when king is exposed next
ENDGAME_URGENCY = 1.25
DAMAGE_FLOOR = -3.5
DAMAGE_CEIL = -0.5
FINISH_CREDIT = 0.7
MISSED_FINISH_PENALTY = -0.35
WEAKEST_TARGET_BONUS = 0.15
ELIXIR_TRADE_SCALE = 0.08          # per-elixir advantage when reliably known
ELIXIR_TRADE_CAP = 0.35
DEFENSE_EFFECT_BONUS = 0.20
COUNTERPUSH_BONUS = 0.25
CHEAP_CYCLE_FARM_SUPPRESS = -0.20
CHEAP_CYCLE_FARM_THRESHOLD = 4     # repeated cheap placings in a lull = farming
FARM_LOOKBACK = 10
REF_BONUS_CAP = 1.5
REF_PENALTY_CAP = -3.5
SPELL_TOWER_DAMAGE = {"fireball": 0.33, "arrows": 0.17, "zap": 0.12,
                      "rocket": 0.38, "lightning": 0.40, "the_log": 0.09,
                      "poison": 0.29}
TOWER_FINISH_THRESHOLD = 0.32
KING_ALIVE_H = 0.0

_MENTOR_PATTERNS = {
    "finish": ("finish", "finishing", "last hit", "kill the tower", "chip out", "lethal"),
    "defend": ("defend", "defense", "protect", "hold", "stop push", "pull"),
    "pressure": ("pressure", "push", "aggressive", "attack", "force"),
    "tower": ("tower", "crown", "king", "princess"),
    "trade": ("trade", "elixir advantage", "positive trade", "value", "up elixir"),
    "counterpush": ("counter-push", "counterpush", "counter push", "opposite lane", "other lane"),
    "caution": ("caution", "careful", "wait", "conserve", "passive", "hold back", "leak"),
    "novelty": ("new", "unusual", "first time", "surprise", "unexpected", "unseen"),
    "elixir": ("elixir", "cycle", "save", "cheap", "bank"),
}
_LEGAL_SPOTS = ("bridge_left", "bridge_right", "left_defend",
                "right_defend", "center_back")


def referee_evaluate_damage(dmg_info, ally_alive=3, tick=0, ally_king=None):
    """Structured deterministic tower-damage evaluation.
    Returns dict: {reward_adjustment, components, evidence, source}.
    Never fabricates damage when hp_drop == 0 and towers_lost == 0 and king_dmg == 0."""
    hp_drop = max(0.0, float(dmg_info.get("ally_hp_drop", 0.0)))
    towers_lost = max(0, int(dmg_info.get("ally_towers_lost", 0)))
    king_dmg = max(0.0, float(dmg_info.get("ally_king_drop", 0.0)))
    king_next = bool(dmg_info.get("king_next", False))
    if ally_king is None:
        ally_king = dmg_info.get("ally_king", 1.0)
    if hp_drop <= 0.0 and towers_lost == 0 and king_dmg <= 0.0:
        return {
            "reward_adjustment": 0.0,
            "components": {},
            "evidence": {"ally_hp_drop": 0.0, "ally_towers_lost": 0, "ally_king_drop": 0.0},
            "source": "referee: no measured damage",
        }
    p = TOWER_DAMAGE_PER_FRACTION * hp_drop
    comps = {}
    if hp_drop > 0.0:
        comps["tower_hp"] = round(-TOWER_DAMAGE_PER_FRACTION * hp_drop, 3)
    if king_dmg > 0.0:
        pk = KING_DAMAGE_MULT * TOWER_DAMAGE_PER_FRACTION * king_dmg
        p += pk
        comps["king_damage"] = round(-pk, 3)
    if towers_lost > 0:
        base_loss = TOWER_LOST_PENALTY * towers_lost
        if king_next or (ally_alive is not None and ally_alive <= 1 + towers_lost):
            esc = base_loss * (KING_NEXT_MULT - 1.0)
            p += base_loss + esc
            comps["tower_loss"] = round(-base_loss, 3)
            comps["king_next_escalation"] = round(-esc, 3)
        else:
            p += base_loss
            comps["tower_loss"] = round(-base_loss, 3)
    if tick > 45 and (ally_king is None or float(ally_king) > 0.0):
        p *= ENDGAME_URGENCY
        comps["endgame_urgency"] = "x1.25"
    adj = max(DAMAGE_FLOOR, min(DAMAGE_CEIL, -abs(p)))
    return {
        "reward_adjustment": round(adj, 3),
        "components": comps,
        "evidence": {
            "ally_hp_drop": round(hp_drop, 4),
            "ally_towers_lost": towers_lost,
            "ally_king_drop": round(king_dmg, 4),
            "ally_alive": ally_alive,
            "tick": tick,
        },
        "source": "referee_rules",
    }


def referee_pain(dmg_info, ally_alive=3, tick=0, ally_king=None):
    """Deterministic tower-damage punishment. Rules carry evidence; nothing is
    invented when no damage flow is present (hp_drop==0 & lost==0 -> 0.0).
    Returns (punish_float, source_str) for direct drop-in compatibility."""
    res = referee_evaluate_damage(dmg_info, ally_alive, tick, ally_king)
    if res["reward_adjustment"] == 0.0:
        return 0.0, "referee: no measured damage"
    return res["reward_adjustment"], "referee_rules"


class CheapCycleMonitor:
    """Anti-farming: bounds repeated cheap placements during threatless lulls.
    Genuine repeated defense under a real threat is NEVER suppressed."""

    def __init__(self, maxlen=FARM_LOOKBACK):
        self.maxlen = maxlen
        self.recent = []                     # [(cost, threat_now)]
        self.suppressed = 0

    def record(self, cost, threat):
        self.recent.append((float(cost), bool(threat)))
        if len(self.recent) > self.maxlen:
            self.recent.pop(0)

    def farming(self, cost, threat_now=False):
        if threat_now:
            return False                     # active threat = genuine defense, never farming
        low = [1 for c, t in self.recent if not t and c <= 2.0]
        return len(low) >= CHEAP_CYCLE_FARM_THRESHOLD and float(cost) <= 2.0


def referee_evaluate(action_ctx, cheap_monitor=None):
    """Full structured Referee evaluation for a battle tick.
    Supports:
      1. Tower damage (if dmg_info provided in action_ctx)
      2. Finisher urgency (credit executed legal finisher; penalize missed legal finisher)
      3. Weakest-tower targeting (bounded bonus when known & weakest living enemy tower)
      4. Elixir trades (only when reliable player_cost and opponent_cost_est are provided;
         never invents hidden enemy elixir)
      5. Defensive effectiveness & counterpush opportunities
      6. Anti-farming (cheap-cycle guard with genuine-threat exemption)
      7. Contribution caps [REF_PENALTY_CAP, REF_BONUS_CAP]
    Returns dict: {reward_adjustment, components, evidence, source}."""
    comps = {}
    evidence = {}
    adj = 0.0

    threat = float(action_ctx.get("threat", 0.0) or 0.0)
    targeted = action_ctx.get("targeted_tower")            # ("left"|"right", hp)
    was_finisher = bool(action_ctx.get("finisher_executed", False))
    finisher_available = bool(action_ctx.get("finisher_available", False))
    finisher_legal = bool(action_ctx.get("finisher_legal", True))
    used_spell = action_ctx.get("spell_used", None)
    cost = action_ctx.get("cost", None)
    opp_cost = action_ctx.get("opponent_cost_est", None)   # only when reliably measured
    enemy_hp = action_ctx.get("enemy_hp", None)
    defended_threat_lane = bool(action_ctx.get("defended_threat_lane", False))
    counterpush_lane = bool(action_ctx.get("counterpush_lane", False))

    # 1. Optional embedded tower damage evaluation
    dmg_info = action_ctx.get("dmg_info")
    if dmg_info:
        dmg_res = referee_evaluate_damage(
            dmg_info,
            ally_alive=action_ctx.get("ally_alive", 3),
            tick=action_ctx.get("tick", 0),
        )
        if dmg_res["reward_adjustment"] != 0.0:
            adj += dmg_res["reward_adjustment"]
            comps["tower_damage"] = {
                "penalty": dmg_res["reward_adjustment"],
                "subcomponents": dmg_res["components"],
                "evidence": dmg_res["evidence"],
            }
            evidence["tower_damage"] = dmg_res["evidence"]

    # 2. Weakest-tower targeting: bounded bonus ONLY when targeted tower is known
    # and is the weakest living enemy tower
    if targeted and isinstance(enemy_hp, dict):
        side, hp = targeted
        el = float(enemy_hp.get("enemy_left", 1.0))
        er = float(enemy_hp.get("enemy_right", 1.0))
        living = [v for v in (el, er) if v > 0.03]
        if living and hp > 0.03:
            weakest_hp = min(living)
            if hp <= weakest_hp + 1e-9:
                b = WEAKEST_TARGET_BONUS
                adj += b
                comps["weakest_tower"] = {
                    "bonus": b,
                    "evidence": f"targeted {side}@{hp:.2f} is weakest living (L={el:.2f}, R={er:.2f})",
                }
                evidence["targeted_tower"] = {"side": side, "hp": hp, "weakest_hp": weakest_hp}

    # 3. Finisher urgency: credit executed legal finisher; penalize missed legal finisher
    if was_finisher and finisher_legal:
        adj += FINISH_CREDIT
        comps["finisher_executed"] = {
            "bonus": FINISH_CREDIT,
            "evidence": "legal verified kill shot executed",
        }
        evidence["finisher"] = "executed"
    elif finisher_available and finisher_legal and not was_finisher:
        # Penalize missed finisher when it was genuinely available and legal
        adj += MISSED_FINISH_PENALTY
        comps["missed_finisher"] = {
            "penalty": MISSED_FINISH_PENALTY,
            "evidence": f"legal kill shot available (spell={used_spell or 'in_hand'}) but not executed",
        }
        evidence["finisher"] = "missed_legal"

    # 4. Elixir trades: compare player and opponent elixir expenditures ONLY when
    # reliable cost estimates are available. Never invent hidden enemy elixir.
    if cost is not None and opp_cost is not None and float(opp_cost) > 0:
        c_val = float(cost)
        o_val = float(opp_cost)
        diff = o_val - c_val
        trade_adj = max(-ELIXIR_TRADE_CAP, min(ELIXIR_TRADE_CAP, round(diff * ELIXIR_TRADE_SCALE, 3)))
        if abs(trade_adj) >= 0.02:
            adj += trade_adj
            comps["elixir_trade"] = {
                "delta": trade_adj,
                "evidence": f"verified trade: spent {c_val:.1f} vs observed enemy {o_val:.1f} ({diff:+.1f})",
            }
            evidence["elixir_trade"] = {"player_cost": c_val, "opponent_cost_est": o_val, "net": diff}

    # 5. Defensive effectiveness & counterpush
    if defended_threat_lane and threat > 0.0:
        adj += DEFENSE_EFFECT_BONUS
        comps["defensive_effectiveness"] = {
            "bonus": DEFENSE_EFFECT_BONUS,
            "evidence": f"defended active threat lane (threat={threat:.2f})",
        }
    if counterpush_lane:
        adj += COUNTERPUSH_BONUS
        comps["counterpush"] = {
            "bonus": COUNTERPUSH_BONUS,
            "evidence": "attacked opposite lane after enemy commitment",
        }

    # 6. Anti-farming: suppress repetitive cheap cycling in threatless lulls;
    # exempt necessary repeated defense under genuine threat.
    if cheap_monitor is not None and cost is not None:
        has_threat = threat > 0.0
        cheap_monitor.record(cost, has_threat)
        if cheap_monitor.farming(cost, threat_now=has_threat):
            cheap_monitor.suppressed += 1
            adj += CHEAP_CYCLE_FARM_SUPPRESS
            comps["cheap_cycle_farm"] = {
                "penalty": CHEAP_CYCLE_FARM_SUPPRESS,
                "evidence": f"repeated cheap cycle (cost={cost}) with threat=0.0",
            }
            evidence["anti_farming"] = "suppressed"

    # 7. Contribution caps (match outcomes stay outside this cap)
    adj = max(REF_PENALTY_CAP, min(REF_BONUS_CAP, adj))
    return {
        "reward_adjustment": round(adj, 3),
        "components": comps,
        "evidence": evidence,
        "source": "referee_deterministic",
    }


def referee_adjust(action_ctx, cheap_monitor=None):
    """Bounded scenario adjustments layered on the shaped reward. Returns
    (adjustment, components) where every entry names its evidence."""
    res = referee_evaluate(action_ctx, cheap_monitor=cheap_monitor)
    return res["reward_adjustment"], res["components"]


def interpret_mentor(critique, recommended_spot=None, context=None):
    """Transparent structural mentor interpreter.
    Supports both signatures:
      1) interpret_mentor(critique_str, recommended_spot_str, context_dict)
      2) interpret_mentor(mentor_dict, towers_dict, mlxir_int)  [legacy bot_ml call]
    Parses words and phrases into structured, inspectable features:
      elixir, finish, defend, pressure, tower, trade, counterpush, caution, novelty
    Validates every recommendation against:
      - known legal spots
      - card affordability
      - current threat lane
      - mandatory physics-finisher decisions
    Returns dict with:
      critique, proposed_action, features, intents (alias), accepted, reason, source.
    This keyword interpreter is a limited structured parser, not proof of deep
    semantic understanding."""
    # Handle legacy signature: interpret_mentor(mentor_dict, towers_dict, mlxir_int)
    if isinstance(critique, dict):
        mentor_dict = critique
        towers_arg = recommended_spot if isinstance(recommended_spot, dict) else {}
        mlxir_arg = context if isinstance(context, (int, float)) else 10
        ctx = context if isinstance(context, dict) else {
            "towers": towers_arg,
            "mlxir": mlxir_arg,
        }
        raw_critique = mentor_dict.get("critique") or ""
        rec_spot = mentor_dict.get("recommended_spot")
    else:
        raw_critique = critique or ""
        rec_spot = recommended_spot
        ctx = context if isinstance(context, dict) else {}

    text = str(raw_critique).lower()
    features = {k: any(pat in text for pat in pats) for k, pats in _MENTOR_PATTERNS.items()}
    # Also keep "counter" alias in intents for backward compatibility with older tests
    intents = dict(features)
    intents["counter"] = features["counterpush"]

    towers = ctx.get("towers") or {}
    mlxir = float(ctx.get("mlxir", 10))
    finishers = ctx.get("finishers") or []
    legal = tuple(ctx.get("legal_spots") or _LEGAL_SPOTS)
    affordable_slots = ctx.get("affordable_slots")
    threat_lane = ctx.get("threat_lane")  # "left" | "right" | None
    physics_kill = ctx.get("physics_finisher")

    # 1. Check card affordability if provided
    if affordable_slots is not None and len(affordable_slots) == 0 and rec_spot is not None:
        return {
            "critique": raw_critique,
            "proposed_action": rec_spot,
            "features": features,
            "intents": intents,
            "accepted": False,
            "reason": "rejected: no affordable cards in hand",
            "source": "mentor_interpreter",
        }

    # 2. Check legal spot if a spot was proposed
    if rec_spot is not None and rec_spot not in legal:
        return {
            "critique": raw_critique,
            "proposed_action": rec_spot,
            "features": features,
            "intents": intents,
            "accepted": False,
            "reason": f"illegal spot {rec_spot}",
            "source": "mentor_interpreter",
        }

    # 3. Mandatory physics-finisher veto: if a verified physics finisher is pending,
    # any mentor recommendation that contradicts the finisher lane is rejected.
    active_finisher_sides = []
    if physics_kill and isinstance(physics_kill, dict) and physics_kill.get("side"):
        active_finisher_sides.append(physics_kill["side"])
    for f in finishers:
        if isinstance(f, dict) and f.get("side"):
            active_finisher_sides.append(f["side"])
    if active_finisher_sides:
        req_spots = {f"bridge_{s}" for s in active_finisher_sides}
        if rec_spot is not None and rec_spot not in req_spots:
            return {
                "critique": raw_critique,
                "proposed_action": rec_spot,
                "features": features,
                "intents": intents,
                "accepted": False,
                "reason": f"physics finisher pending on {active_finisher_sides[0]} overrides mentor spot {rec_spot}",
                "source": "mentor_interpreter",
            }

    # 4. Threat-lane consistency check: if severe imminent threat exists on one lane
    # and mentor proposes defending the wrong lane, reject.
    if threat_lane in ("left", "right") and rec_spot in ("left_defend", "right_defend"):
        expected_def = f"{threat_lane}_defend"
        if rec_spot != expected_def:
            return {
                "critique": raw_critique,
                "proposed_action": rec_spot,
                "features": features,
                "intents": intents,
                "accepted": False,
                "reason": f"threat is on {threat_lane} lane, rejected wrong-lane defense {rec_spot}",
                "source": "mentor_interpreter",
            }

    # 5. Evaluate semantic feature alignment against measured state
    accepted, reason = False, "no actionable intent"
    el_hp = float(towers.get("enemy_left", 1.0))
    er_hp = float(towers.get("enemy_right", 1.0))
    low_hp = min(el_hp, er_hp)

    if features["finish"]:
        # Check if targeted side (if mentioned in text or spot) is actually in finish range
        mentions_right = "right" in text or rec_spot == "bridge_right"
        mentions_left = "left" in text or rec_spot == "bridge_left"
        if mentions_right and not mentions_left:
            target_in_range = er_hp <= TOWER_FINISH_THRESHOLD or bool(active_finisher_sides)
        elif mentions_left and not mentions_right:
            target_in_range = el_hp <= TOWER_FINISH_THRESHOLD or bool(active_finisher_sides)
        else:
            target_in_range = low_hp <= TOWER_FINISH_THRESHOLD or bool(active_finisher_sides)

        if not target_in_range:
            accepted, reason = False, "critique=finish but target tower is not in finish range"
        elif rec_spot is not None and rec_spot not in ("bridge_left", "bridge_right"):
            accepted, reason = False, "finish intent but recommendation is not a finisher lane"
        else:
            accepted, reason = True, "critique=finish aligns with lowest-HP / finisher law"
    elif features["defend"]:
        accepted, reason = True, "critique=defend is legal and aligned"
    elif features["caution"] and mlxir <= 4:
        accepted, reason = True, "caution at low elixir is consistent"
    elif features["elixir"] and mlxir <= 4:
        accepted, reason = True, "critique=save consistent with low elixir"
    elif features["counterpush"]:
        if rec_spot is None or rec_spot in ("bridge_left", "bridge_right"):
            accepted, reason = True, "counterpush intent aligned with attacking lane"
        else:
            accepted, reason = False, "counterpush intent requires bridge placement"
    elif features["trade"] or features["pressure"] or features["tower"]:
        if rec_spot is None or rec_spot in legal:
            accepted, reason = True, "tactical intent validated against legal spots"
    elif rec_spot in legal:
        accepted, reason = True, "generic legal placement, no conflicting intent"

    return {
        "critique": raw_critique,
        "proposed_action": rec_spot,
        "features": features,
        "intents": intents,
        "accepted": accepted,
        "reason": reason,
        "source": "mentor_interpreter",
    }


def groq_damage_punishment(dmg_info, img=None):
    """Compatibility wrapper for old call sites: purely deterministic, keeps
    the (punish, source) two-tuple contract. NEVER contacts Groq."""
    p, _src = referee_pain(dmg_info, dmg_info.get("ally_alive", 3), 0)
    if p == 0.0 and float(dmg_info.get("ally_hp_drop", 0.0)) > 0:
        p = max(DAMAGE_CEIL, -max(0.5, round(dmg_info.get("ally_hp_drop", 0.0) * 4.0, 2)))
    return p, "referee_deterministic"
