"""Tactical placement layer — the deterministic half of "don't play stupid".

The DQN proposes a play; referee.py scores it afterwards; this module is the
*before* check. It is the rule-brain pattern used by the working open-source
Clash Royale bots (sacha9214/clash-ai: "cheapest defender that wins the duel",
never a Fireball on a lone Giant, counter-push behind surviving defenders;
itzik123/ClashRoyaleAi: per-card legality masks; Duncan-17/clash-royale-ai-bot:
rule-based policy on top of perception), reduced to what this bot's vision can
actually support.

Three faults it removes, none of which is a learning problem:

  1. Answering the wrong lane while a push is live.
  2. Spending elixir on defence when nothing is coming, or converting nothing
     at a high bank instead of pushing the cheaper tower.
  3. Dropping a spell at a small push — a 4-elixir Fireball on one troop is a
     negative elixir trade no matter what the Q-values say.

Placement indices are resolved from the BOT's own spot-name list, never
hardcoded: bot_ml.SPOT_NAMES is ["bridge_left", "bridge_right", "left_defend",
"right_defend", "center_back"], so index 4 is center_back.

API
---
advise(action, elixir, hand_deck, afford, towers, arena, cards_db,
       spot_names=SPOT_NAMES) -> (action, note)

`note` is "" when the plan is already sound; a non-empty note is the rule label
the bot logs (it shows in the dashboard's decision provenance). Callers treat
the NOTE, not the action, as the override signal.
"""

SPOT_NAMES = ("bridge_left", "bridge_right", "left_defend", "right_defend", "center_back")

THREAT_PX = 4        # enemy bar pixels in a lane that count as a live push
PUSH_ELIXIR = 7.0    # bank from which doing nothing is a waste
SWARM_PX = 12        # only at this much enemy presence is a spell worth casting
SUPPORT_ALLY_PX = 4  # surviving friendly pixels worth pushing behind

DEFENSIVE_ROLES = ("support", "building", "win_condition", "tank")


def _num(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f else default          # NaN-safe


def _info(cards_db, name):
    key = (name or "").lower()
    if not key:
        return {}
    if key in cards_db:
        return cards_db[key] or {}
    try:
        import card_index
        return cards_db.get(card_index.normalize_card(key), {}) or {}
    except Exception:
        return {}


def _role(cards_db, name):
    return _info(cards_db, name).get("role", "support")


def _cost(cards_db, name, default=4):
    return int(_num(_info(cards_db, name).get("cost"), default))


def spot_index(spot_names, name):
    try:
        return list(spot_names).index(name)
    except (ValueError, AttributeError):
        return None


def threat_lanes(arena, threat_px=THREAT_PX):
    arena = arena or {}
    left = _num(arena.get("enemy_left")) + _num(arena.get("enemy_bridge_l"))
    right = _num(arena.get("enemy_right")) + _num(arena.get("enemy_bridge_r"))
    return left, right


def ally_presence(arena):
    arena = arena or {}
    return _num(arena.get("ally_left")), _num(arena.get("ally_right"))


def weak_side(towers):
    towers = towers or {}
    alive = [(_num(towers.get("enemy_left"), 1.0), "left"),
             (_num(towers.get("enemy_right"), 1.0), "right")]
    alive = [(hp, side) for hp, side in alive if hp > 0.03]
    if not alive:
        return None
    return min(alive)[1]


def playable_slots(hand_deck, afford):
    return [i for i in range(min(4, len(hand_deck or [])))
            if i < len(afford or []) and _num(afford[i]) >= 0.9]


def cheapest_defender(hand_deck, afford, cards_db, allow_spell=False):
    """Cheapest affordable card that can actually defend -> (rank, slot, role, name).

    Support/building beat a win-condition at equal cost; spells are excluded
    unless the caller explicitly allows them.
    """
    best = None
    for i in playable_slots(hand_deck, afford):
        name = (hand_deck[i] or "").lower()
        role = _role(cards_db, name)
        if role == "spell" and not allow_spell:
            continue
        penalty = 0 if role in ("support", "building") else 1
        rank = (_cost(cards_db, name), penalty, i)
        if best is None or rank < best[0]:
            best = (rank, i, role, name)
    return best


def advise(action, elixir, hand_deck, afford, towers, arena, cards_db,
           spot_names=SPOT_NAMES, threat_px=THREAT_PX, push_elixir=PUSH_ELIXIR,
           swarm_px=SWARM_PX, support_ally_px=SUPPORT_ALLY_PX):
    """Return (possibly corrected action, rule note or "")."""
    hand_deck = hand_deck or []
    afford = afford or []
    action = int(action)
    slot, spot_i = divmod(action, 5)

    i_bridge = {"left": spot_index(spot_names, "bridge_left"),
                "right": spot_index(spot_names, "bridge_right")}
    i_defend = {"left": spot_index(spot_names, "left_defend"),
                "right": spot_index(spot_names, "right_defend")}
    i_center = spot_index(spot_names, "center_back")
    if None in i_bridge.values() or None in i_defend.values() or i_center is None:
        return action, ""
    passive = {i_center, i_defend["left"], i_defend["right"]}

    playable = playable_slots(hand_deck, afford)
    if not playable:
        return action, ""
    # Rule 0 — a play you cannot pay for is not a play.
    if slot not in playable:
        fell = cheapest_defender(hand_deck, afford, cards_db, allow_spell=True)
        keep = fell[1] if fell else playable[0]
        return keep * 5 + spot_i, f"afford:{hand_deck[keep]}"

    left, right = threat_lanes(arena, threat_px)
    cur_name = (hand_deck[slot] or "").lower()
    cur_role = _role(cards_db, cur_name)

    # Rule 1 — a live push: answer that lane, with troops, cheaply.
    if left >= threat_px or right >= threat_px:
        side = "left" if left >= right else "right"
        px = max(left, right)
        target_spot = i_defend[side]
        if cur_role == "spell" and px < swarm_px:
            d = cheapest_defender(hand_deck, afford, cards_db)
            if d:
                return d[1] * 5 + target_spot, \
                    f"defend_{side} with {d[3]} (no spell for a {px:.0f}px push)"
        if spot_i != target_spot:
            return slot * 5 + target_spot, f"defend_{side} (push {px:.0f}px)"
        return action, ""

    # Rule 2 — nothing incoming: convert the bank instead of defending air.
    if _num(elixir) >= push_elixir and spot_i in passive:
        al, ar = ally_presence(arena)
        if max(al, ar) >= support_ally_px:
            # counter-push behind troops that already survived a defence
            side = "left" if al >= ar else "right"
            return slot * 5 + i_bridge[side], \
                f"support_{side} (allies {max(al, ar):.0f}px already on the field)"
        side = weak_side(towers)
        if side:
            return slot * 5 + i_bridge[side], \
                f"pressure_{side} (no threat at {_num(elixir):.0f} elixir)"

    return action, ""
