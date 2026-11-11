"""Tactical placement layer — the deterministic half of "don't play stupid".

The DQN proposes a play, referee.py scores it after the fact; this module is the
*before* check on placement alone. It exists because the same two errors kept
showing up in the reward ledger and in the mentor feed, and neither is a
learning problem — both are decidable from the current observation:

  1. Defending an empty lane (or dropping center_back) at 7+ elixir, i.e.
     converting nothing while the opponent banks elixir.
  2. Defending the wrong lane while a live push is on the other side.

Pure functions: no torch, no I/O, no globals mutated. Safe to unit test and safe
to call on the phone where torch is expensive.

Placement indices are resolved from the BOT's own spot-name list, never
hardcoded: bot_ml.SPOT_NAMES is ["bridge_left", "bridge_right", "left_defend",
"right_defend", "center_back"], so index 4 is center_back — the exact opposite
end of the list from "right". Passing the list in keeps this module correct for
both the desktop and phone builds.

API
---
advise(action, elixir, hand_deck, afford, towers, arena, cards_db,
       spot_names=SPOT_NAMES) -> (action, note)

`note` is "" when the plan is already sound. A non-empty note is a short rule
label the bot logs verbatim (it surfaces in the dashboard's decision
provenance). Callers must treat the NOTE, not the action, as the override
signal: returning the same action with an empty note means "checked, no change".
"""

SPOT_NAMES = ("bridge_left", "bridge_right", "left_defend", "right_defend", "center_back")

# A lane counts as threatened at 4+ px of enemy health bar in it (the vision
# scanner reports bar pixels, not unit counts — 4px is roughly one mid troop).
THREAT_PX = 4
# At or above this elixir with no threat, defending is a wasted turn.
PUSH_ELIXIR = 7.0


def _num(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f else default          # NaN-safe


def _info(cards_db, name):
    """Card record from the shared DB, tolerant of naming style."""
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


def _cost(cards_db, name, default=4):
    return int(_num(_info(cards_db, name).get("cost"), default))


def spot_index(spot_names, name):
    """Index of a spot by NAME in the bot's own list; None if the bot lacks it."""
    try:
        return list(spot_names).index(name)
    except (ValueError, AttributeError):
        return None


def threat_lanes(arena, threat_px=THREAT_PX):
    """(left_px, right_px) of enemy presence per lane.

    The scanner exposes both a lane total and a bridge column; either one means
    the opponent has committed troops there, so they are summed.
    """
    arena = arena or {}
    left = _num(arena.get("enemy_left")) + _num(arena.get("enemy_bridge_l"))
    right = _num(arena.get("enemy_right")) + _num(arena.get("enemy_bridge_r"))
    return left, right


def weak_side(towers):
    """Which enemy tower is cheapest to push — the lower HP one still standing."""
    towers = towers or {}
    alive = [(_num(towers.get("enemy_left"), 1.0), "left"),
             (_num(towers.get("enemy_right"), 1.0), "right")]
    alive = [(hp, side) for hp, side in alive if hp > 0.03]
    if not alive:
        return None
    return min(alive)[1]


def advise(action, elixir, hand_deck, afford, towers, arena, cards_db,
           spot_names=SPOT_NAMES, threat_px=THREAT_PX, push_elixir=PUSH_ELIXIR):
    """Return (possibly corrected action, rule note or "")."""
    hand_deck = hand_deck or []
    afford = afford or []
    action = int(action)
    slot, spot_i = divmod(action, 5)

    # Resolve every placement this module can redirect, by name, once.
    i_bridge = {"left": spot_index(spot_names, "bridge_left"),
                "right": spot_index(spot_names, "bridge_right")}
    i_defend = {"left": spot_index(spot_names, "left_defend"),
                "right": spot_index(spot_names, "right_defend")}
    i_center = spot_index(spot_names, "center_back")
    if None in i_bridge.values() or None in i_defend.values() or i_center is None:
        return action, ""                    # unknown spot model: never guess
    passive = {i_center, i_defend["left"], i_defend["right"]}

    playable = [i for i in range(min(4, len(hand_deck)))
                if i < len(afford) and _num(afford[i]) >= 0.9]
    if not playable:
        return action, ""
    # Rule 0 — a play you cannot pay for is not a play.
    if slot not in playable:
        keep = min(playable, key=lambda i: _cost(cards_db, hand_deck[i]))
        return keep * 5 + spot_i, f"afford:{hand_deck[keep]}"

    left, right = threat_lanes(arena, threat_px)
    spot = spot_names[spot_i] if 0 <= spot_i < len(spot_names) else ""

    # Rule 1 — live push: answer the lane that is actually under attack.
    if left >= threat_px or right >= threat_px:
        side = "left" if left >= right else "right"
        if spot != spot_names[i_defend[side]]:
            # a same-side bridge drop is a legitimate counter-push
            countering = spot_i == i_bridge[side]
            if not countering:
                return slot * 5 + i_defend[side], \
                    f"defend_{side} (push {max(left, right):.0f}px)"
        return action, ""

    # Rule 2 — nothing to defend: do not burn elixir on defence/center at a
    # high bank. Push the cheaper tower instead of converting nothing.
    if _num(elixir) >= push_elixir and spot_i in passive:
        side = weak_side(towers)
        if side:
            return slot * 5 + i_bridge[side], \
                f"pressure_{side} (no threat at {_num(elixir):.0f} elixir)"

    return action, ""
