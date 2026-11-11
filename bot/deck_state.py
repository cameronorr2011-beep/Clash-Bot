"""NeuroML deck_state — the deck TRUTH object (PROMPT 4 P1A).

Separates four concepts that the old code conflated in my_deck.json:
  active_deck       deck confirmed/equipped in-game (verified) — or UNCERTAIN.
  recommended_deck  a proposal that has NOT been applied (own file).
  observed_hand     the four cards actually visible in battle (from the tracker).
  deck_cycle_state  known/inferred order of cards entering the hand.

Never overwrite active with a recommendation; recommendations live in
recommended_deck.json and show as such on the dashboard. Uncertainty is
carried explicitly — never pretended away.
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ACTIVE_PATH = os.path.join(HERE, "active_deck.json")
RECO_PATH = os.path.join(HERE, "recommended_deck.json")


def _jload(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _jsave(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def load_active(default_cards=None, default_source="migrated-from-my_deck"):
    """The ACTIVE deck — but written back with an explicit verification flag.
    Old installs migrated: cards copied from my_deck.json, flagged UNVERIFIED
    (my_deck order was never proof of the in-game deck)."""
    st = _jload(ACTIVE_PATH, None)
    if st is None:
        mydeck = _jload(os.path.join(HERE, "my_deck.json"), {})
        cards = (mydeck.get("deck") if isinstance(mydeck, dict) else None) or default_cards or []
        st = {"deck": cards, "verified": False, "source": default_source,
              "updated": time.strftime("%Y-%m-%d %H:%M"),
              "uncertain_reason": "migrated from my_deck.json — never verified in-game"}
        _jsave(ACTIVE_PATH, st)
    st.setdefault("verified", False)
    st.setdefault("uncertain_reason", "unknown provenance" if not st.get("verified") else "")
    return st


def mark_verified_active(cards, source="in-game verified"):
    st = load_active()
    st["deck"] = list(cards)
    st["verified"] = True
    st["source"] = source
    st["updated"] = time.strftime("%Y-%m-%d %H:%M")
    st.pop("uncertain_reason", None)
    _jsave(ACTIVE_PATH, st)
    return st


def load_recommended():
    return _jload(RECO_PATH, None)


def set_recommended(cards, name, score, explanation, meta_asof=None):
    """Store a recommendation SEPARATELY. Never touches active_deck.json."""
    rec = {"deck": list(cards), "name": name, "score": round(score, 3),
           "explanation": explanation,
           "source": "deck-architect", "applied": False,
           "created": time.strftime("%Y-%m-%d %H:%M"),
           "meta_asof": meta_asof}
    _jsave(RECO_PATH, rec)
    return rec


def mark_recommended_applied(verified_cards, source="user confirmed"):
    """ONLY path from recommendation to active: explicit user confirmation or
    independent in-game verification. Records the transition in both files."""
    st = mark_verified_active(verified_cards, source=source)
    rec = load_recommended()
    if rec and list(rec.get("deck", [])) == list(verified_cards):
        rec["applied"] = True
        rec["applied_at"] = st["updated"]
        _jsave(RECO_PATH, rec)
    return st


def uncertain_active():
    st = load_active()
    if st.get("verified"):
        return None
    return st.get("uncertain_reason") or "active deck unverified"


def summary():
    st = load_active()
    rec = load_recommended()
    return {
        "active": st,
        "recommended": rec,
        "active_uncertain": uncertain_active(),
        "has_pending_recommendation": bool(rec and not rec.get("applied")),
    }
