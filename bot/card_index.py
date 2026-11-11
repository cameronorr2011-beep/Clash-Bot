"""NeuroML card_index — the ONE canonical card database + validation (v26).

Fixes the deck-blindness root cause: bot_ml.py used to load cards.json (13
cards, missing ice_golem / ice_spirit / the_log that the ACTIVE deck plays)
and silently substitute `cost=9, role="support"` for every unknown card —
so cards in the deck were unplayable/wrong-roled forever.

Rules here (PROMPT 4 P1B):
  * ONE canonical schema (cards_v2.json), aliases normalized by ONE function.
  * Entries carry source + confidence; unknown card = loud validation error.
  * cost/role/targets come from verified core facts; spell damage fractions
    are community estimates kept bounded and labeled (never invented).
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "cards_v2.json")

_DB = None
_ERRORS = []


def normalize_card(name):
    """THE canonical normalizer used everywhere (aliases + case + separators).
    Returns the canonical card id or the lowercased input if unrecognized —
    callers must check membership instead of inventing entries."""
    if not isinstance(name, str):
        return ""
    s = re.sub(r"[\s\-]+", "_", name.strip().lower()).strip("_")
    db = load()
    if s in db:
        return s
    for cid, cinfo in db.items():
        if cid.startswith("_"):
            continue
        if s in [a.replace(" ", "_") for a in cinfo.get("aliases", [])]:
            return cid
    return s


def alias_norm(setsingle):
    return {normalize_card(x) for x in setsingle}


def load():
    """Load cards_v2.json once. Missing DB = loud validation errors, never
    silent fake metadata."""
    global _DB, _ERRORS
    if _DB is None:
        try:
            with open(DB_PATH, "r", encoding="utf-8") as f:
                _DB = json.load(f)
        except Exception as e:                     # corrupt/missing → loud
            _DB = {}
            _ERRORS.append(f"cards_v2.json unreadable: {e}")
    return _DB


def info(card_name):
    """Canonical metadata for a RAW (probably aliased) name, or {} if unknown.
    NO fabricated cost/role — the caller must handle unknown itself."""
    db = load()
    return db.get(normalize_card(card_name), {})


def cost(card_name, strict=True):
    c = info(card_name).get("cost")
    if c is None:
        if strict:
            raise KeyError(f"unknown card {card_name!r}: no cost in card_index "
                           f"(deck validation must gate this before tactics)")
        return None
    return c


def role(card_name):
    return info(card_name).get("role")


def is_spell(card_name):
    return role(card_name) == "spell"


def spell_profile(card_name):
    """Spell behavior from canonical data (kind/geometry/delay/reaches_towers/
    tower_damage_frac). Non-spells → None. Damage fractions are community
    estimates, explicitly labeled by source in the DB."""
    return info(card_name).get("spell")


def validate_deck(cards, source="active"):
    """Strict deck validation (P1B/P3). Returns (ok, problems, details).
    * exactly 8 cards, all known, no duplicates, valid costs,
    * at least one credible win condition, role coverage summarized;
      problems are DATA facts."""
    problems, details = [], {}
    if not isinstance(cards, list):
        return False, ["deck is not a list"], {}
    norm = [normalize_card(c) for c in cards]
    unknown = [(raw, n) for raw, n in zip(cards, norm) if not info(raw)]
    if unknown:
        problems.append(
            "unknown cards (no verified metadata — REJECTED, not substituted): "
            + ", ".join(sorted({raw for raw, _ in unknown})))
    dups = sorted({c for c in norm if norm.count(c) > 1})
    if dups:
        problems.append("duplicate cards: " + ", ".join(dups))
    if len(cards) != 8:
        problems.append(f"deck has {len(cards)} cards, expected 8")
    costs = []
    winconds = []
    anti_air = []
    anti_tank = []
    spells = []
    for raw, nid in zip(cards, norm):
        inf = info(raw)
        if not inf:
            continue
        c = inf.get("cost")
        if not isinstance(c, int) or not (1 <= c <= 9):
            problems.append(f"{nid}: bad cost {c!r}")
            continue
        costs.append(c)
        sub = inf.get("subtype", "")
        if inf.get("role") in ("win_condition", "tank") or "win" in sub:
            winconds.append((nid, sub))
        t = inf.get("targets", {})
        if inf.get("role") == "spell":
            spells.append((nid, inf.get("spell") or {}))
        if t.get("air"):
            anti_air.append(nid)
        if ("tank" in sub or "beast" in sub
                or any("tank" in x for x in inf.get("counters_hard", []))):
            anti_tank.append(nid)
    if costs:
        details["avg_elixir"] = round(sum(costs) / len(costs), 2)
    details["win_conditions"] = [w[0] for w in winconds]
    details["anti_air"] = anti_air
    details["anti_tank"] = anti_tank
    details["spells"] = [s[0] for s in spells]
    if not winconds:
        problems.append("no credible win condition (no win_condition/tank card)")
    if len(anti_air) == 0 and not spells:
        problems.append("no air defense at all")
    ok = not problems
    return ok, problems, details


def deck_summary(cards):
    ok, problems, details = validate_deck(cards)
    return {"ok": ok, "problems": problems, **details,
            "unknown_after_norm": [normalize_card(c) for c in cards]}


def meta_supported(meta_path=None):
    """P3: verify every meta candidate against canonical metadata; label age
    and source; drop/flag invalid ones. Returns (valid_entries, rejected)."""
    path = meta_path or os.path.join(HERE, "meta_decks.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as e:
        return [], [{"name": "?", "reason": f"meta_decks.json unreadable: {e}"}]
    asof = meta.get("fetched_at", "unknown-date")
    src = meta.get("fetched_from", "unknown-source")
    valid, rejected = [], []
    for d in meta.get("top_meta_decks", []):
        ok, problems, details = validate_deck(d.get("cards", []), source="meta")
        entry = dict(d)
        entry["meta_asof"] = asof
        entry["meta_source"] = src
        entry["validated"] = ok
        entry["validation"] = problems or "ok"
        entry["_details"] = details
        (valid if ok else rejected).append(entry)
    return valid, rejected


if __name__ == "__main__":
    db = load()
    live = ["hogrider", "musketeer", "ice_golem", "cannon",
            "skeletons", "ice_spirit", "fireball", "the_log"]
    ok, probs, det = validate_deck(live)
    print(f"active deck ok={ok} details={det}")
    for p in probs:
        print("  PROBLEM:", p)
    valid, rejected = meta_supported()
    print(f"meta decks: {len(valid)} valid, {len(rejected)} rejected")
    for r in rejected:
        v = r["validation"]
        print("  REJECTED:", r.get("name"), "->", "; ".join(v) if isinstance(v, list) else v)
