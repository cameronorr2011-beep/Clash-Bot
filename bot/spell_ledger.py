"""NeuroML spell_ledger — evidence-based spell accounting (PROMPT 4 P2A/B).

Replaces the broken detector that credited a spell whenever "enemies existed
somewhere in the arena pre-cast" and indexed the 8-card deck file by the OLD
hand-slot number after the hand had rotated.

Design:
  * record() captures a structured snapshot AT CAST TIME: verified card
    identity (HandTracker), hand slot at cast, cast coordinates, intended
    lane, pre-cast tower HP + pixel evidence of enemy units inside the
    spell's own effect area, and the expected effect geometry from
    cards_v2.json (radius / path / zone, impact delay, tower fraction).
  * evaluate() is called on LATER frames, inside the spell's OWN window
    (instant ~2.5 s; delayed/persistent per its kind), and votes on
    in-area evidence ONLY:
       CONNECTED  – in-area enemy evidence fell AND (tower chip if aimed
                    at tower) — the spell did measurable work.
       WASTED     – the spell could demonstrably do work but in-area
                    evidence stayed flat through its window (miss).
       UNKNOWN    – evidence absent or contradictory (missing frame,
                    occlusion, vision noise). NO reward, NO punishment —
                    the honest third outcome.
  * exactly-once attribution per cast_id; each cast resolves exactly one
    verdict and one reward, and the verdict is what gets logged (P6).
  * REAL pain: a verified WASTED spell costs its elixir with interest —
    the deterministic penalty -(elixir * PAIN_PER_ELIXIR_WASTED) is fed to
    the same pain-channel machinery tower damage uses, bounded, and the
    UNKNOWN outcome is never punished. Nothing LLM-derived enters here.
"""
import time

# Deterministic, bounded pain scale: per elixir of verified waste.
PAIN_PER_ELIXIR_WASTED = 0.55
# In-area pixel evidence thresholds (enemy HP-bar px, from scan-style vision).
MIN_DROP_FOR_CONNECTED = 2          # px of red bar mass that must leave
FLAT_EPS = 1                        # <=1 px swing = no evidence either way

RESULT_CONNECT = "CONNECTED"
RESULT_WASTED = "WASTED"
RESULT_UNKNOWN = "UNKNOWN"


def collect_bars(img, y0=480, y1=1650, x0=100, x1=980, step=8):
    """Collect the red enemy HP-bar pixel evidence a cast verdict scans.
    Same predicate as bot_ml.scan_arena_vision so counts agree exactly."""
    px = img.getpixel
    out = []
    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            r, g, b = px((x, y))[:3]
            if r > 180 and g < 70 and b < 95 and r > b + 90:
                out.append((x, y, r, g, b))
    return out


def frame_envidence(frame_stats, cx, cy, r_frac, screen_w, screen_h):
    """Sum red HP-bar pixels inside the spell's area. Uses the same bar-color
    predicate as bot_ml.scan_arena_vision (r>180,g<70,b<95) so cast verdicts
    and arena counts agree. Radius is a fraction of screen WIDTH."""
    if frame_stats is None:
        return None
    hits = 0
    r = int(r_frac * screen_w)
    for (x, y, r_, g_, b_) in frame_stats:
        dx = x - cx
        dy = y - cy
        if dx * dx + dy * dy <= r * r:
            if r_ > 180 and g_ < 70 and b_ < 95 and r_ > b_ + 90:
                hits += 1
    return hits


class SpellLedger:
    def __init__(self):
        self.casts = []          # active (unresolved) records
        self.resolved = []       # verdicts, newest last (cap for log size)
        self._seen_ids = set()   # exactly-once
        self._next_id = 1

    # ---------- recording (called at the moment of the cast) ----------
    def record(self, card, hand_slot, spot_at_cast, cast_xy, pre_frame_stats,
               pre_towers, card_info_spell, screen_w, screen_h, lane=None,
               finisher=False, tick=0):
        sp = card_info_spell or {}
        kind = sp.get("kind", "radius")
        rec = {
            "id": self._next_id,
            "card": card,
            "slot_at_cast": hand_slot,
            "spot_at_cast": spot_at_cast,
            "cast_xy": cast_xy,
            "lane": lane or ("left" if (cast_xy or (0, 0))[0] < screen_w / 2 else "right"),
            "pre": {"towers": dict(pre_towers or {}),
                    "in_area_px": None,
                    "frame_stats": pre_frame_stats,
                    "_stats_shape": (screen_w, screen_h)},
            "profile": sp,
            "finisher": bool(finisher),
            "tick": tick,
            "t_cast": time.time(),
            "deadline": time.time() + self._window_s(sp),
            "outcome": None,
            "evidence": [],
            "confidence": 0.0,
        }
        if pre_frame_stats is not None:
            rec["pre"]["in_area_px"] = self._area_hit_count(rec)
        self.casts.append(rec)
        self._next_id += 1
        return rec["id"]

    def _window_s(self, sp):
        kind = sp.get("kind", "radius")
        delay = float(sp.get("impact_delay_s", 0.0) or 0.0)
        dur = float(sp.get("duration_s", 0.0) or 0.0)
        base = 2.5                              # two battle ticks to see work
        if kind in ("zone", "delivery"):
            return 4.0 + dur
        if kind == "path":
            return 3.2 + delay
        return base + delay

    def _area_hit_count(self, rec):
        w, h = rec["pre"]["_stats_shape"]
        prof = rec["profile"]
        r_frac = float(prof.get("radius_px_frac", 0.12) or 0.12)
        if prof.get("kind") == "path":
            r_frac = max(0.06, 0.5 * float((prof.get("path_len_px_frac") or [0, 0.2])[1]))
        cx, cy = rec["cast_xy"] or (int(w * 0.5), int(h * 0.55))
        return frame_envidence(rec["pre"]["frame_stats"], cx, cy, r_frac, w, h)

    # ---------- resolution (called on later vision ticks) ----------
    def evaluate(self, post_frame_stats, post_towers, now=None):
        now = now or time.time()
        verdicts = []
        for rec in list(self.casts):
            if rec["outcome"] is not None:
                continue
            cast_xy = rec["cast_xy"]
            if cast_xy is None:
                self._settle(rec, RESULT_UNKNOWN, "no cast coordinates recorded",
                             0.0, verdicts)
                continue
            w, h = rec["pre"]["_stats_shape"]
            prof = rec["profile"]
            r_frac = float(prof.get("radius_px_frac", 0.12) or 0.12)
            if prof.get("kind") == "path":
                r_frac = max(0.06, 0.5 * float((prof.get("path_len_px_frac") or [0, 0.2])[1]))
            post_px = frame_envidence(post_frame_stats, cast_xy[0], cast_xy[1],
                                      r_frac, w, h)
            pre_px = rec["pre"]["in_area_px"]
            if pre_px is None or post_px is None:
                self._settle_if_expired(rec, now, verdicts, "no frame evidence")
                continue
            drop = max(0, pre_px - post_px)
            # tower chip evidence only counts for the tower the spell was aimed at
            tw = None
            if prof.get("reaches_towers"):
                el_new = float(post_towers.get("enemy_left", 1.0))
                er_new = float(post_towers.get("enemy_right", 1.0))
                el_old = float(rec["pre"]["towers"].get("enemy_left", 1.0))
                er_old = float(rec["pre"]["towers"].get("enemy_right", 1.0))
                side = rec["lane"]
                dpora = max(0.0, (el_old - el_new) if side == "left" else (er_old - er_new))
                tw = dpora

            has_units_pre = pre_px >= 1
            if has_units_pre and drop >= MIN_DROP_FOR_CONNECTED:
                self._settle(rec, RESULT_CONNECT,
                             f"in-area enemy evidence fell {pre_px}->{post_px}px",
                             0.8, verdicts, extra={"tower_drop": tw})
            elif (not has_units_pre) and rec["finisher"]:
                # finisher aimed at a bare tower: judge by tower drop
                drop_t = tw or 0.0
                if drop_t >= 0.02:
                    self._settle(rec, RESULT_CONNECT,
                                 f"tower fell {drop_t*100:.0f}% in effect window",
                                 0.8, verdicts, extra={"tower_drop": tw})
                elif now > rec["deadline"]:
                    self._settle(rec, RESULT_WASTED,
                                 f"finisher hit bare tower, chip drop {drop_t*100:.0f}% "
                                 f"< threshold in window", 0.6, verdicts, extra={"tower_drop": tw})
                # else: window still open
            elif abs(pre_px - post_px) <= FLAT_EPS and has_units_pre:
                self._settle_if_expired(rec, now, verdicts,
                                        f"in-area enemy evidence flat ({pre_px}px "
                                        f"through the window) — spell did nothing")
            else:
                self._settle_if_expired(rec, now, verdicts,
                                        "ambiguous in-area evidence")
        return verdicts

    def _settle_if_expired(self, rec, now, verdicts, reason):
        if now > rec["deadline"]:
            # Once the window expires: flat pre-existing enemies + no work
            # done = a REAL miss (demonstrably castable, provably harmless).
            # Flat-with-no-enemies pre → the spell had nothing to hit: WASTED.
            rec_evidence_px = rec["pre"]["in_area_px"]
            if rec_evidence_px is None:
                self._settle(rec, RESULT_UNKNOWN, reason + " (no evidence)", 0.2, verdicts)
            elif rec_evidence_px == 0 and not rec["finisher"]:
                self._settle(rec, RESULT_WASTED,
                             "spell landed on empty ground (no in-area targets pre-cast)",
                             0.7, verdicts)
            else:
                self._settle(rec, RESULT_UNKNOWN, reason, 0.4, verdicts)

    def _settle(self, rec, outcome, reason, conf, verdicts, extra=None):
        rec["outcome"] = outcome
        rec["reason"] = reason
        rec["confidence"] = conf
        if extra:
            rec["evidence"].append(extra)
        if rec["id"] not in self._seen_ids:
            self._seen_ids.add(rec["id"])
            verdicts.append({
                "cast_id": rec["id"], "card": rec["card"],
                "slot_at_cast": rec["slot_at_cast"],
                "spot_at_cast": rec["spot_at_cast"],"cast_xy": rec["cast_xy"],
                "outcome": outcome, "reason": reason, "confidence": conf,
                "finisher": rec["finisher"], "cast_at": rec["t_cast"],
            })
        try:
            self.casts.remove(rec)
            self.resolved.append(rec)
            del self.resolved[:-500]
        except ValueError:
            pass
        return verdicts

    # ---------- reward helpers (deterministic; the referee owns caps) ----------
    @staticmethod
    def pain_for(outcome, card_cost):
        """REAL pain: verified waste costs its elixir with interest.
        CONNECTED → 0 (credit flows via the normal dopamine path).
        UNKNOWN → 0.0 — never punish uncertainty (P2A)."""
        if outcome == RESULT_WASTED and card_cost:
            return -round(PAIN_PER_ELIXIR_WASTED * float(card_cost), 3)
        return 0.0

    @staticmethod
    def credit_for(outcome, card_cost, finisher):
        if outcome != RESULT_CONNECT:
            return 0.0
        base = 0.25 + 0.15 * float(card_cost or 0)   # connected value ∝ elixir
        if finisher:
            base += 0.8
        return round(min(base, 2.2), 3)

    def stats(self):
        n = len(self.resolved)
        if not n:
            return {"casts_resolved": 0, "connected": 0, "wasted": 0, "unknown": 0,
                    "waste_rate": 0.0, "pending": len(self.casts)}
        c = sum(1 for r in self.resolved if r["outcome"] == RESULT_CONNECT)
        w = sum(1 for r in self.resolved if r["outcome"] == RESULT_WASTED)
        u = sum(1 for r in self.resolved if r["outcome"] == RESULT_UNKNOWN)
        return {"casts_resolved": n, "connected": c, "wasted": w, "unknown": u,
                "waste_rate": round(w / max(1, c + w + u), 3),
                "pending": len(self.casts)}

    def dump(self, limit=40):
        return [{
            "id": r["id"], "card": r["card"], "outcome": r["outcome"],
            "reason": r.get("reason", ""), "t": r["t_cast"],
        } for r in self.resolved[-limit:]]


_RESULT_LABEL = {RESULT_CONNECT: "✨ CONNECTED",
                 RESULT_WASTED: "🚫 WASTED",
                 RESULT_UNKNOWN: "❔ UNKNOWN"}


def label(outcome):
    return _RESULT_LABEL.get(outcome, outcome)
