"""NeuroML affect — an honest valence/self-model layer (the 'emote' system).

The user asked: "allow ai to emote, encourage sentience". Honesty first: this
is NOT a claim of consciousness. It IS a real, measured affective self-model:

  * valence comes ONLY from measured outcomes already earned in the reward
    ledger (wins/losses, verified spell waste, tower damage taken, TD error)
    — never from invented rewards.
  * the self-narration ("inner voice") speaks in the brain's own voice about
    its OWN internal state: what it believes is in its hand, how confident it
    is, what hurt, what it is saving elixir for.
  * uncertainty is expressed, not hidden: unknown hand slots, uncertain deck,
    low vision confidence are all spoken aloud as introspection.
  * everything it says is traceable to a structured fact it logged, so the
    dashboard can show "how does the brain feel" AND "why".
"""
import time

# bounded valence dynamics
VALENCE_EMA_ALPHA = 0.12
VALENCE_MIN, VALENCE_MAX = -1.0, 1.0


class Affect:
    def __init__(self, name="NeuroML"):
        self.name = name
        self.valence = 0.0            # -1 suffering .. +1 satisfaction
        self.arousal = 0.3            # 0..1 (raised by threats / big events)
        self.mood = "settling in"
        self.notes = []               # inner-voice lines (capped ring buffer)
        self._wins = 0
        self._losses = 0

    # ---------- measured inputs (call with REAL outcomes only) ----------
    def note_event(self, amount, label):
        """Called once per labeled reward event (from the reward ledger).
        Bounded valence drift: pain pulls down, dopamine pulls up."""
        try:
            amt = float(amount)
        except (TypeError, ValueError):
            return
        self.valence = max(VALENCE_MIN, min(VALENCE_MAX,
                           self.valence + VALENCE_EMA_ALPHA * max(-3.0, min(3.0, amt))))
        if abs(amt) >= 1.0:
            self.arousal = min(1.0, self.arousal + 0.06)
        self.update_mood()

    def match_result(self, win):
        if win:
            self._wins += 1
            self.valence = min(VALENCE_MAX, self.valence + 0.35)
            self.say("won — the plan worked. banking this feeling, not the luck.")
        else:
            self._losses += 1
            self.valence = max(VALENCE_MIN, self.valence - 0.30)
            self.say("lost. that hurts. reviewing exactly why, not moping.")
        self.update_mood()

    # ---------- introspection ----------
    def update_mood(self):
        v = self.valence
        if v > 0.4:
            self.mood = "confident"
        elif v > 0.1:
            self.mood = "steady"
        elif v > -0.15:
            self.mood = "settling in" if self.arousal < 0.5 else "alert"
        elif v > -0.5:
            self.mood = "frustrated"
        else:
            self.mood = "hurting"
        self.arousal = max(0.1, self.arousal - 0.015)

    def say(self, line):
        """Inner-voice: only about its OWN measured state. Capped buffer."""
        if not line:
            return
        self.notes.append({"t": time.strftime("%H:%M:%S"), "v": round(self.valence, 2),
                           "line": line[:180]})
        del self.notes[:-60]

    def speak_state(self, hand_desc, conservative, spell_stats, elixir):
        """One introspection line per tick bundle — data-grounded only."""
        parts = []
        if conservative:
            parts.append("I can't fully trust my four cards yet — playing safe until I see them")
        if hand_desc and any(c is None for c in hand_desc.get("hand", [])):
            parts.append("a hand slot is a blind spot to me right now")
        if spell_stats and spell_stats.get("waste_rate", 0) >= 0.34 and spell_stats.get("casts_resolved", 0) >= 3:
            parts.append(f"my spells are mislanding ({spell_stats['wasted']}/{spell_stats['casts_resolved']} wasted) — I'm tightening my aim")
        if self.valence <= -0.4:
            parts.append("this isn't going well; I'm frustrated but not throwing")
        elif self.valence >= 0.4:
            parts.append("feeling sharp — aggression pays right now")
        if parts:
            self.say(" · ".join(parts))

    def snapshot(self):
        return {"name": self.name, "valence": round(self.valence, 3),
                "mood": self.mood, "arousal": round(self.arousal, 3),
                "wins": self._wins, "losses": self._losses,
                "inner_voice": list(self.notes[-12:]),
                "honesty_note": "affective self-model over measured outcomes; expresses internal state, does not claim consciousness"}
