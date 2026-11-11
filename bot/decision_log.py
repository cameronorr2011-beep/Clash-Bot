"""NeuroML decision_log — one structured record per battle tick (PROMPT 4 P6).

Answers, from the log alone:
  "What did the AI think was in its hand, why did it cast this spell, what
   evidence showed whether the spell worked, and how did that outcome change
   the model?"
One JSON line per tick in decisions.jsonl (rotating cap), plus a periodic
JSON blob the dashboard can read — no decorative numbers, every field measured.
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "decisions.jsonl")
CAP = 4000          # rolling file: keep the last CAP lines


def _trim():
    try:
        n = os.path.getsize(PATH) if os.path.exists(PATH) else 0
        if n < 2_000_000:
            return
        with open(PATH, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        with open(PATH, "w", encoding="utf-8") as f:
            f.writelines(lines[-CAP // 2:])
    except Exception:
        pass


class DecisionLogger:
    def __init__(self, path=None):
        self.path = path or PATH

    def tick(self, *, deck_id, deck_version, hand, conf, cycle_ok, conservative,
             elixir, afford, legal_mask_n, q_top3, chosen_q, slot, spot, card,
             target_xy, origin, rule_note, vetoes, spell, reward_events,
             train_loss, td_err, replay_size, eps, model_profile,
             model_mb, affect=None):
        rec = {
            "t": time.strftime("%Y-%m-%d %H:%M:%S"),
            "deck": {"id": deck_id, "metrics_version": deck_version,
                     "uncertain": not deck_id.verified if hasattr(deck_id, "verified") else None},
            "hand": {"cards": hand, "conf": conf, "cycle_known": cycle_ok,
                     "conservative": conservative},
            "elixir": elixir,
            "afford": afford,
            "legal_actions_n": legal_mask_n,
            "q_top3": q_top3,
            "chosen_q": chosen_q,
            "action": {"slot": slot, "spot": spot, "card": card,
                       "target_xy": target_xy, "origin": origin},
            "rules": {"note": rule_note, "vetoes": vetoes},
            "spell": spell,
            "reward_events": reward_events,
            "training": {"loss": train_loss, "td_err": td_err,
                         "replay": replay_size, "eps": round(eps, 3),
                         "profile": model_profile,
                         "policy_mb": model_mb},
            "affect": (affect or {}).get("mood") if isinstance(affect, dict) else affect,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            _trim()
        except Exception:
            pass
        return rec


def read_last(n=50, path=None):
    path = path or PATH
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readlines()[-n:]
    except Exception:
        return []
