"""Sync the canonical bot/ implementation into the public_gui/ distribution.

The regression test (test_overhaul.py, scenario 12) requires these to be
byte-identical; run this after any edit to a mirrored file:

    python bot/sync_mirrors.py
"""
import filecmp
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GUI = os.path.join(ROOT, "public_gui")

MIRRORED = [
    "bot_ml.py",
    "cards_v2.json",
    "card_index.py",
    "deck_state.py",
    "hand_tracker.py",
    "spell_ledger.py",
    "affect.py",
    "brain_xl.py",
    "decision_log.py",
    "referee.py",
    "dashboard.py",
    "eval_scenarios.py",
    "test_opponent_read.py",
    "test_counter_adaptation.py",
]


def main():
    bad = []
    for f in MIRRORED:
        src = os.path.join(HERE, f)
        dst = os.path.join(GUI, f)
        if not os.path.exists(src):
            bad.append(f + " missing in bot/")
            continue
        if os.path.exists(dst) and filecmp.cmp(src, dst, shallow=False):
            continue
        shutil.copy2(src, dst)
        print("synced", f)
    missing_after = []
    for f in MIRRORED:
        src = os.path.join(HERE, f)
        dst = os.path.join(GUI, f)
        if not os.path.exists(dst) or not filecmp.cmp(src, dst, shallow=False):
            missing_after.append(f)
    if bad or missing_after:
        print("SYNC FAILED:", bad, missing_after)
        return 1
    print("all mirrored files identical to bot/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
