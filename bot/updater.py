"""NeuroML Updater — upgrades code, NEVER memory.

The one rule: data files (weights, replay, game memory, pain channels,
spiking checkpoints, opponent memory, logs, configs) are always preserved.
An update only replaces code + card-DB schema + docs, and takes a timestamped
backup of every overwritten code file so any update can be reverted.

Usage:
    python updater.py --from-dir <extracted_fresh_copy>   # from a downloaded zip
    python updater.py --dry-run                           # show what would change

This is also the safe path for federated-model ingestion: new code from other
devices lands through this same check so nobody's memory is ever wiped.
"""
import argparse
import json
import os
import shutil
import sys
import time

# Everything the user "is" — never touched by an update.
NEVER_TOUCH = {
    "ml_policy.pt", "ml_policy.pt.bak", "ml_replay.json", "ml_metrics.json",
    "training.log", "game_memory.json", "opponent_memory.json",
    "pain_channels.json", "spiking_brain.json", "arena_twin.pt",
    "arena_twin.pt.bak", "arena_twin.json", "hippocampus.jsonl",
    "hippocampus_rotate.log", "bot_config.json", "my_deck.json",
    "ai_coach.jsonl", "mentor.log", "groq_key.txt", "bot.log",
    "bot_stdout.log", "dash_stdout.log", "federated_outbox.json",
    "federated_imports.log", "federated_seed_ledger.json",
    "federated_device_id.txt",
}

CODE_FILES = ["bot_ml.py", "bot.py", "dashboard.py", "spiking_core.py",
              "referee.py", "arena_twin.py", "updater.py", "federated.py",
              "cards.json", "meta_decks.json", "bot_config.example.json"]

BACKUP_DIR = ".update_backups"


def plan_update(src_dir):
    """Return (code_changes, preserved) for the given fresh copy directory."""
    code_changes, preserved = [], []
    for f in CODE_FILES:
        src = os.path.join(src_dir, f)
        if not os.path.exists(src):
            continue
        dst = os.path.join(".", f)
        if os.path.exists(dst):
            with open(src, "rb") as a, open(dst, "rb") as b:
                same = a.read() == b.read()
            (preserved if same else code_changes).append(f)
        else:
            code_changes.append(f)
    return code_changes, preserved


def apply_update(src_dir, dry_run=False):
    if not os.path.isdir(src_dir):
        print(f"update source '{src_dir}' not found");  return 1
    code_changes, preserved = plan_update(src_dir)
    print(f"code to update   : {len(code_changes)} files")
    for f in code_changes:
        print(f"  ~ {f}")
    print("memory preserved : " + ", ".join(sorted(
        f for f in NEVER_TOUCH if os.path.exists(f))))
    if dry_run:
        print("DRY RUN — nothing written.")
        return 0
    if code_changes:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        bak = os.path.join(BACKUP_DIR, stamp)
        os.makedirs(bak, exist_ok=True)
        for f in code_changes:
            if os.path.exists(f):
                shutil.copy2(f, os.path.join(bak, f))
            shutil.copy2(os.path.join(src_dir, f), f)
        # rollback pointer for one-command revert
        with open(os.path.join(BACKUP_DIR, "latest.txt"), "w") as fh:
            fh.write(bak)
        print(f"updated {len(code_changes)} files; backup at {bak}")
    print("update complete — all memory intact.")
    return 0


def revert():
    if not os.path.exists(os.path.join(BACKUP_DIR, "latest.txt")):
        print("no previous update to revert.");  return 1
    bak = open(os.path.join(BACKUP_DIR, "latest.txt")).read().strip()
    for f in os.listdir(bak):
        shutil.copy2(os.path.join(bak, f), f)
    print(f"reverted to {bak} — memory untouched.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args(sys.argv[1:])
    if a.revert:
        sys.exit(revert())
    if not a.from_dir:
        print("usage: python updater.py --from-dir <copy> [--dry-run] [--revert]")
        sys.exit(1)
    sys.exit(apply_update(a.from_dir, a.dry_run))
