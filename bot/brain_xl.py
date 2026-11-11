"""NeuroML brain_xl — measurable model profiles (PROMPT 4 P4).

Old claim: advanced ≈ 34-35M params (~140 MB). New honest plan:
  compact   768  x 6   ≈ 7-8M     small fallback
  standard  1024 x 6   ≈ 13.7M    small fallback
  extended  1408 x 8   ≈ 34M      (desktop default v27–v30)
  advanced  2560 x 20  ≈ ~270M   ≈ ~1.0+ GiB fp32 policy weights — REAL
                                    architecture, real save path, honestly
                                    labeled TRAIN-LIMITED (see below).
                                    v31 DESKTOP DEFAULT — superseded by
                                    'apex' in v32 (owner: "more").
  apex      3584 x 24  ≈ ~630M  ≈ ~2.4 GiB fp32 policy weights
                                    v32: DESKTOP DEFAULT (owner asked for
                                    MORE neurons; brain runs on the
                                    computer only). Same honest rules.

Honesty rules enforced here:
  * parameter counts come from instantiated modules (bot_ml.param_report),
    never from comments; checkpoint BYTE counts are measured from the file.
  * target network / optimizer state / replay are reported SEPARATELY —
    the 1 GiB claim may only cover the POLICY weights file (no
    double-counting of the target net, no replay padding).
  * v30: the phone CAN load extended AND advanced — the old hard RAM guard
    was removed at the owner's explicit request ("270M params for the phone,
    modern phones have the storage"). In its place the phone build runs a
    NATIVE PRE-FLIGHT before importing the brain: it proves the OS will
    actually grant the memory, and steps down LOUDLY (advanced -> extended
    -> standard) if it won't — never silently, never lying in the metrics.
    (v31: the mobile app is no longer shipped — computer-only operation.)
  * the Advanced profile is still label-marked "train-limited until enough
    verified match data exists" (a few tens of matches cannot meaningfully
    train ~270M params) — loading it is fine, only the expectations are
    managed; measured param/loss telemetry stays exactly as honest as before.
"""
import os

PROFILE_TABLE = {
    "compact":  (768, 6),
    "standard": (1024, 6),
    "extended": (1408, 8),   # desktop default v27–v30
    "advanced": (2560, 20),  # v31 desktop default — ~1.0+ GiB fp32 policy
    "apex":    (3584, 24),   # v32 DESKTOP DEFAULT — ~630M params, ~2.4 GiB fp32
}

CHECKPOINTS = {
    "compact":  "ml_policy_compact.pt",
    "standard": "ml_policy.pt",
    "extended": "ml_policy_extended.pt",
    "advanced": "ml_policy_advanced.pt",
}


def choose_profile(cfg=None, env=None, phone=False, default=None):
    """Resolve the requested profile. Since v32 the DESKTOP default is
    'apex' — the ~630M ≈ 2.4 GiB-fp32 policy brain the owner escalated to
    ("only 270M... more"; they run the brain ONLY on the computer now).
    Explicit requests still win; unknown names fall back loudly. The
    phone-side pre-flight in android_main is unchanged (though the mobile
    app is no longer shipped).
    Note for tests/federated: pass default='standard' explicitly when a
    specific profile must be pinned regardless of device."""
    if default is None:
        default = "standard" if phone else "apex"
    env = env if env is not None else os.environ
    requested = None
    try:
        requested = (cfg or {}).get("model_profile")
    except Exception:
        requested = None
    requested = requested or env.get("NEUROML_PROFILE") or env.get("QNET_PROFILE")
    requested = (requested or "").strip().lower() or None
    if requested in PROFILE_TABLE:
        return requested, "requested profile selected"
    if requested:
        return default, f"unknown profile {requested!r}; fell back to {default}"
    return default, "default profile"


def guard_load_ok(profile, device_phone=False):
    """v30: the hard phone refusal is gone (owner request). Kept as an honest
    capability note — a plain bool check reportable to the dashboard — not a
    gate. The REAL gate for the phone is the pre-flight in android_main."""
    return True, f"{profile} allowed on {'phone' if device_phone else 'desktop'} (v30: guard removed)"


def measure_policy(net):
    """Measured policy-weight report from the instantiated module itself."""
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {
        "profile": getattr(net, "profile", "?"),
        "policy_params": params,
        "policy_bytes_fp32": params * 4,
        "policy_mb_fp32": round(params * 4 / 1024 / 1024, 1),
        "meets_1_gib_claim": params * 4 >= 1024 * 1024 * 1024,
    }


def measure_checkpoint(path):
    """Measured saved-file bytes (policy weights only) — the ONLY honest way
    to state the 1 GiB claim."""
    try:
        n = os.path.getsize(path)
    except OSError:
        n = 0
    return {"path": path, "bytes": n, "mb": round(n / 1024 / 1024, 1)}


SEPARATE_COSTS_NOTE = (
    "Reported separately and NEVER counted inside the 1 GiB policy-weight "
    "claim: target-network copy (≈1 extra policy-sized file), AdamW optimizer "
    "state (2 extra parameter-sized buffers), replay/episodic memory, twin "
    "world-model ensemble, peak training/inference RAM.")
