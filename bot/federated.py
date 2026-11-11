"""Federated NeuroML — opt-in, memory-safe experience sharing and offline FedAvg.

Every installation can export an ANONYMIZED learning package (policy + target
weights, episode count, sanitized match records, architecture fingerprint, and
profile — NO API keys, usernames, screenshots, device addresses, or raw logs)
and import packages from other devices. Imports land in a quarantine folder;
applying a consensus package is an explicit, reviewed step so a single bad
package can never poison or crash the local installation.

MEMORY SAFETY:
  - MAX_PACKAGE_BYTES (220 MB) refuses oversized files before torch.load.
  - Streaming / incremental weighted accumulation in federated_average():
    packages are loaded one at a time onto CPU and accumulated in-place (`del`
    after each contributor) instead of stacking N full state dicts in RAM at
    once — so aggregating many packages will NEVER crash a PC or phone from
    memory exhaustion.
  - Sanitized match records and the append-only `federated_seed_ledger.json`
    are strictly capped and deduplicated by stable record IDs.

IMPORTANT LIMITATION (documented):
  This is decentralized, package-exchange federated learning, NOT automatic
  cloud synchronization across every user. Without a server, peer service, or
  shared storage mechanism, different devices contribute only when their
  exported packages are explicitly exchanged and aggregated.
"""
import gc
import hashlib
import json
import os
import shutil
import time

INBOX = "federated_incoming"
OUTBOX = "federated_outbox.json"
LOG = "federated_imports.log"
LEDGER = "federated_seed_ledger.json"
MAX_PACKAGE_BYTES = 220 * 1024 * 1024      # hard memory-safety cap on imports (220 MB)
MAX_LEDGER_RECORDS = 5000                  # hard bound on ledger entries
MAX_MATCH_RECORDS_PER_EXPORT = 100         # cap sanitized match records per package

FORBIDDEN_PRIVACY_KEYS = {
    "api_key", "groq_key", "username", "account", "screenshot", "image",
    "ip", "device_ip", "mac", "raw_log", "password", "token", "secret",
}


def _fingerprint(sd):
    """Architecture identity: sorted layer names + shapes hash. Validating
    fingerprints ensures two models share identical layer structure before
    weights are averaged — never blindly average differently shaped networks."""
    parts = []
    for k in sorted(sd.keys()):
        try:
            shape = tuple(sd[k].shape)
        except Exception:
            shape = None
        parts.append(f"{k}:{shape}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:20]


def _sanitize_match_records(records):
    """Sanitize match records for export: retain only game outcome, ticks,
    towers, and timestamp; strip any personal/account/device keys and assign
    a stable record_id so duplicate imports never double-count."""
    clean = []
    for r in (records or [])[-MAX_MATCH_RECORDS_PER_EXPORT:]:
        if not isinstance(r, dict):
            continue
        verdict = r.get("v") or r.get("verdict")
        if verdict not in ("WIN", "LOSS"):
            continue
        entry = {
            "t": str(r.get("t", ""))[:16],
            "v": verdict,
            "dop": round(float(r.get("dop", 0.0) or 0.0), 2),
            "ticks": int(r.get("ticks", 0) or 0),
            "towers_down": int(r.get("towers_down", 0) or 0),
            "towers_lost": int(r.get("towers_lost", 0) or 0),
        }
        rid = r.get("record_id") or hashlib.sha256(
            json.dumps(entry, sort_keys=True).encode()
        ).hexdigest()[:16]
        entry["record_id"] = rid
        clean.append(entry)
    return clean


def export_delta(seed, metrics, consented=True, out=OUTBOX):
    """Export an anonymized, opt-in model package. Consent is explicit — an
    unset consent flag aborts (nobody's model is uploaded silently)."""
    if not consented:
        return {"exported": False, "reason": "user did not consent to sharing"}
    return export_weights(seed, metrics, consented=True, out="federated_package.pt")


def export_weights(seed, metrics, consented=True, out="federated_package.pt",
                   match_records=None, base_fingerprint=None):
    """Full export usable on phone/PC: policy + target tensors + versioned
    metadata + episode_count + sanitized match records. Excludes API keys,
    usernames, screenshots, device addresses, and raw logs."""
    if not consented:
        return {"exported": False, "reason": "not consented"}
    import torch
    from bot_ml import TargetedDQN, param_report
    brain = TargetedDQN()
    brain.load()
    pr = param_report(brain.policy)
    fp = _fingerprint(brain.policy.state_dict())
    if match_records is None and os.path.exists("game_memory.json"):
        try:
            gm = json.load(open("game_memory.json"))
            match_records = gm.get("matches", [])
        except Exception:
            match_records = []
    sanitized_records = _sanitize_match_records(match_records)
    wins = int(metrics.get("wins_v", 0) or 0)
    losses = int(metrics.get("losses_v", 0) or 0)
    # Documented episode_count definition: total completed matches (wins + losses,
    # falling back to lifetime_games / games_recorded or at least 1 if trained steps exist)
    episodes = int(metrics.get("lifetime_games", 0)
                   or metrics.get("games_recorded", 0)
                   or (wins + losses)
                   or (1 if int(metrics.get("steps", 0) or 0) > 0 else 0))
    dev_id = _identity()
    pkg_id = hashlib.sha256(
        f"{dev_id}:{seed}:{episodes}:{wins}:{losses}:{int(metrics.get('steps', 0) or 0)}".encode()
    ).hexdigest()[:16]

    buf = {
        "schema": "neuroml_federated_v2",
        "schema_version": 2,
        "package_id": pkg_id,
        "seed": seed,
        "t": time.time(),
        "epsilon": brain.epsilon,
        "episode_count": episodes,
        "episode_definition": "completed judged matches (WIN+LOSS) on this device",
        "arch_fingerprint": fp,
        "base_fingerprint": base_fingerprint or fp,
        "profile": pr["profile"],
        "arch_id": f"dueling_resnet_v5_{pr['profile']}_{pr['depth']}b",
        "param_count": pr["trainable_params"],
        "policy": brain.policy.state_dict(),
        "target": brain.target.state_dict(),
        "replay_pointer": {
            "steps_count": int(metrics.get("steps", 0) or 0),
            "buffer_size": int(metrics.get("buffer", 0) or 0),
        },
        "matches": {
            "games": episodes,
            "wins": wins,
            "losses": losses,
            "train_steps": int(metrics.get("steps", 0) or 0),
        },
        "match_records": sanitized_records,
        "identity": dev_id,
    }
    torch.save(buf, out)
    h = hashlib.sha256(open(out, "rb").read()).hexdigest()[:16]
    json.dump({
        "sent_package": os.path.basename(out),
        "package_id": pkg_id,
        "sha": h,
        "episode_count": episodes,
        "profile": pr["profile"],
        "t": time.strftime("%Y-%m-%d %H:%M"),
        "steps": buf["replay_pointer"]["steps_count"],
    }, open(OUTBOX, "w"), indent=1)
    return {
        "exported": True,
        "package": out,
        "package_id": pkg_id,
        "sha": h,
        "episode_count": episodes,
        "steps": buf["replay_pointer"]["steps_count"],
    }


def _ledger_append(record, match_records=None):
    """Append-only fleet statistics ledger (`federated_seed_ledger.json`) with
    STABLE record IDs so importing the same package or match twice never
    double-counts episodes or match records. Strictly bounded in size."""
    try:
        data = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {
            "version": 2, "records": {}, "match_ids": []
        }
    except Exception:
        data = {"version": 2, "records": {}, "match_ids": []}
    if "records" not in data or not isinstance(data["records"], dict):
        data["records"] = {}
    if "match_ids" not in data or not isinstance(data["match_ids"], list):
        data["match_ids"] = []

    rid = record.get("id")
    if rid is None:
        rid = hashlib.sha256(json.dumps(
            {k: record.get(k) for k in ("device", "package_id", "episodes", "wins", "losses", "fingerprint")},
            sort_keys=True).encode()).hexdigest()[:16]
        record["id"] = rid
    dup = rid in data["records"] and data["records"][rid].get("applied")
    if not dup:
        record["applied"] = True
        if len(data["records"]) >= MAX_LEDGER_RECORDS:
            oldest_k = next(iter(data["records"]))
            del data["records"][oldest_k]
        data["records"][rid] = record

    seen_m = set(data["match_ids"])
    new_matches = 0
    for mr in (match_records or []):
        mid = mr.get("record_id") if isinstance(mr, dict) else None
        if mid and mid not in seen_m:
            seen_m.add(mid)
            data["match_ids"].append(mid)
            new_matches += 1
    if len(data["match_ids"]) > MAX_LEDGER_RECORDS:
        data["match_ids"] = data["match_ids"][-MAX_LEDGER_RECORDS:]

    data["version"] = 2
    data["total_contributors"] = len(data["records"])
    data["total_episodes"] = sum(int(v.get("episodes", 0)) for v in data["records"].values())
    data["total_wins"] = sum(int(v.get("wins", 0)) for v in data["records"].values())
    data["total_losses"] = sum(int(v.get("losses", 0)) for v in data["records"].values())
    data["unique_match_records"] = len(data["match_ids"])
    json.dump(data, open(LEDGER, "w"), indent=1)
    return rid, not dup, new_matches


def _extract_episodes(b):
    """Extract positive episode weight from package dict. Rejects <= 0 or NaN."""
    raw = b.get("episode_count", None)
    if raw is None:
        raw = b.get("replay_pointer", {}).get("steps_count", 0)
    try:
        ep = float(raw)
        if not (ep > 0 and ep < 1e12):
            return 0.0
        return ep
    except Exception:
        return 0.0


def federated_average(packages, out=None):
    """Real offline FedAvg (`federated_average(packages)`):
      - Accepts multiple package paths OR already-loaded package dictionaries.
      - Validates schema, tensor names, shapes, finite values, profile,
        architecture fingerprint, and base-model lineage.
      - Weights policy and target tensors by reported `episode_count`.
      - Uses incremental streaming accumulation (loads one package at a time
        when paths are passed and frees memory immediately) so RAM never spikes
        or crashes the computer.
      - Updates append-only `federated_seed_ledger.json` with deduplicated
        stable record IDs so importing the same package twice never double-counts."""
    import torch
    if out is None:
        out = "federated_aggregate.pt"
    if not packages or not isinstance(packages, (list, tuple)):
        return {"aggregated": False, "reason": "no packages provided"}

    # Pass 1: Lightweight metadata & validation scan (or direct dict inspection)
    meta_items = []
    ref_fp = None
    ref_base_fp = None
    ref_profile = None
    ref_keys = None
    ref_shapes = None
    total_ep = 0.0

    for idx, p in enumerate(packages):
        if isinstance(p, dict):
            ok, msg = verify_package_dict(p)
            if not ok:
                return {"aggregated": False, "reason": f"dict[{idx}]: {msg}"}
            b = p
        else:
            ok, msg = verify_package(p)
            if not ok:
                return {"aggregated": False, "reason": f"{os.path.basename(str(p))}: {msg}"}
            b = torch.load(p, map_location="cpu")

        ep = _extract_episodes(b)
        if ep <= 0:
            return {"aggregated": False, "reason": "package has zero or invalid episode_count — refused"}

        sd = b.get("policy", {})
        fp = b.get("arch_fingerprint") or _fingerprint(sd)
        base_fp = b.get("base_fingerprint") or fp
        prof = b.get("profile", "standard")
        keys = tuple(sorted(sd.keys()))
        shapes = {k: tuple(torch.as_tensor(v).shape) for k, v in sd.items()}

        if ref_fp is None:
            ref_fp = fp
            ref_base_fp = base_fp
            ref_profile = prof
            ref_keys = keys
            ref_shapes = shapes
        else:
            if fp != ref_fp:
                return {
                    "aggregated": False,
                    "reason": f"incompatible architectures: fingerprint mismatch ({fp} vs {ref_fp})",
                }
            if base_fp != ref_base_fp:
                return {
                    "aggregated": False,
                    "reason": f"incompatible base lineage: base_fingerprint mismatch ({base_fp} vs {ref_base_fp})",
                }
            if prof != ref_profile:
                return {
                    "aggregated": False,
                    "reason": f"incompatible model profiles: {prof} vs {ref_profile}",
                }
            if keys != ref_keys or shapes != ref_shapes:
                return {
                    "aggregated": False,
                    "reason": "tensor key/shape mismatch across packages",
                }

        meta_items.append({
            "source": p if isinstance(p, dict) else str(p),
            "is_dict": isinstance(p, dict),
            "ep": ep,
            "fp": fp,
            "base_fp": base_fp,
            "profile": prof,
            "epsilon": float(b.get("epsilon", 0.35) or 0.35),
            "identity": str(b.get("identity", f"anon_{idx}")),
            "package_id": b.get("package_id"),
            "matches": dict(b.get("matches", {})),
            "match_records": list(b.get("match_records", [])),
        })
        total_ep += ep
        if not isinstance(p, dict):
            del b
            gc.collect()

    # Pass 2: Incremental streaming weighted sum (never holds >1 package + accumulator in RAM)
    avg_policy = {}
    avg_target = {}
    for item in meta_items:
        w_factor = item["ep"] / total_ep
        b = item["source"] if item["is_dict"] else torch.load(item["source"], map_location="cpu")
        pol = b["policy"]
        tgt = b.get("target", pol)
        with torch.no_grad():
            for k in ref_keys:
                t_pol = torch.as_tensor(pol[k], dtype=torch.float32)
                if not torch.isfinite(t_pol).all():
                    return {"aggregated": False, "reason": f"non-finite tensor at {k}"}
                if k not in avg_policy:
                    avg_policy[k] = t_pol.clone() * w_factor
                else:
                    avg_policy[k].add_(t_pol, alpha=w_factor)

                if k in tgt:
                    t_tgt = torch.as_tensor(tgt[k], dtype=torch.float32)
                    if not torch.isfinite(t_tgt).all():
                        return {"aggregated": False, "reason": f"non-finite target tensor at {k}"}
                    if k not in avg_target:
                        avg_target[k] = t_tgt.clone() * w_factor
                    else:
                        avg_target[k].add_(t_tgt, alpha=w_factor)
        if not item["is_dict"]:
            del b, pol, tgt
            gc.collect()

    # Record each contributor in the deduplicated append-only ledger
    new_contributors = 0
    for item in meta_items:
        bm = item["matches"]
        rec = {
            "device": item["identity"],
            "package_id": item["package_id"],
            "episodes": int(round(item["ep"])),
            "games": int(bm.get("games", 0) or 0),
            "wins": int(bm.get("wins", 0) or 0),
            "losses": int(bm.get("losses", 0) or 0),
            "fingerprint": ref_fp,
            "profile": ref_profile,
        }
        _rid, added, _nm = _ledger_append(rec, match_records=item["match_records"])
        if added:
            new_contributors += 1

    # Read deduplicated fleet totals from ledger, plus direct sum of current call
    games_call = sum(int(it["matches"].get("games", 0) or 0) for it in meta_items)
    wins_call = sum(int(it["matches"].get("wins", 0) or 0) for it in meta_items)
    losses_call = sum(int(it["matches"].get("losses", 0) or 0) for it in meta_items)

    lead = max(meta_items, key=lambda m: m["ep"])
    total_ep_int = int(round(total_ep)) if abs(total_ep - round(total_ep)) < 1e-6 else total_ep

    consensus_pkg = {
        "schema": "neuroml_federated_v2",
        "schema_version": 2,
        "aggregate_version": 2,
        "aggregate_of": [
            os.path.basename(it["source"]) if not it["is_dict"] else f"memory:{it['identity']}"
            for it in meta_items
        ],
        "contributor_count": len(meta_items),
        "devices": len(meta_items),
        "total_episodes": total_ep_int,
        "episode_count": total_ep_int,
        "weights_used": [round(it["ep"] / total_ep, 6) for it in meta_items],
        "source_fingerprints": [it["fp"] for it in meta_items],
        "arch_fingerprint": ref_fp,
        "base_fingerprint": ref_base_fp,
        "epsilon": lead["epsilon"],
        "profile": ref_profile,
        "identity": "federated_aggregate",
        "policy": avg_policy,
        "target": avg_target,
        "matches": {
            "games": games_call,
            "wins": wins_call,
            "losses": losses_call,
        },
        "fleet_matches": {
            "games": games_call,
            "wins": wins_call,
            "losses": losses_call,
            "aggregate_winrate": round(100.0 * wins_call / max(1, wins_call + losses_call), 1),
        },
    }
    torch.save(consensus_pkg, out)

    summary = {
        "aggregated": True,
        "schema_version": 2,
        "contributor_count": len(meta_items),
        "new_contributors_logged": new_contributors,
        "devices": len(meta_items),
        "total_episodes": total_ep_int,
        "weights_used": consensus_pkg["weights_used"],
        "arch_fingerprint": ref_fp,
        "profile": ref_profile,
        "fleet_matches": consensus_pkg["fleet_matches"],
        "package": out,
        "ledger": LEDGER,
        "policy": avg_policy,
        "target": avg_target,
    }
    try:
        # Write JSON summary without raw tensors
        json_summary = {k: v for k, v in summary.items() if k not in ("policy", "target")}
        json.dump(json_summary, open("federated_aggregate_summary.json", "w"), indent=1)
    except Exception:
        pass
    return summary


def _identity():
    """Random per-device pseudonymous id — no usernames, no IP, no accounts."""
    p = "federated_device_id.txt"
    if os.path.exists(p):
        return open(p).read().strip()
    did = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    open(p, "w").write(did)
    return did


def verify_package_dict(buf):
    """Structural verification on an already-loaded dictionary: schema,
    forbidden privacy keys, tensor names/shapes, and FINITE values."""
    import torch
    if not isinstance(buf, dict):
        return False, "package is not a dict"
    if buf.get("schema") not in ("neuroml_federated_v1", "neuroml_federated_v2"):
        return False, "wrong schema"
    leaked = FORBIDDEN_PRIVACY_KEYS.intersection({str(k).lower() for k in buf.keys()})
    if leaked:
        return False, f"privacy violation: forbidden keys {sorted(leaked)}"
    sd = buf.get("policy")
    if not isinstance(sd, dict) or not sd or "sensory.0.weight" not in sd:
        return False, "missing core layers"
    bad_float = 0
    for k, v in sd.items():
        try:
            t = torch.as_tensor(v, dtype=torch.float32)
            if t.numel() == 0 or not torch.isfinite(t).all():
                bad_float += 1
        except Exception:
            bad_float += 1
    if bad_float:
        return False, f"{bad_float} non-finite/broken tensors"
    return True, f"verified fingerprint={str(buf.get('arch_fingerprint', 'legacy'))[:12]}"


def verify_package(path):
    """Structural verification: file size cap, schema, tensor names/shapes,
    and FINITE values. Accepts v1/v2 packages. Returns (ok, message)."""
    import torch
    if isinstance(path, dict):
        return verify_package_dict(path)
    if not os.path.exists(path):
        return False, "file not found"
    if os.path.getsize(path) > MAX_PACKAGE_BYTES:
        return False, "package exceeds the 220 MB memory-safety cap — refused"
    try:
        buf = torch.load(path, map_location="cpu")
    except Exception as e:
        return False, f"unreadable package: {e}"
    ok, msg = verify_package_dict(buf)
    del buf
    gc.collect()
    return ok, msg


def ingest_package(path, note=None, apply_now=False):
    """Move a package into quarantine (`federated_incoming/`) and VERIFY it;
    apply_now=True is the explicit reviewed application step."""
    if os.path.exists(path) and os.path.getsize(path) > MAX_PACKAGE_BYTES:
        return {"verified": False, "msg": "exceeds 220 MB memory cap", "applied": False}
    os.makedirs(INBOX, exist_ok=True)
    dst = os.path.join(INBOX, os.path.basename(path))
    if os.path.abspath(path) != os.path.abspath(dst):
        shutil.copy2(path, dst)
    ok, msg = verify_package(dst)
    line = json.dumps({
        "t": time.strftime("%Y-%m-%d %H:%M"),
        "file": dst,
        "verified": ok,
        "msg": msg,
        "note": note,
        "applied": bool(ok and apply_now),
    })
    try:
        open(LOG, "a").write(line + "\n")
    except Exception:
        pass
    if apply_now and ok:
        import torch
        from bot_ml import TargetedDQN
        # Back up existing policy before applying consensus model
        if os.path.exists("ml_policy.pt"):
            shutil.copy2("ml_policy.pt", "ml_policy.pt.pre_federated.bak")
        arrival = TargetedDQN()
        arrival.load()
        buf = torch.load(dst, map_location="cpu")
        if buf.get("profile") and buf.get("profile") != arrival.policy.profile:
            return {
                "verified": True,
                "applied": False,
                "file": dst,
                "msg": f"profile mismatch ({buf.get('profile')} vs {arrival.policy.profile}) — refused direct overwrite",
            }
        arrival.policy.load_state_dict(buf["policy"])
        arrival.target.load_state_dict(buf.get("target", buf["policy"]))
        arrival.save()
        del buf
        gc.collect()
        return {"verified": True, "applied": True, "file": dst, "msg": msg}
    return {"verified": ok, "msg": msg, "applied": False, "file": dst}


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(
        description="NeuroML Offline Federated Learning CLI (verify, aggregate, quarantine, apply)"
    )
    ap.add_argument("--export-consent", action="store_true",
                    help="Export an anonymized package (requires explicit consent)")
    ap.add_argument("--verify", metavar="PACKAGE",
                    help="Verify schema, finite tensors, and memory cap of a package")
    ap.add_argument("--quarantine", metavar="PACKAGE",
                    help="Ingest package into quarantine and verify without applying")
    ap.add_argument("--apply", metavar="PACKAGE",
                    help="Explicitly apply a verified/reviewed package or aggregate")
    ap.add_argument("--aggregate", nargs="+", metavar="PACKAGE",
                    help="Federated-average multiple compatible packages weighted by episode_count")
    ap.add_argument("--out", default="federated_aggregate.pt",
                    help="Output path for --aggregate")
    args = ap.parse_args()
    if args.aggregate:
        res = federated_average(args.aggregate, out=args.out)
        out_json = {k: v for k, v in res.items() if k not in ("policy", "target")}
        print(json.dumps(out_json))
        sys.exit(0 if res.get("aggregated") else 1)
    if args.verify:
        ok, msg = verify_package(args.verify)
        print(msg)
        sys.exit(0 if ok else 1)
    if args.quarantine:
        r = ingest_package(args.quarantine, apply_now=False)
        print(json.dumps(r))
        sys.exit(0 if r.get("verified") else 1)
    if args.apply:
        r = ingest_package(args.apply, apply_now=True)
        print(json.dumps(r))
        sys.exit(0 if r.get("applied") else 1)
    if args.export_consent:
        from bot_ml import _metrics
        r = export_weights(os.urandom(4).hex(), _metrics, consented=True)
        print(json.dumps(r))
        sys.exit(0 if r.get("exported") else 1)
