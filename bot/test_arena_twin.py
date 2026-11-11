"""Arena twin tests: Layer A transition sync, no future-state leakage,
Layer B deterministic mechanics, Layer C ensemble & readiness gates,
Layer D risk-aware planning, Layer E assimilation, Layer F atomic persistence
and memory hard-caps."""
import os
import sys
import random
from arena_twin import (ArenaTwin, TwinObs, ArenaMechanics, PersistenceBaseline,
                        MeanDeltaBaseline, MAX_TWIN_BUFFER, MIN_DATA_FOR_TRAINING,
                        TWIN_PLAN_ENABLED, TWIN_VERSION)
import arena_twin as at

PASS = 0
FAIL = 0
def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("PASS", name)
    else:
        FAIL += 1
        print("FAIL", name)

# ---------- Layer A transition sync ----------
twin = ArenaTwin()
seq1 = twin.record([0.1] * 38, 0, 0.5, [0.2] * 38)
check("valid transition gets a sequence id", seq1 is not None and seq1 == 1)
bad = twin.record([0.1] * 10, 0, 0.5, [0.2] * 38)      # wrong state dimension
check("corrupted transition REJECTED, not guessed", bad is None and twin.rejected_transitions >= 1)
bad2 = twin.record([0.1] * 38, 99, 0.5, [0.2] * 38)    # impossible action index
check("invalid action rejected", bad2 is None)
seq3 = twin.record([0.1] * 38, 0, 0.5, [0.2] * 38, stale=True, quality=0.5)
check("stale frame is stored STAMPED with quality metadata",
      twin.buffer[-1].stale and twin.buffer[-1].quality == 0.5)

# ---------- Layer B deterministic mechanics ----------
st_test = [0.0] * 38
st_test[0] = 0.2         # 2 elixir
st_test[3] = 0.5         # slot 0 costs 5 elixir
st_test[7] = 0.0         # slot 0 unaffordable
ok_act, reason_act = ArenaMechanics.validate_action(0, state=st_test)
check("Layer B rejects unaffordable slot action", not ok_act and "unaffordable" in reason_act)
st_test[0] = 0.6         # 6 elixir
st_test[7] = 1.0         # slot 0 affordable
ok_act2, _ = ArenaMechanics.validate_action(0, state=st_test)
check("Layer B accepts affordable valid action", ok_act2)
st_stepped = ArenaMechanics.step_deterministic(st_test, 2, dt_sec=2.8)
check("Layer B updates last slot and spot one-hot & clamps [0,1]",
      st_stepped[14] == 1.0 and st_stepped[20] == 1.0 and all(0.0 <= v <= 1.0 for v in st_stepped))
st_term = list(st_test)
st_term[25] = 0.0        # king destroyed -> absorbing state
check("Layer B treats terminal state as absorbing",
      ArenaMechanics.step_deterministic(st_term, 2, terminal=True) == st_term)

# ---------- Persistence & MeanDelta baselines ----------
s = [0.0] * 38
s[0] = 0.5                       # elixir at 5
s[23] = 0.7                      # ally left tower
pred = PersistenceBaseline.predict(s)
check("persistence keeps towers and regens elixir",
      abs(pred[23] - 0.7) < 1e-9 and abs(pred[0] - 0.75) < 1e-9)
mdb = MeanDeltaBaseline()
mdb.update([0.5] * 38, [0.6] * 38)
check("second baseline (MeanDeltaBaseline) updates from real transitions",
      mdb.count == 1 and mdb.predict([0.5] * 38)[0] > 0.5)

# ---------- Readiness gate: training only after enough data ----------
check("readiness gate documented", MIN_DATA_FOR_TRAINING >= 100)
twin2 = ArenaTwin()
for i in range(120):
    st = [0.0] * 38
    st[0] = (i % 10) / 10.0
    st[7] = 1.0; st[8] = 1.0; st[9] = 1.0; st[10] = 1.0
    st[23] = max(0.05, 1.0 - 0.002 * i)
    ns = list(st)
    ns[0] = min(1.0, st[0] + 0.25)
    ns[23] = max(0.0, st[23] - 0.002)
    twin2.record(st, i % 20, 0.05, ns)
before = twin2.metrics["train_steps"]
check("training refused before gate (honest untrained model)", before == 0 and not twin2.ready)
check("planning disabled without explicit env flag + validation", twin2.plan(st, [0, 1]) is None)

# ---------- Memory hard-cap (federated/device safety) ----------
check("twin replay is hard-capped", 0 < MAX_TWIN_BUFFER <= 40000)
check("checkpoint size guard exists", at.MAX_CHECKPOINT_BYTES == 220 * 1024 * 1024)

# ---------- Layer C & E: ensemble training + validation + assimilation ----------
for i in range(600):
    st = [0.0] * 38
    st[0] = 0.55
    st[7] = 1.0; st[8] = 1.0; st[9] = 1.0; st[10] = 1.0
    st[23] = max(0.05, 1.0 - 0.001 * i)
    st[26] = max(0.05, 1.0 - 0.001 * i)
    ns = list(st)
    ns[23] = max(0.0, st[23] - 0.001)
    ns[0] = min(1.0, st[0] - 0.1)        # baseline predicts +0.25; truth -0.1
    twin2.record(st, i % 20, 0.05, ns)
m = twin2.validate()
check("validation ran on held-out split", m is not None and m["n_scored"] > 0)
check("twin elixir & tower MAE computed against BOTH baselines",
      m["twin_mae_elixir"] is not None and m["baseline_mae_elixir"] is not None
      and m["mean_baseline_mae_towers"] is not None)
check("beats_baseline is a strict boolean verdict", m["beats_baseline"] in (True, False))
check("Layer E outcome assimilation populated after ready prediction",
      twin2.metrics.get("last_assimilated") is not None)
p1 = twin2.predict_one_step(st, 0)
check("predict_one_step returns state_pred, reward_pred, done_prob, uncertainty, feature_uncertainty",
      p1 is not None and len(p1["state_pred"]) == 38 and len(p1["feature_uncertainty"]) == 38)

# ---------- Layer F: save/load roundtrip + version guard ----------
ok = twin2.save()
check("atomic save returns bool", ok is True)
loaded = ArenaTwin()
check("restored twin inherits seq + ready + metrics", loaded.load() and loaded.ready
      and loaded.seq == twin2.seq and loaded.metrics["train_steps"] > 0)
for fn in (at.TWIN_FILE, at.TWIN_META_FILE, at.TWIN_FILE + ".bak"):
    if os.path.exists(fn):
        os.remove(fn)

check("plan gate is intersection (flag AND beats_baseline)",
      loaded.plan(st, [0, 1]) is None or (getattr(at, "TWIN_PLAN_ENABLED", False)
                                          and loaded.metrics.get("beats_baseline")))

print(f"\nARENA TWIN TESTS: {PASS} passed, {FAIL} failed")
if FAIL:
    sys.exit(1)
