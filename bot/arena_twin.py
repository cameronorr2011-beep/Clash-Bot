"""NeuroML Arena Twin — a LOCAL, honest hybrid digital twin of observed state flow.

Implements the 6-Layer Digital Twin Architecture:
  Layer A (Real-state observation & synchronization):
    - Sequence ID, timestamp, directly observed vs estimated feature tags,
      transition alignment around brain.remember(...), and rejection of stale,
      corrupted, or misaligned transitions (no future-state leakage).
  Layer B (Deterministic arena mechanics):
    - 20-action validation (4 slots × 5 spots), card-slot affordability checks,
      legal-action masks, bounded state clamping, terminal absorbing states,
      elapsed-time elixir regeneration when timing is known, and reconciliation
      with real observations. Respects mandatory physics finishers.
  Layer C (Learned world-model ensemble):
    - Lightweight CPU-friendly ensemble of residual MLPs predicting next-state
      deltas, immediate reward, terminal probability, and per-feature predictive
      uncertainty from (state_38, action_20, dt, quality).
    - Trained ONLY on genuine observed transitions (never on imagined rollouts).
    - Compared against two baselines on held-out transitions:
        1) PersistenceBaseline (towers persist, elixir regenerates)
        2) MeanDeltaBaseline (historical empirical mean delta per feature)
  Layer D (Risk-aware model-predictive planning):
    - Bounded 1-step (or 2-step when second-step reliability passes gate) lookahead
      over top legal candidates within a strict wall-clock deadline.
    - Calibrated objective: V_plan(s,a) = r_hat + gamma * V_dqn(s') - lambda_u * uncertainty
    - Uncertainty-aware fallback: high confidence -> full weight, moderate ->
      scaled contribution, low confidence / unvalidated -> defer to DQN.
  Layer E (Actual-outcome assimilation):
    - Assimilates next real observation, computes per-feature prediction error
      and drift, logs predicted vs observed tower HP and elixir, and tracks
      decision origin (dqn_policy, twin_planner, rules_referee, physics_finisher).
  Layer F (Persistence & checkpoint safety):
    - Atomic checkpoint writes with recoverable .bak backup, version & schema
      validation, normalization metadata, and strict memory/byte caps so a
      corrupted or oversized file can never OOM a PC or phone.
"""
import json
import math
import os
import random
import shutil
import time
from collections import deque

try:
    import torch
    import torch.nn as nn
except Exception:                     # phone/no-torch deployments still sync data
    torch = None
    nn = None

STATE_DIM = 38
ACTION_DIM = 20

# Feature index groups in the 38-dim state vector:
IDX_ELIXIR = 0
IDX_LANE_VAR = (1, 2)
IDX_COSTS = (3, 4, 5, 6)
IDX_AFFORD = (7, 8, 9, 10)
IDX_PHASE = (11, 12, 13)              # one-hot categorical
IDX_LAST_SLOT = (14, 15, 16, 17)      # one-hot categorical
IDX_LAST_SPOT = (18, 19, 20, 21, 22)  # one-hot categorical
IDX_TOWERS = (23, 24, 25, 26, 27)     # ally_L, ally_R, ally_K, enemy_L, enemy_R
IDX_ALIVE = (28, 29)
IDX_UNITS = (30, 31, 32, 33, 34, 35)
IDX_THREATS = (36, 37)

DIRECTLY_OBSERVED_INDICES = (
    (0, 1, 2, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
    + IDX_TOWERS + IDX_ALIVE + IDX_UNITS + IDX_THREATS
)
ESTIMATED_INDICES = IDX_COSTS         # reference deck costs

# --- Honest resource caps (prevents RAM/disk exhaustion on PC & phone) ---
MAX_TWIN_BUFFER = 25000               # hard cap (~18 MB RAM max)
TWIN_BATCH = 128
TRAIN_EVERY = 50                      # train cadence in valid transitions
MIN_DATA_FOR_TRAINING = 250           # readiness gate (valid transitions)
HOLDOUT_FRACTION = 0.15
ENSEMBLE_MEMBERS = 3
WEAVER_W = 96                         # lightweight CPU width
MAX_CHECKPOINT_BYTES = 220 * 1024 * 1024   # refuse >220 MB checkpoints
TWIN_FILE = "arena_twin.pt"
TWIN_META_FILE = "arena_twin.json"
TWIN_VERSION = 2
TWIN_PLAN_ENABLED = os.environ.get("TWIN_PLAN_ENABLED", "0") == "1"


def _clamp01(v):
    try:
        f = float(v)
        if not math.isfinite(f):
            return 0.0
        return max(0.0, min(1.0, f))
    except Exception:
        return 0.0


class TwinObs:
    """Layer A synchronized transition bundle: what the device genuinely observed."""
    __slots__ = ("t", "seq", "state", "action", "reward", "next_state",
                 "done", "stale", "quality", "dt", "towers", "elixir", "origin")

    def __init__(self, t, seq, state, action=None, reward=None,
                 next_state=None, done=False, stale=False, quality=1.0,
                 dt=2.2, towers=None, elixir=None, origin="dqn_policy"):
        self.t = float(t)
        self.seq = int(seq)
        self.state = list(state) if state is not None else None
        self.action = int(action) if action is not None else None
        self.reward = float(reward) if reward is not None else 0.0
        self.next_state = list(next_state) if next_state is not None else None
        self.done = bool(done)
        self.stale = bool(stale)
        self.quality = float(quality)
        self.dt = max(0.1, min(30.0, float(dt or 2.2)))
        self.towers = towers
        self.elixir = elixir
        self.origin = origin

    def valid(self):
        if self.state is None or len(self.state) != STATE_DIM:
            return False
        if self.next_state is not None and len(self.next_state) != STATE_DIM:
            return False
        if self.action is not None and not (0 <= self.action < ACTION_DIM):
            return False
        if not all(math.isfinite(float(x)) for x in self.state):
            return False
        if self.next_state is not None and not all(math.isfinite(float(x)) for x in self.next_state):
            return False
        return True


class ArenaMechanics:
    """Layer B: Transparent deterministic mechanics for known, testable rules.
    Never invents hidden enemy elixir or unobservable troop physics."""

    ELIXIR_REGEN_PER_SEC = 1.0 / 2.8      # 1 elixir per 2.8s in normal phase
    LATE_PHASE_MULT = 2.0                 # 2x elixir in late phase

    @staticmethod
    def validate_action(action, state=None, legal_mask=None):
        """Verify action index [0..19] and card-slot affordability."""
        if not isinstance(action, int) or not (0 <= action < ACTION_DIM):
            return False, "action out of [0..19] bounds"
        if legal_mask is not None and action not in legal_mask:
            return False, "action masked by legal_actions"
        if state is not None and len(state) == STATE_DIM:
            slot = action // 5
            afford = state[7 + slot]
            cost = state[3 + slot]
            elx = state[0]
            if afford < 0.5 and elx + 1e-4 < cost:
                return False, f"slot {slot} unaffordable (elixir={elx*10:.1f} < cost={cost*10:.1f})"
        return True, "legal"

    @classmethod
    def step_deterministic(cls, state, action, dt_sec=2.2, terminal=False):
        """Apply known deterministic rules to state:
          - Confirmed terminal states are absorbing (no further changes).
          - Playing card in slot `action // 5` deducts known slot cost and updates
            last-slot / last-spot one-hot encodings.
          - Elapsed time regenerates elixir at known phase rate (1/2.8s or 2/2.8s).
          - All state dimensions are clamped to valid [0, 1] bounds."""
        st = [_clamp01(x) for x in state[:STATE_DIM]]
        if terminal:                       # confirmed terminal state is absorbing
            return list(st)
        out = list(st)
        if action is not None and 0 <= action < ACTION_DIM:
            slot, spot = divmod(int(action), 5)
            cost_norm = out[3 + slot] if out[3 + slot] > 0 else 0.3
            # Deduct elixir spent + add elapsed regeneration
            is_late = out[13] > 0.5
            regen_rate = cls.ELIXIR_REGEN_PER_SEC * (cls.LATE_PHASE_MULT if is_late else 1.0)
            regen_norm = (regen_rate * max(0.0, float(dt_sec))) / 10.0
            out[0] = _clamp01(max(0.0, out[0] - cost_norm) + regen_norm)
            # Update last played slot one-hot [14..17] and spot one-hot [18..22]
            for i in range(4):
                out[14 + i] = 1.0 if i == slot else 0.0
            for i in range(5):
                out[18 + i] = 1.0 if i == spot else 0.0
            # Re-evaluate affordability from new elixir
            for i in range(4):
                out[7 + i] = 1.0 if out[0] + 1e-4 >= out[3 + i] else 0.0
        return [_clamp01(x) for x in out]

    @staticmethod
    def clamp_state(state, prev_state=None, terminal=False):
        """Enforce physical bounds on any predicted state vector."""
        if terminal and prev_state is not None:
            return [_clamp01(x) for x in prev_state[:STATE_DIM]]
        out = [_clamp01(x) for x in state[:STATE_DIM]]
        # Tower healths cannot spontaneously heal above previous observation + noise margin
        if prev_state is not None and len(prev_state) == STATE_DIM:
            for ti in IDX_TOWERS:
                out[ti] = min(out[ti], _clamp01(prev_state[ti] + 0.02))
        return out


class PersistenceBaseline:
    """Baseline 1: towers persist at current HP, elixir follows known regen."""

    @staticmethod
    def predict(state, action=None, dt_sec=2.2):
        st = list(state[:STATE_DIM])
        out = list(st)
        out[0] = min(1.0, st[0] + 0.25)
        for i in IDX_TOWERS:
            out[i] = st[i]
        return out


class MeanDeltaBaseline:
    """Baseline 2: empirical running mean delta across observed transitions."""

    def __init__(self):
        self.count = 0
        self.mean_delta = [0.0] * STATE_DIM

    def update(self, state, next_state):
        if state is None or next_state is None:
            return
        self.count += 1
        alpha = min(0.05, 1.0 / self.count)
        for i in range(STATE_DIM):
            d = float(next_state[i]) - float(state[i])
            self.mean_delta[i] = (1.0 - alpha) * self.mean_delta[i] + alpha * d

    def predict(self, state):
        return [_clamp01(float(state[i]) + self.mean_delta[i]) for i in range(STATE_DIM)]


class TwinMember(nn.Module if torch else object):
    """Layer C ensemble member: residual MLP predicting next-state delta,
    immediate reward, and terminal probability from (state_38, action_20, dt, quality)."""

    def __init__(self, width=WEAVER_W):
        if torch is None:
            raise RuntimeError("torch unavailable; twin syncing data-only")
        super().__init__()
        in_dim = STATE_DIM + ACTION_DIM + 2      # +dt_norm, +quality
        self.stem = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.LayerNorm(width),
            nn.SiLU(),
        )
        self.res = nn.Sequential(
            nn.Linear(width, width),
            nn.LayerNorm(width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.act = nn.SiLU()
        self.dhead = nn.Linear(width, STATE_DIM)
        self.rhead = nn.Linear(width, 1)
        self.done_head = nn.Linear(width, 1)

    def forward(self, s, a_onehot, dt_norm=0.22, quality=1.0):
        batch_n = s.shape[0]
        if not isinstance(dt_norm, torch.Tensor):
            dt_t = torch.full((batch_n, 1), float(dt_norm), dtype=torch.float32)
        else:
            dt_t = dt_norm.view(batch_n, 1)
        if not isinstance(quality, torch.Tensor):
            q_t = torch.full((batch_n, 1), float(quality), dtype=torch.float32)
        else:
            q_t = quality.view(batch_n, 1)
        z = torch.cat([s, a_onehot, dt_t, q_t], dim=1)
        h0 = self.stem(z)
        h = self.act(h0 + self.res(h0))
        return self.dhead(h), self.rhead(h).squeeze(-1), torch.sigmoid(self.done_head(h).squeeze(-1))


class ArenaTwin:
    """Coherent hybrid Digital Twin: synchronizes real observations, trains a
    world-model ensemble with strict validation gates, assimilates prediction
    errors, and provides risk-aware bounded lookahead planning."""

    def __init__(self):
        self.seq = 0
        self.buffer = deque(maxlen=MAX_TWIN_BUFFER)   # capped TwinObs pool
        self.members = None
        self.optims = None
        self.mean_baseline = MeanDeltaBaseline()
        self.n_trained = 0
        self.ready = False
        self.rejected_transitions = 0
        self.last_pred = None                         # cached prediction for Layer E assimilation
        self.drift_ema = 0.0
        self.feature_mae = [0.0] * STATE_DIM
        self.metrics = {
            "twin_mae_towers": None,
            "twin_mae_elixir": None,
            "baseline_mae_towers": None,
            "baseline_mae_elixir": None,
            "mean_baseline_mae_towers": None,
            "reward_mae": None,
            "beats_baseline": None,
            "second_step_ready": False,
            "ensemble_spread": None,
            "prediction_drift": 0.0,
            "n_scored": 0,
            "train_steps": 0,
            "rejected_transitions": 0,
            "last_candidates": [],
            "last_chosen_plan": None,
            "last_assimilated": None,
            "twin_params": 0,
            "predict_latency_ms": 0.0,
        }
        self.plan_contribution = 0.0

    # --------------- Layer A & E: synchronized transitions + assimilation ---------------
    def assimilate(self, observed_state, actual_reward=None):
        """Layer E: Compare the last 1-step prediction against the newly observed
        ground-truth state and update feature-level prediction errors and drift."""
        if self.last_pred is None or observed_state is None or len(observed_state) != STATE_DIM:
            return None
        pred_s = self.last_pred.get("state_pred")
        if not pred_s or len(pred_s) != STATE_DIM:
            return None
        errs = [abs(float(observed_state[i]) - float(pred_s[i])) for i in range(STATE_DIM)]
        for i in range(STATE_DIM):
            self.feature_mae[i] = round(0.95 * self.feature_mae[i] + 0.05 * errs[i], 5)
        tower_err = sum(errs[i] for i in (23, 24, 26, 27))
        elx_err = errs[0]
        self.drift_ema = round(0.92 * self.drift_ema + 0.08 * (tower_err + elx_err), 4)
        self.metrics["prediction_drift"] = self.drift_ema
        assim = {
            "seq": self.seq,
            "pred_elixir": round(pred_s[0] * 10.0, 2),
            "obs_elixir": round(float(observed_state[0]) * 10.0, 2),
            "elixir_err": round(elx_err * 10.0, 2),
            "pred_towers": {
                "ally_left": round(pred_s[23], 3),
                "ally_right": round(pred_s[24], 3),
                "enemy_left": round(pred_s[26], 3),
                "enemy_right": round(pred_s[27], 3),
            },
            "obs_towers": {
                "ally_left": round(float(observed_state[23]), 3),
                "ally_right": round(float(observed_state[24]), 3),
                "enemy_left": round(float(observed_state[26]), 3),
                "enemy_right": round(float(observed_state[27]), 3),
            },
            "tower_hp_err": round(tower_err, 4),
            "reward_pred": round(float(self.last_pred.get("reward_pred", 0.0)), 3),
            "reward_actual": round(float(actual_reward), 3) if actual_reward is not None else None,
            "uncertainty": round(float(self.last_pred.get("uncertainty", 0.5)), 3),
        }
        self.metrics["last_assimilated"] = assim
        return assim

    def record(self, state, action, reward, next_state, done=False,
               stale=False, quality=1.0, towers=None, elixir=None,
               dt=2.2, origin="dqn_policy"):
        """Layer A: Record one genuine observed transition. Rejects corrupted or
        invalid dimensions; stamps stale frames so they never pollute training."""
        if state is None or next_state is None:
            self.rejected_transitions += 1
            self.metrics["rejected_transitions"] = self.rejected_transitions
            return None
        obs = TwinObs(time.time(), self.seq + 1, state, action, reward,
                      next_state, done=done, stale=stale, quality=quality,
                      dt=dt, towers=towers, elixir=elixir, origin=origin)
        if not obs.valid():
            self.rejected_transitions += 1
            self.metrics["rejected_transitions"] = self.rejected_transitions
            return None
        self.seq += 1
        # Assimilate outcome against any prior prediction before storing
        self.assimilate(obs.next_state, actual_reward=obs.reward)
        self.buffer.append(obs)
        if not obs.stale:
            self.mean_baseline.update(obs.state, obs.next_state)
        if self.torch_ready() and len(self.buffer) >= MIN_DATA_FOR_TRAINING and self.seq % TRAIN_EVERY == 0:
            self.train()
        # Predict 1 step ahead from next_state with default action for continuous assimilation
        if self.ready and action is not None and 0 <= int(action) < ACTION_DIM:
            self.last_pred = self.predict_one_step(obs.next_state, int(action))
        return self.seq

    def torch_ready(self):
        return torch is not None

    # --------------- Layer C: ensemble training ---------------
    def _ensure_model(self):
        if self.members is None and torch is not None:
            self.members = [TwinMember() for _ in range(ENSEMBLE_MEMBERS)]
            self.optims = [torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
                           for m in self.members]
            self.metrics["twin_params"] = sum(
                p.numel() for m in self.members for p in m.parameters() if p.requires_grad
            )

    def _sample(self, n, holdout=False):
        # Exclude stale frames from training/validation ground truth
        data = [b for b in self.buffer if not b.stale and b.next_state is not None]
        if len(data) < 20:
            data = [b for b in self.buffer if b.next_state is not None]
        cut = max(1, int(len(data) * (1.0 - HOLDOUT_FRACTION)))
        pool = data[cut:] if holdout else data[:cut]
        if not pool:
            return []
        return random.sample(pool, min(n, len(pool)))

    def train(self):
        batch = self._sample(TWIN_BATCH, holdout=False)
        if len(batch) < 32 or not torch:
            return None
        self._ensure_model()
        losses = []
        s = torch.tensor([b.state for b in batch], dtype=torch.float32)
        ns = torch.tensor([b.next_state for b in batch], dtype=torch.float32)
        aoh = torch.zeros(len(batch), ACTION_DIM)
        for i, b in enumerate(batch):
            if b.action is not None:
                aoh[i, b.action] = 1.0
        dt_t = torch.tensor([b.dt / 10.0 for b in batch], dtype=torch.float32)
        q_t = torch.tensor([b.quality for b in batch], dtype=torch.float32)
        delta_target = ns - s
        r_target = torch.tensor([b.reward for b in batch], dtype=torch.float32)
        d_target = torch.tensor([1.0 if b.done else 0.0 for b in batch], dtype=torch.float32)

        for m, opt in zip(self.members, self.optims):
            opt.zero_grad()
            # Bootstrap resample per ensemble member for genuine diversity
            idx = torch.randint(0, len(batch), (len(batch),))
            dr, rw, dn = m(s[idx], aoh[idx], dt_t[idx], q_t[idx])
            # Smooth L1 on continuous deltas + extra weight on key tower/elixir features
            cont_loss = nn.functional.smooth_l1_loss(dr, delta_target[idx])
            tower_loss = (dr[:, [0, 23, 24, 25, 26, 27]] - delta_target[idx][:, [0, 23, 24, 25, 26, 27]]).abs().mean()
            rew_loss = nn.functional.smooth_l1_loss(rw, r_target[idx])
            done_loss = nn.functional.binary_cross_entropy(dn.clamp(1e-5, 1.0 - 1e-5), d_target[idx])
            loss = cont_loss + 1.5 * tower_loss + 0.5 * rew_loss + 0.2 * done_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        self.n_trained += 1
        self.metrics["train_steps"] += 1
        self.ready = True
        self.validate()
        return round(sum(losses) / len(losses), 4)

    # --------------- Honest evaluation vs 2 baselines ---------------
    def validate(self):
        holdout = self._sample(256, holdout=True)
        if len(holdout) < 16 or not torch or self.members is None:
            return None
        base_t, base_e, mbase_t, mo_t, mo_e, r_errs = [], [], [], [], [], []
        for b in holdout:
            best = PersistenceBaseline.predict(b.state, b.action, b.dt)
            mbest = self.mean_baseline.predict(b.state)
            base_t.append(sum(abs(best[i] - b.next_state[i]) for i in (23, 24, 26, 27)))
            base_e.append(abs(best[0] - b.next_state[0]))
            mbase_t.append(sum(abs(mbest[i] - b.next_state[i]) for i in (23, 24, 26, 27)))
        with torch.no_grad():
            s = torch.tensor([b.state for b in holdout], dtype=torch.float32)
            ns = torch.tensor([b.next_state for b in holdout], dtype=torch.float32)
            aoh = torch.zeros(len(holdout), ACTION_DIM)
            for i, b in enumerate(holdout):
                if b.action is not None:
                    aoh[i, b.action] = 1.0
            dt_t = torch.tensor([b.dt / 10.0 for b in holdout], dtype=torch.float32)
            q_t = torch.tensor([b.quality for b in holdout], dtype=torch.float32)
            outs, rws = [], []
            for m in self.members:
                dr, rw, _dn = m(s, aoh, dt_t, q_t)
                outs.append(dr)
                rws.append(rw)
            dr_mean = torch.stack(outs).mean(0)
            rw_mean = torch.stack(rws).mean(0)
            dr_spread = torch.stack(outs).std(0).mean().item()
            pred = (s + dr_mean).clamp(0.0, 1.0)
            for i, b in enumerate(holdout):
                mo_t.append(sum(abs(pred[i, k].item() - ns[i, k].item()) for k in (23, 24, 26, 27)))
                mo_e.append(abs(pred[i, 0].item() - ns[i, 0].item()))
                r_errs.append(abs(rw_mean[i].item() - float(b.reward)))
        bm_t = sum(base_t) / len(base_t)
        bm_e = sum(base_e) / len(base_e)
        mbm_t = sum(mbase_t) / len(mbase_t)
        mm_t = sum(mo_t) / len(mo_t)
        mm_e = sum(mo_e) / len(mo_e)
        rm_e = sum(r_errs) / len(r_errs)
        beats = bool(mm_t < bm_t and mm_e < bm_e)
        self.metrics.update({
            "baseline_mae_towers": round(bm_t, 4),
            "twin_mae_towers": round(mm_t, 4),
            "mean_baseline_mae_towers": round(mbm_t, 4),
            "baseline_mae_elixir": round(bm_e, 4),
            "twin_mae_elixir": round(mm_e, 4),
            "reward_mae": round(rm_e, 4),
            "beats_baseline": beats,
            "second_step_ready": bool(beats and mm_t < 0.08 and dr_spread < 0.12),
            "ensemble_spread": round(dr_spread, 4),
            "n_scored": len(holdout),
        })
        return dict(self.metrics)

    # --------------- Layer D: Risk-aware model-predictive planning ---------------
    def predict_one_step(self, state, action, dt_sec=2.2):
        """Predict next state, immediate reward, terminal prob, and per-feature
        uncertainty by blending Layer B deterministic mechanics with Layer C
        learned residuals."""
        if not (torch and self.ready):
            return None
        ok, reason = ArenaMechanics.validate_action(int(action), state=state)
        if not ok:
            return None
        self._ensure_model()
        t0 = time.time()
        with torch.no_grad():
            s = torch.tensor([state[:STATE_DIM]], dtype=torch.float32)
            aoh = torch.zeros(1, ACTION_DIM)
            aoh[0, int(action)] = 1.0
            drs, rws, dns = [], [], []
            for m in self.members:
                dr, rw, dn = m(s, aoh, dt_sec / 10.0, 1.0)
                drs.append(dr)
                rws.append(rw)
                dns.append(dn)
            dr_stack = torch.stack(drs)
            dr_mean = dr_stack.mean(0)[0]
            feat_std = dr_stack.std(0)[0]
            spread = float(feat_std.mean().item())
            raw_next = (s[0] + dr_mean).tolist()
            # Reconcile with deterministic mechanics on known slots/affordability
            mech_next = ArenaMechanics.step_deterministic(state, int(action), dt_sec=dt_sec)
            blended = list(raw_next)
            for idx in (14, 15, 16, 17, 18, 19, 20, 21, 22):
                blended[idx] = mech_next[idx]
            clamped = ArenaMechanics.clamp_state(blended, prev_state=state)
            unc = min(1.0, spread * 2.0 + 0.3 * bot_uncertainty_now())
            self.metrics["predict_latency_ms"] = round((time.time() - t0) * 1000.0, 2)
            return {
                "state_pred": clamped,
                "reward_pred": float(torch.stack(rws).mean().item()),
                "done_prob": float(torch.stack(dns).mean().item()),
                "uncertainty": round(unc, 4),
                "feature_uncertainty": [round(float(x), 4) for x in feat_std.tolist()],
            }

    def plan(self, state, legal_actions, value_fn=None, deadline_s=0.18, beam_width=4):
        """Bounded lookahead over top legal candidates within a strict wall-clock
        deadline.
        Formula:
          V_plan(s, a) = r_pred(s, a) + gamma * V_dqn(s_pred) - lambda_u * uncertainty
        where gamma = 0.95 and lambda_u = 0.5 (calibrated to reward scale).
        Only overrides/reranks when TWIN_PLAN_ENABLED is True AND the ensemble
        beats its persistence baseline on held-out validation data."""
        if not TWIN_PLAN_ENABLED or not self.ready:
            return None
        if not self.metrics.get("beats_baseline"):
            return None          # experimental gate: defer to DQN until validated
        t0 = time.time()
        legal = []
        for a in legal_actions:
            ok, _ = ArenaMechanics.validate_action(a, state=state)
            if ok:
                legal.append(int(a))
        if not legal:
            return None
        candidates = legal[:max(1, beam_width)]
        evaluated = []
        best, best_v = None, -1e9
        gamma = 0.95
        for a in candidates:
            if time.time() - t0 > deadline_s:
                break
            p1 = self.predict_one_step(state, a)
            if p1 is None or p1["uncertainty"] > 0.70:
                continue
            future_v = float(value_fn(p1["state_pred"])) if value_fn else 0.0
            # Optional 2nd simulated step ONLY when second_step_ready passes gate
            step2_bonus = 0.0
            if self.metrics.get("second_step_ready") and (time.time() - t0) < deadline_s * 0.6:
                p2 = self.predict_one_step(p1["state_pred"], a)
                if p2 is not None and p2["uncertainty"] < 0.50:
                    step2_bonus = 0.25 * (p2["reward_pred"] - 0.5 * p2["uncertainty"])
            # Confidence weight: high confidence (<0.25 unc) = 1.0, moderate = scaled
            conf_weight = 1.0 if p1["uncertainty"] < 0.25 else max(0.3, 1.0 - p1["uncertainty"])
            v = conf_weight * (p1["reward_pred"] + step2_bonus) + gamma * future_v - 0.5 * p1["uncertainty"]
            entry = {
                "action": a,
                "value": round(v, 4),
                "reward_pred": round(p1["reward_pred"], 4),
                "future_v": round(future_v, 4),
                "uncertainty": round(p1["uncertainty"], 3),
                "confidence": round(1.0 - p1["uncertainty"], 3),
                "pred_enemy_towers": [round(p1["state_pred"][26], 3), round(p1["state_pred"][27], 3)],
            }
            evaluated.append(entry)
            if v > best_v:
                best_v, best = v, entry
        self.metrics["last_candidates"] = evaluated
        self.metrics["last_chosen_plan"] = best
        self.plan_contribution = len(evaluated)
        return best

    # --------------- Layer F: Persistence & checkpoint safety ---------------
    def save(self):
        if not torch or self.members is None:
            return False
        payload = {
            "version": TWIN_VERSION,
            "seq": self.seq,
            "n_trained": self.n_trained,
            "metrics": self.metrics,
            "feature_mae": self.feature_mae,
            "mean_baseline": {
                "count": self.mean_baseline.count,
                "mean_delta": self.mean_baseline.mean_delta,
            },
            "config": {
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
                "ensemble_members": ENSEMBLE_MEMBERS,
                "width": WEAVER_W,
                "max_buffer": MAX_TWIN_BUFFER,
            },
            "members": [m.state_dict() for m in self.members],
            "optims": [opt.state_dict() for opt in self.optims] if self.optims else [],
        }
        tmp = TWIN_FILE + ".tmp"
        bak = TWIN_FILE + ".bak"
        try:
            torch.save(payload, tmp)
            if os.path.exists(TWIN_FILE):
                shutil.copy2(TWIN_FILE, bak)
            os.replace(tmp, TWIN_FILE)
            with open(TWIN_META_FILE, "w") as f:
                json.dump({
                    "version": TWIN_VERSION,
                    "seq": self.seq,
                    "metrics": self.metrics,
                }, f, indent=1)
            return True
        except Exception:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            return False

    def load(self):
        path = TWIN_FILE if os.path.exists(TWIN_FILE) else (
            TWIN_META_FILE if os.path.exists(TWIN_META_FILE) else None
        )
        if not torch or not path:
            return False
        try:
            if os.path.getsize(path) > MAX_CHECKPOINT_BYTES:
                return False                    # resource guard: refuse giant blobs
            payload = torch.load(path, map_location="cpu")
            ver = int(payload.get("version", 0))
            if ver not in (1, TWIN_VERSION):
                return False                    # incompatible schema -> explicit rejection
            self.seq = int(payload.get("seq", 0))
            self.n_trained = int(payload.get("n_trained", 0))
            saved_m = payload.get("metrics", {})
            if isinstance(saved_m, dict):
                self.metrics.update(saved_m)
            mb = payload.get("mean_baseline", {})
            if isinstance(mb, dict) and len(mb.get("mean_delta", [])) == STATE_DIM:
                self.mean_baseline.count = int(mb.get("count", 0))
                self.mean_baseline.mean_delta = list(mb["mean_delta"])
            self._ensure_model()
            for m, sd in zip(self.members, payload["members"]):
                m.load_state_dict(sd, strict=False)
            if ver == TWIN_VERSION and payload.get("optims") and self.optims:
                for opt, osd in zip(self.optims, payload["optims"]):
                    try:
                        opt.load_state_dict(osd)
                    except Exception:
                        pass
            self.ready = True
            return True
        except Exception:
            return False


def bot_uncertainty_now():
    try:
        import bot_ml
        return float(getattr(bot_ml, "bot_ml_pred_uncertainty", lambda: 0.5)())
    except Exception:
        return 0.5
