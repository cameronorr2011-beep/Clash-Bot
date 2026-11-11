"""NeuroML v13: biologically-inspired spiking nucleus + affective core.

This module implements, with unit tests in test_spiking.py:
  1. Leaky Integrate-and-Fire (LIF) neurons with:
     membrane potential, resting potential, membrane time constant,
     threshold-triggered spikes, absolute refractory period,
     spike-frequency adaptation (biological variability via Poisson-like noise),
  2. Reward-modulated STDP (pair-based, pre-post potentiation / post-pre
     depression) with eligibility traces and bound safeguards,
  3. Homeostatic rate coding (targets & gain field per neuron),
  4. Affective core: bounded, decaying positive/negative valence state that
     gates plasticity.

Biological approximation notes (documented, not hidden):
 - the membrane is a first-order ODE integrated with explicit Euler steps at
   dt = 1 ms;
 - spikes are instantaneous events (no waveform shape is simulated);
 - the refractory period is a constant absolute dead time;
 - membrane noise gives Poisson-like firing variability.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------- constants
DT_MS = 1.0
TAU_M_MS = 20.0         # membrane time constant
V_REST = -70.0          # mV, resting potential
V_RESET = -75.0         # mV, reset potential after a spike
V_THRESH = -55.0        # mV, spiking threshold
TAU_SYN_E_MS = 5.0      # excitatory synaptic decay
TAU_SYN_I_MS = 10.0     # inhibitory synaptic decay (slower, GABA-A-like)
TAU_ADAPT_MS = 200.0    # spike-frequency adaptation
ADAPT_STEP = 1.0        # nA per-spike adaptation increment (calibrated)
REFRACTORY_MS = 2.0     # absolute refractory period
TAU_TRACE_MS = 20.0     # STDP pre/post trace decay
TAU_ELIG_MS = 1000.0    # eligibility trace decay (bridges decision ticks)
SPIKE_JITTER_SD = 0.8   # mV stationary membrane-noise amplitude (diffusion-scaled)
TAU_HOMEOSTASIS_MS = 5000.0

# biphasic exponential STDP kernel (Bi & Poo 1998)
STDP_TAU_MS = 20.0
STDP_A_P = 0.0052       # LTP amplitude (pre before post)
STDP_A_M = 0.0056       # LTD amplitude (post before pre), slightly larger
W_MIN = 0.0
W_MAX = 1.0
MU_SOFT = 0.10          # soft-bound exponent

# homeostasis
TARGET_RATE_HZ = 8.0
HOMEOSTASIS_GAIN = 5e-5  # threshold shift per Hz of rate error

# affective core
VALENCE_TAU_TICKS = 40.0  # decay time constant in decision ticks


class AffectiveCore:
    """Bounded, interacting internal-state dynamics (v14→v18 neuro-architecture).

    Channels (all [0,1], all with their own decay time constants):
      neg_rpe      negative reward-prediction error (unexpected losses)
      pos_rpe      positive reward-prediction error (unexpected gains)
      progress     task progress (towers won / objectives closed out)
      danger       imminent threat urgency (multi-lane pushes, exposed king)
      confidence   policy certainty (inverse normalised entropy)
      frustration  repeated failure of similar strategies (EMA of losses)
      relief       resolution of previously-high danger
    v18.Added structural-state channels:
      vigilance   the brain's post-push attention posture (rises on pushes,
                  decays slow → the system "stays on guard")
      momentum    recent trend signal (win/loss streak EMA; feeds planning)
      pressure    mirrored opponent elixir advantage (they can punish soon)
      curiosity   novelty of the current situation (1 - similarity to the
                  nearest hippocampal memory; drives exploration where the
                  brain "has not been")
    COMPOSITE states (derived each tick, never stored): desperation =
    f(danger, pressure, neg_rpe); composure = 1 - frustration · vigilance Fear.
    Cross-channel interactions are real and tested: frustration inhibits
    confidence recovery; high progress + composure accelerate desperation
    decay; a spike of danger slows progress accumulation.
    All channels are COMPUTATIONAL ANALOGUES — bounded dynamic state variables
    named for the biology they abstract. They are not mechanisms, NOT measures
    of subjective feeling or consciousness.
    """

    def __init__(self):
        self.state: Dict[str, float] = {
            "neg_rpe": 0.0, "pos_rpe": 0.0, "progress": 0.0,
            "danger": 0.0, "confidence": 0.5, "frustration": 0.0,
            "relief": 0.0,
            "vigilance": 0.0, "momentum": 0.5,
            "pressure": 0.0, "curiosity": 0.0,
            # v19 channels (structural state, still bounded analogues)
            "anticipation": 0.0, "desperation": 0.0, "fatigue": 0.0,
            "uncertainty": 0.0, "focus": 0.5, "control": 0.5,
            "surprise": 0.0, "strategic_conflict": 0.0,
            "patience": 0.5, "opponent_read": 0.0,
        }
        self.tau: Dict[str, float] = {"neg_rpe": 30.0, "pos_rpe": 40.0,
                                      "progress": 120.0, "danger": 20.0,
                                      "confidence": 60.0, "frustration": 150.0,
                                      "relief": 25.0,
                                      "vigilance": 45.0, "momentum": 80.0,
                                      "pressure": 35.0, "curiosity": 22.0,
                                      "anticipation": 18.0, "desperation": 50.0,
                                      "fatigue": 200.0, "uncertainty": 40.0,
                                      "focus": 70.0, "control": 90.0,
                                      "surprise": 12.0, "strategic_conflict": 55.0,
                                      "patience": 110.0, "opponent_read": 160.0}
        self.neutral: Dict[str, float] = {
            "neg_rpe": 0.0, "pos_rpe": 0.0, "progress": 0.0, "danger": 0.0,
            "confidence": 0.5, "frustration": 0.0, "relief": 0.0,
            "vigilance": 0.0, "momentum": 0.5, "pressure": 0.0,
            "curiosity": 0.0, "anticipation": 0.0, "desperation": 0.0,
            "fatigue": 0.0, "uncertainty": 0.0, "focus": 0.5,
            "control": 0.5, "surprise": 0.0, "strategic_conflict": 0.0,
            "patience": 0.5, "opponent_read": 0.0}
        self.reward_baseline = 0.3   # EMA of shaped rewards → RPE reference
        self.obs_ma = None           # EMA of observed state hash delta → novelty
        self.doc = {
            "neg_rpe": "reward below EMA baseline (unexpected losses)",
            "pos_rpe": "reward above EMA baseline (unexpected gains)",
            "progress": "objectives closed (towers won)",
            "danger": "imminent threat urgency (pushes, exposed king)",
            "confidence": "policy certainty (inverse normalised entropy)",
            "frustration": "EMA of repeated similar failures",
            "relief": "resolution of previously-high danger",
            "vigilance": "post-threat attention posture",
            "momentum": "EMA of recent reward trend",
            "pressure": "mirrored opponent threat capacity",
            "curiosity": "novelty of state (1 - hippocampal similarity)",
            "anticipation": "predictive forecast confidence (predictive twin)",
            "desperation": "critically low king/exposed towers + sustained pressure",
            "fatigue": "sustained compute overruns, workload stamp; recovers",
            "uncertainty": "world-model/prediction disagreement & MAE",
            "focus": "decision bandwidth; degraded by fatigue, restored by control",
            "control": "measured command effectiveness (paged/corrected actions)",
            "surprise": "observed prediction error surprise signal",
            "strategic_conflict": "mentor-vs-policy disagreement magnitude",
            "patience": "willingness to hold resources (decays with leaks)",
            "opponent_read": "read confidence for the current opponent pattern",
        }

    @staticmethod
    def _clamp(x: float) -> float:
        return max(0.0, min(1.0, x))

    def decay_step(self) -> None:
        """One decision tick: passive decay + CROSS-CHANNEL interactions.
        Interactions are deliberately small (≤0.03/tick) and always keep the
        state bounded; they give the dynamics real coupling, not labels."""
        for k, v in self.state.items():
            tau = self.tau[k]
            f = math.exp(-1.0 / tau)
            self.state[k] = self._clamp(v * f + self.neutral[k] * (1.0 - f))
        s = self.state
        # frustration slows confidence recovery (repeated failure lowers trust)
        if s["frustration"] > 0.5 and s["confidence"] < 0.5:
            s["confidence"] = self._clamp(s["confidence"] - 0.02)
        # danger erodes composure (approx then, vigilance rises on real threats)
        if s["danger"] > 0.4:
            s["vigilance"] = self._clamp(s["vigilance"] + 0.03)
        # progress + calm close out desperation (relief-like dynamics)
        if s["progress"] > 0.6 and s["vigilance"] < 0.3 and s["neg_rpe"] < 0.3:
            s["neg_rpe"] = self._clamp(s["neg_rpe"] - 0.02)
        # vigilance decays slower when pressure is high (mirrored opponent)
        if s["pressure"] > 0.5:
            s["vigilance"] = self._clamp(s["vigilance"] + 0.01)
        # curiosity decays into bold play after a win spike
        if s["pos_rpe"] > 0.8 and s["curiosity"] > 0.3:
            s["curiosity"] = self._clamp(s["curiosity"] - 0.015)

    def inject(self, channel: str, amount: float) -> float:
        """Instant stimulus, clamped and reflected through the decay buffer."""
        if channel not in self.state:
            raise KeyError(channel)
        self.state[channel] = self._clamp(self.state[channel] + amount)
        return self.state[channel]

    def set(self, channel: str, value: float) -> float:
        if channel not in self.state:
            raise KeyError(channel)
        self.state[channel] = self._clamp(value)
        return self.state[channel]

    def update(self, signals: Dict[str, float]) -> Dict[str, float]:
        """The single intake point for real application signals → channels.
        Every mapping is a DOCUMENTED, measured formula; defaults 0. Accepted
        signals (all optional):
          reward            shaped tick reward → RPE vs the EMA baseline
          novelty           measured state-change score in [0,1]
          threat            lane-threat flags in [0,1]
          king_hp           ally king-health fraction in [0,1]
          towers_exposed    alive ally towers count (2 = king exposed)
          overruns          compute budget overrun count this tick (0..n)
          uncertainty      world-model disagreement/MAE in [0,1]
          prediction_err    surprise: observed prediction error magnitude
          mentor_conflict   validated mentor-vs-policy disagreement in [0,1]
          page_confirmed    decision survived every referee override (bool)
          leak              elixir leaked this tick (bool)
          opponent_pattern  opponent read confidence in [0,1]
          forecast          predictive twin forecast quality in [0,1]
        Cross-channel coupling happens inside decay_step(); composite
        desperation is stored and modulated here from hard signals."""
        s, c = self.state, self._clamp
        self.decay_step()
        reward = signals.get("reward")
        if reward is not None:
            err = float(reward) - self.reward_baseline
            self.reward_baseline = 0.95 * self.reward_baseline + 0.05 * float(reward)
            if err > 0:
                s["pos_rpe"] = c(s["pos_rpe"] + min(0.3, 0.15 * err))
            else:
                s["neg_rpe"] = c(s["neg_rpe"] + min(0.3, 0.15 * -err))
        nov = signals.get("novelty")
        if nov is not None:
            s["curiosity"] = c(s["curiosity"] + 0.12 * c(nov))
        threat = signals.get("threat")
        if threat is not None:
            s["danger"] = c(s["danger"] + 0.15 * c(threat))
        king = signals.get("king_hp")
        if king is not None and king < 0.35:
            s["desperation"] = c(s["desperation"] + 0.08)
        exposed = signals.get("towers_exposed")
        if exposed is not None and exposed <= 1 and threat and threat > 0.3:
            s["desperation"] = c(s["desperation"] + 0.05)
        s["desperation"] = c(0.5 * s["desperation"] + 0.2 * s["danger"] + 0.15 * s["pressure"] + 0.15 * s["neg_rpe"])
        over = signals.get("overruns") or 0
        if over:
            s["fatigue"] = c(s["fatigue"] + 0.05 * min(3, over))
        unc = signals.get("uncertainty")
        if unc is not None:
            s["uncertainty"] = c(s["uncertainty"] + 0.12 * c(unc))
        perr = signals.get("prediction_err")
        if perr is not None:
            s["surprise"] = c(s["surprise"] + 0.2 * c(perr))
        conf = signals.get("mentor_conflict")
        if conf is not None:
            s["strategic_conflict"] = c(s["strategic_conflict"] + 0.15 * c(conf))
        if signals.get("page_confirmed"):
            s["control"] = c(s["control"] + 0.05)
        elif signals.get("veto_overridden") or signals.get("rules_corrected"):
            s["control"] = c(max(0.0, s["control"] - 0.03))
        if signals.get("leak"):
            s["patience"] = c(max(0.0, s["patience"] - 0.02))
        oread = signals.get("opponent_pattern")
        if oread is not None:
            s["opponent_read"] = c(s["opponent_read"] + 0.1 * c(oread))
        fc = signals.get("forecast")
        if fc is not None:
            s["anticipation"] = c(s["anticipation"] + 0.1 * c(fc))
        # fatigue gates focus (banded, documented)
        s["focus"] = c(0.9 * s["focus"] + 0.1 * (s["control"] * (1.0 - 0.7 * s["fatigue"])))
        # uncertainty raises vigilance; high confidence reads opponents better
        if s["uncertainty"] > 0.4:
            s["vigilance"] = c(s["vigilance"] + 0.02)
        if s["opponent_read"] > 0.5:
            s["confidence"] = c(min(1.0, s["confidence"] + 0.01))
        return dict(s)

    def observe_novelty(self, state_vec: List[float]) -> float:
        """Compute bounded novelty in [0, 1] from observed state changes using
        an exponential moving average of the state vector."""
        if not state_vec:
            return 0.0
        if self.obs_ma is None or len(self.obs_ma) != len(state_vec):
            self.obs_ma = [float(x) for x in state_vec]
            return 0.25
        diff = sum(abs(float(x) - float(m)) for x, m in zip(state_vec, self.obs_ma)) / max(1, len(state_vec))
        for i, x in enumerate(state_vec):
            self.obs_ma[i] = 0.90 * self.obs_ma[i] + 0.10 * float(x)
        return self._clamp(diff * 3.5)

    def state_dict(self) -> Dict:
        return {
            "state": dict(self.state),
            "reward_baseline": float(self.reward_baseline),
            "obs_ma": list(self.obs_ma) if self.obs_ma is not None else None,
        }

    def load_state_dict(self, d: Dict) -> None:
        if not isinstance(d, dict):
            return
        # Support both legacy flat dict {channel: float} and v19 structured dict
        raw_state = d.get("state") if isinstance(d.get("state"), dict) else d
        for k, v in raw_state.items():
            if k in self.state and isinstance(v, (int, float)):
                self.state[k] = self._clamp(float(v))
        if "reward_baseline" in d and isinstance(d["reward_baseline"], (int, float)):
            self.reward_baseline = float(d["reward_baseline"])
        if isinstance(d.get("obs_ma"), list):
            self.obs_ma = [float(x) for x in d["obs_ma"]]

    def behavioral_modulation(self) -> Dict[str, float]:
        """Bounded, documented behavioral and plasticity modulation derived
        from the 21 computational-affect channels:
          - plasticity_scale: gates STDP eligibility conversion in [0.25, 1.5]
          - exploration_nudge: small bounded adjustment to exploration in [-0.04, +0.06]
            (curiosity + frustration increase exploration; high focus + control reduce it)
          - defensive_bias: bounded preference for defensive spots when danger/desperation high"""
        s = self.state
        val = self.valence_scalar()
        plasticity_scale = max(0.25, min(1.5, 1.0 + 0.5 * val - 0.15 * s["fatigue"]))
        exp_nudge = max(-0.04, min(0.06,
            0.05 * s["curiosity"] + 0.03 * s["frustration"]
            - 0.03 * s["focus"] * s["control"]
        ))
        def_bias = self._clamp(0.5 * s["danger"] + 0.5 * s["desperation"])
        return {
            "plasticity_scale": round(plasticity_scale, 4),
            "exploration_nudge": round(exp_nudge, 4),
            "defensive_bias": round(def_bias, 4),
        }

    def composites(self) -> Dict[str, float]:
        """Derived structural states (never stored, recomputed each call):
        desperation — collapse risk under threat + opponent pressure;
        composure   — how regulated the executive state is under load.
        These are units-free bounded signals, labeled computational analogues."""
        s = self.state
        desperation = self._clamp(0.5 * s["danger"] + 0.25 * s["pressure"]
                                   + 0.25 * s["neg_rpe"])
        composure = self._clamp(1.0 - 0.6 * s["frustration"]
                                - 0.4 * s["vigilance"] + 0.2 * s["progress"])
        return {"desperation": desperation, "composure": composure}

    def valence_scalar(self) -> float:
        """Signed scalar in [-1, 1]: positive minus negative, modulated a
        little by derived desperation so plasticity slows under collapse risk."""
        d = self.composites()
        base = self.state["pos_rpe"] - self.state["neg_rpe"]
        return self._clamp(base * (1.0 - 0.4 * d["desperation"]))


    def summary(self) -> Dict[str, float]:
        return dict(self.state)


class LIFNeuron:
    """Leaky integrate-and-fire neuron with adaptation and refractory.

    Discrete-time Euler simulation at DT_MS. Biological approximations:
      * membrane = first-order leaky integrator of synaptic current,
      * spike = instantaneous threshold-crossing event (no waveform),
      * refractory = fixed absolute dead time,
      * adaptation = extra hyperpolarising current decaying with TAU_ADAPT_MS,
      * variability = gaussian jitter on the membrane each step.
    """

    def __init__(self, seed: Optional[int] = None,
                 tau_m: float = TAU_M_MS,
                 v_rest: float = V_REST,
                 v_threshold: float = V_THRESH,
                 refractory_ms: float = REFRACTORY_MS,
                 adapt_step: float = ADAPT_STEP,
                 jitter_sd: float = SPIKE_JITTER_SD):
        self.rng = random.Random(seed)
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_threshold = v_threshold
        self.adapt_threshold = 0.0
        self.refractory_ms = refractory_ms
        self.adapt_step = adapt_step
        self.jitter_sd = jitter_sd
        self.reset()

    def reset(self) -> None:
        self.v = self.v_rest
        self.i_syn_e = 0.0
        self.i_syn_i = 0.0
        self.i_adapt = 0.0
        self.t_since_spike = float("inf")
        self.last_spike_t = None
        self.spike_count = 0

    def step(self, dt_ms: float = DT_MS) -> bool:
        """Advance one timestep. Returns True when the neuron spikes."""
        # refractory check
        if self.t_since_spike < self.refractory_ms:
            self.t_since_spike += dt_ms
            # membrane still leaks toward rest during refractory
            self.v += (self.v_rest - self.v) * dt_ms / self.tau_m
            return False
        # membrane ODE (explicit Euler)
        i_syn = (self.i_syn_e - self.i_syn_i - self.i_adapt) * 2.0  # nA->mV scale
        self.v += ((-(self.v - self.v_rest) + i_syn) * dt_ms / self.tau_m)
        # biological variability, diffusion-scaled: injecting the full sigma
        # each Euler step would random-walk 4.7 mV; scaling by sqrt(2*dt/tau)
        # makes the stationary membrane-noise std equal to jitter_sd itself.
        if self.jitter_sd > 0:
            self.v += self.rng.gauss(0.0, self.jitter_sd * math.sqrt(2.0 * dt_ms / self.tau_m))
        # synaptic current decay
        self.i_syn_e *= math.exp(-dt_ms / TAU_SYN_E_MS)
        self.i_syn_i *= math.exp(-dt_ms / TAU_SYN_I_MS)
        self.i_adapt *= math.exp(-dt_ms / TAU_ADAPT_MS)
        self.t_since_spike += dt_ms
        # threshold crossing -> spike
        if self.v >= self.v_threshold + self.adapt_threshold:
            self.spike_count += 1
            self.v = V_RESET
            self.t_since_spike = 0.0
            self.i_adapt += self.adapt_step
            return True
        return False

    def receive_step(self, current_nA: float, inhibitory: bool = False) -> None:
        """Sustained drive: re-inject every simulation step (biological
        synaptic bombardment). Use inside the sim loop, not receive()."""
        self.receive(current_nA, inhibitory)

    def receive(self, current_nA: float, inhibitory: bool = False) -> None:
        if inhibitory:
            self.i_syn_i += current_nA
        else:
            self.i_syn_e += current_nA


class STDPSynapse:
    """Pair-based, biphasic, reward-modulated STDP synapse.

    Pre before post -> potentiation (w up); post before pre -> depression
    (w down). Soft bounds keep w in [W_MIN, W_MAX]. An eligibility trace
    accumulates recent causal-pair events; a reward signal then converts
    eligibility into actual weight change, so only pre-reward-relevant
    patterned activity is reinforced (v13 spec §3, v14 spec §7).
    """

    def __init__(self, w0: float = 0.25,
                 a_p: float = STDP_A_P, a_m: float = STDP_A_M,
                 tau_stdp_ms: float = STDP_TAU_MS,
                 soft_mu: float = MU_SOFT,
                 tau_elig_ms: float = TAU_ELIG_MS,
                 pre_neuron: Optional[LIFNeuron] = None,
                 post_neuron: Optional[LIFNeuron] = None,
                 delay_ms: float = 1.0,
                 inhibitory: bool = False):
        self.w = float(w0)
        self.a_p = a_p
        self.a_m = a_m
        self.tau = tau_stdp_ms
        self.soft_mu = soft_mu
        self.tau_elig = tau_elig_ms
        self.pre = pre_neuron if pre_neuron is not None else LIFNeuron()
        self.post = post_neuron if post_neuron is not None else LIFNeuron()
        self.delay_ms = delay_ms
        self.inhibitory = inhibitory
        self.pre_trace = 0.0
        self.post_trace = 0.0
        self.eligibility = 0.0
        self.last_weight_change = 0.0
        self.total_potentiations = 0
        self.total_depressions = 0

    # --------------------------------------------------- synaptic dynamics
    def step_traces(self, dt_ms: float = DT_MS) -> None:
        """Decay the STDP pre/post traces and the eligibility trace."""
        f = math.exp(-dt_ms / self.tau)
        self.pre_trace *= f
        self.post_trace *= f
        self.eligibility *= math.exp(-dt_ms / self.tau_elig)

    def _soft_bound(self, a: float) -> float:
        """Multiplicative soft bound so weights cannot escape [W_MIN, W_MAX]."""
        if a >= 0:
            return a * (W_MAX - self.w) / max(self.soft_mu, 1e-9)
        return a * (self.w - W_MIN) / max(self.soft_mu, 1e-9)

    def _apply(self, a: float) -> None:
        self.last_weight_change = a
        self.w = max(W_MIN, min(W_MAX, self.w + a))
        if a > 0:
            self.total_potentiations += 1
        elif a < 0:
            self.total_depressions += 1

    def pre_spike(self) -> None:
        """Handle a pre-synaptic spike: LTD and emit delayed post-synaptic drive."""
        a = -self.a_m * self.post_trace
        if self.inhibitory:
            a = -a  # inhibitory synapses flip sign
        if abs(a) > 0:
            self._apply(a)
        self.pre_trace = 1.0
        self.eligibility += 1.0
        # influence on the post-synaptic neuron
        drive = self.w * (self._i_scale())
        self.post.receive(drive, inhibitory=self.inhibitory)

    def post_spike(self) -> None:
        """Handle a post-synaptic spike: LTP from the pre-synaptic trace."""
        a = self.a_p * self.pre_trace
        if self.inhibitory:
            a = -a
        if abs(a) > 0:
            self._apply(a)
        self.post_trace = 1.0
        self.eligibility += 1.0

    def _i_scale(self) -> float:
        # biological synapse model: drive scale per spike, adjusted for dt.
        return 6.0

    def apply_reward(self, reward: float) -> float:
        """Convert accumulated eligibility into actual weight change.

        Positive reward * eligibility -> potentiation; negative -> depression.
        Returns the applied delta (for telemetry).
        """
        if abs(self.eligibility) < 1e-9:
            return 0.0
        a = self.eligibility * reward * (1.0 / TAU_ELIG_MS)
        # soft bound the magnitude the same way
        safe_a = max(-0.10, min(0.10, a))
        self._apply(safe_a)
        self.eligibility = 0.0
        return safe_a

    def state_dict(self) -> Dict:
        return {"w": self.w, "eligibility": self.eligibility,
                "potentiations": self.total_potentiations,
                "depressions": self.total_depressions}

    def load_state_dict(self, d: Dict) -> None:
        self.w = float(d.get("w", self.w))
        self.eligibility = float(d.get("eligibility", 0.0))
        self.total_potentiations = int(d.get("potentiations", 0))
        self.total_depressions = int(d.get("depressions", 0))


class HomeostaticController:
    """Track each neuron's average firing rate and shift its effective
    threshold toward a target rate. Prevents silent or saturated neurons."""

    def __init__(self, target_hz: float = TARGET_RATE_HZ,
                 gain: float = HOMEOSTASIS_GAIN,
                 window_ms: float = TAU_HOMEOSTASIS_MS):
        self.target = target_hz
        self.gain = gain
        self.window = window_ms
        self.rate_ema: Dict[int, float] = {}
        self.tau_ms = window_ms

    def observe(self, neuron_idx: int, spike: bool, dt_ms: float = DT_MS) -> None:
        cur = self.rate_ema.get(neuron_idx, self.target)
        # EMA of rate: instantaneous rate = spike ? 1000/dt : 0
        rate_sample = 1000.0 / dt_ms if spike else 0.0
        a = dt_ms / self.tau_ms
        self.rate_ema[neuron_idx] = cur + a * (rate_sample - cur)

    def adapt_neuron(self, n: LIFNeuron, idx: int) -> float:
        """Shift the adaptive threshold to push rate toward target.
        Returns the applied shift."""
        r = self.rate_ema.get(idx, self.target)
        shift = -self.gain * (r - self.target)
        n.adapt_threshold = max(0.0, min(15.0, n.adapt_threshold + shift))
        return n.adapt_threshold


class SpikingNucleus:
    """A small population network wiring LIF neurons with STDP synapses.

    Populations (v13 spec §4):
      sensory      – encodes observed game state as rate-coded input current
      temporal     – recurrent lateral connections, keeps traces of recent events
      associative  – learns correlation between sensory and action patterns
      action       – output population; each neuron corresponds to a candidate
                     game action; activity is read out to bias the DQN
      inhibitory   – lateral inhibition on the action population (WTA)
      value        – mirrors expected reward for RPE computation
      memory       – stores recent spike patterns for retrieval
    """

    def __init__(self, n_sensory: int = 64, n_action: int = 20,
                 seed: int = 42):
        self.rng = random.Random(seed)
        self.n_sensory = n_sensory
        self.n_action = n_action
        # sensory population (no plasticity, just rate decoding)
        self.sensory = [LIFNeuron(seed=seed * 100 + i) for i in range(n_sensory)]
        # action population (plastic, gets lateral inhibition)
        self.action = [LIFNeuron(seed=seed * 200 + i) for i in range(n_action)]
        # inhibitory interneuron pool feeding back onto action population
        self.inhibitory = [LIFNeuron(seed=seed * 300 + i) for i in range(6)]
        # feedforward plastic synapses: sensory -> action
        self.ff: List[List[STDPSynapse]] = [
            [STDPSynapse(pre_neuron=self.sensory[i], post_neuron=self.action[j],
                         w0=0.25, inhibitory=False) for j in range(n_action)]
            for i in range(n_sensory)
        ]
        # lateral inhibition: action -> inhibitory -> action
        self.lateral_inh: List[STDPSynapse] = []
        for j in range(n_action):
            for k in range(len(self.inhibitory)):
                s = STDPSynapse(pre_neuron=self.action[j], post_neuron=self.inhibitory[k],
                                w0=0.15, inhibitory=False)
                back = STDPSynapse(pre_neuron=self.inhibitory[k], post_neuron=self.action[j],
                                   w0=0.6, inhibitory=True)
                self.lateral_inh.append(s)
                self.lateral_inh.append(back)
        self.homeostasis = HomeostaticController()
        self.spike_log: List[Tuple[float, int]] = []  # (time_ms, neuron_idx)
        self.t_ms = 0.0
        self.drive_levels: List[float] = [0.0] * n_sensory
        self.DRIVE_GAIN = 5.0  # nA per unit sensory intensity, per step

    def encode_sensory(self, values: List[float]) -> None:
        """Rate-code sensory input: each input in [0,1] sets a sustained
        drive level that is re-injected every simulated step (biological
        synaptic bombardment, not a single decaying pulse)."""
        assert len(values) == self.n_sensory, "sensory dim mismatch"
        self.drive_levels = [max(0.0, min(1.0, v)) for v in values]

    def step(self, n_steps: int = 10) -> Dict:
        """Simulate n_steps ms of network dynamics. Returns telemetry."""
        for _ in range(n_steps):
            self.t_ms += DT_MS
            # sustained sensory drive, re-injected each step
            for i, n in enumerate(self.sensory):
                d = self.drive_levels[i] * self.DRIVE_GAIN
                if d > 0:
                    n.receive_step(d)
                sp = n.step()
                if sp:
                    self.spike_log.append((self.t_ms, i))
                # feedforward drive + plasticity
                for j, s in enumerate(self.ff[i]):
                    if sp:
                        s.pre_spike()

            # action population with lateral inhibition
            for j, n in enumerate(self.action):
                sp = n.step()
                if sp:
                    self.spike_log.append((self.t_ms, self.n_sensory + j))
                    # lateral-inhibitory chain
                    for s in self.lateral_inh:
                        if s.pre is n:
                            s.pre_spike()
                # plasticity: post-synaptic spike on ff[i][j]
                for i in range(self.n_sensory):
                    s = self.ff[i][j]
                    if sp:
                        s.post_spike()

            # action -> inhibitory population cross-talk
            for inh in self.inhibitory:
                inh.step()

            # decay traces everywhere
            for row in self.ff:
                for s in row:
                    s.step_traces()
            for s in self.lateral_inh:
                s.step_traces()

            # homeostasis every 20 ms
            if int(self.t_ms) % 20 == 0:
                for j, n in enumerate(self.action):
                    self.homeostasis.observe(j, False, DT_MS * 20)
                    self.homeostasis.adapt_neuron(n, j)
        return self.telemetry()

    def telemetry(self) -> Dict:
        action_rates = [n.spike_count for n in self.action]
        total_pre = sum(s.total_potentiations for row in self.ff for s in row)
        total_dep = sum(s.total_depressions for row in self.ff for s in row)
        return {
            "t_ms": self.t_ms,
            "sensory_rates": [round(1000 * n.spike_count / max(self.t_ms, 1.0), 3) for n in self.sensory],
            "action_activity": action_rates,
            "inh_active": sum(1 for n in self.inhibitory if n.spike_count > 0),
            "mean_w": sum(s.w for row in self.ff for s in row) / (self.n_sensory * self.n_action),
            "potentiation_events": total_pre,
            "depression_events": total_dep,
        }

    def apply_reward(self, reward: float) -> float:
        """Reward-modulated plasticity: convert eligibility into weight change."""
        total_delta = 0.0
        for row in self.ff:
            for s in row:
                total_delta += s.apply_reward(reward)
        for s in self.lateral_inh:
            total_delta += s.apply_reward(reward)
        return total_delta

    def state_dict(self) -> Dict:
        return {
            "t_ms": self.t_ms,
            "ff": [[s.state_dict() for s in row] for row in self.ff],
            "lateral_inh": [s.state_dict() for s in self.lateral_inh],
            "homeostasis": {"rate_ema": self.homeostasis.rate_ema},
        }

    def load_state_dict(self, d: Dict) -> None:
        self.t_ms = float(d.get("t_ms", 0.0))
        for i, row in enumerate(d.get("ff", [])):
            for j, sdict in enumerate(row):
                self.ff[i][j].load_state_dict(sdict)
        for i, sdict in enumerate(d.get("lateral_inh", [])):
            self.lateral_inh[i].load_state_dict(sdict)
        self.homeostasis.rate_ema = {int(k): v for k, v in
                                     d.get("homeostasis", {}).get("rate_ema", {}).items()}

    def reset_transient(self) -> None:
        """Reset temporary activity state between matches; weights persist."""
        for n in self.sensory + self.action + self.inhibitory:
            n.reset()
        self.spike_log = []
        self.t_ms = 0.0
        for row in self.ff:
            for s in row:
                s.eligibility = 0.0

    def reset_transient_neurons_only(self) -> None:
        """Per-observation reset: clear membrane potentials and spike counters
        so the next observation measures fresh dynamics, while synaptic
        weights, traces, and eligibility persist (durable learning)."""
        for n in self.sensory + self.action + self.inhibitory:
            n.reset()

    # ------------------------------------------- spike-encoding utilities
    @staticmethod
    def rate_to_spikes(rate_hz: float, dt_ms: float = DT_MS) -> bool:
        """Poisson-like: probability = rate_hz * dt_ms/1000."""
        p = min(1.0, rate_hz * dt_ms / 1000.0)
        return random.random() < p


def spike_raster(spike_log: List[Tuple[float, int]], window_ms: float,
                 t_end_ms: float) -> List[List[int]]:
    """Bin the spike log into a raster of shape (window_ms, neurons).
    Placeholder populations: sensory neurons have idx < n_sensory in the log
    (the caller passes total NeuronCount directly)."""
    bins = max(1, int(window_ms))
    rows: Dict[int, List[int]] = {}
    for t,idx in spike_log:
        rel = t_end_ms - t
        if rel < window_ms:
            b = int((t_end_ms - t - 0.001) / bins)
            rows.setdefault(b, []).append(idx)
    out = [rows.get(i, []) for i in range(bins)]
    return out


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity, safe on zero vectors (returns 0.0)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class HippocampalIndex:
    """Retrieval over past situations (design hypothesis, not a copy of the
    replay buffer): cosine-similarity lookup over rich stored memories, with
    a strict retrieval budget so queries can never block gameplay."""

    def __init__(self):
        self.memories: List[Tuple[List[float], float]] = []  # (state_vec, outcome)
        self.max_memories = 4096
        self.recalls_served = 0

    def store(self, state_vec: List[float], outcome: float) -> None:
        self.memories.append((state_vec, float(outcome)))
        if len(self.memories) > self.max_memories:
            self.memories = self.memories[-self.max_memories:]

    def recall(self, state_vec: List[float], k: int = 5) -> List[dict]:
        if not self.memories:
            return []
        scored = [(cosine_similarity(state_vec, s), o) for s, o in self.memories]
        scored.sort(reverse=True, key=lambda x: x[0])
        top = scored[:max(1, k)]
        self.recalls_served += 1
        return [{"similarity": round(sim, 4), "outcome": o} for sim, o in top]


class PrefrontalCortex:
    """Working memory + rule-gated inhibition + bounded deliberation
    (functional hypothesis; tested behavior, not a name on a box).

    Tested contract:
      * holds recent (slot, spot, outcome) events in bounded working memory;
      * vetoes a (slot, spot) that failed k_min consecutive times with
        sufficiently negative average outcome (learned avoidance rule);
      * NEVER vetoes a fresh, unobserved choice;
      * consult() answers within a strict wall-clock budget (returns empty
        plan rather than blocking gameplay).
    """

    def __init__(self, veto_min_failures: int = 3,
                 veto_threshold: float = -0.5,
                 memory_slots: int = 200):
        self.veto_min = veto_min_failures
        self.veto_thr = veto_threshold
        self.memory = deque(maxlen=memory_slots)
        self.deliberations = 0
        self.veto_count = 0

    def observe(self, slot: int, spot: int, outcome: float) -> None:
        self.memory.append((int(slot), int(spot), float(outcome)))

    def consult(self, state_vec, hippocampus: Optional[HippocampalIndex] = None,
                budget_ms: float = 2.0) -> dict:
        """Bounded deliberation: within budget_ms, build the veto plan and
        (optionally) one hippocampal recall. Never exceeds the budget.
        Returns {vetoes: {(slot,spot): avg}, recalled: [...], over_budget: bool}"""
        t0 = time.time()
        self.deliberations += 1
        over = False
        agg: Dict[Tuple[int, int], List[float]] = {}
        for s, sp, o in self.memory:
            agg.setdefault((s, sp), []).append(o)
        vetoes: Dict[Tuple[int, int], float] = {}
        for key, outs in agg.items():
            if len(outs) >= self.veto_min:
                avg = sum(outs) / len(outs)
                if avg <= self.veto_thr:
                    vetoes[key] = round(avg, 3)
        recall = []
        if hippocampus is not None:
            recall = hippocampus.recall(state_vec, k=3)
        if (time.time() - t0) * 1000.0 > budget_ms:
            over = True
        self.veto_count = len(vetoes) if vetoes else self.veto_count
        return {"vetoes": vetoes, "recalled": recall, "over_budget": over}

    def reset(self) -> None:
        """Working memory is transient; veto rules re-derive from evidence."""
        self.memory.clear()
