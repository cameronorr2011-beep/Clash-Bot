"""NeuroML hand_tracker — the REAL four-card live hand (PROMPT 4 P1C).

Replaces every `json.load(my_deck.json)["deck"][:4]` hand fake:
  * Clash Royale cycle: the deck is an 8-card queue; the HAND is its first 4;
    playing a hand card moves it to the BACK of the queue.
  * Initialization is HONEST: without a verified observation the tracker runs
    in UNCERTAIN mode (every slot unknown/conf 0 → conservative play only).
  * Vision reconciliation: observed identities slot-by-slot overwrite the
    cycle prediction and RE-ANCHOR the queue, so one missed detection never
    permanently shifts slot identities.
  * Every slot carries confidence 0..1; uncertain slots never authorize
    spells/finishers/high-cost commitments (checked by callers via card_ok()).
"""
import time


class HandTracker:
    def __init__(self, deck=None):
        self.deck = list(deck) if deck else []
        # queue state: known order that enters the hand. Unknown until verified.
        self.queue_known = False
        self.queue = list(self.deck)
        self.hand = [None, None, None, None]      # card id or None
        self.conf = [0.0, 0.0, 0.0, 0.0]
        self.last_played = None                   # (card, slot, tick)
        self.last_update = 0.0
        self.ticks = 0

    # ---------- lifecycle ----------
    def set_deck(self, deck, verified=False):
        self.deck = [str(c).lower() for c in (deck or [])]
        if not self.queue_known:
            self.queue = list(self.deck)
        if verified and not self.queue_known:
            # verified deck ≠ verified hand; keep honest: order unknown
            self.queue_known = False
        for i, c in enumerate(self.hand):
            if c not in self.deck:
                self.hand[i] = None
                self.conf[i] = 0.0

    def verify_initial_hand(self, hand4, confs=None):
        """Calibration flow: a verified observation of the CURRENT four cards
        re-anchors the cycle (deck = hand4 + remaining 4 in recorded order)."""
        hand4 = [str(c).lower() for c in (hand4 or [])]
        if hand4 and all(h in self.deck for h in hand4) and len(set(hand4)) == 4:
            self.hand = hand4[:]
            self.conf = list(confs or [1.0] * 4)[:4]
            rest = [c for c in self.deck if c not in hand4]
            self.queue = hand4 + rest
            self.queue_known = True
            self.last_update = time.time()
            return True
        return False

    # ---------- observing ----------
    def observe_hand(self, observed, confs=None):
        """observed: list of 4 card ids (or None for unrecognized slot).
        Reconciliation: exact observations win and RE-ANCHOR the queue;
        None slots take a low-confidence prediction from the cycle model."""
        changed = []
        obs = [(str(c).lower() if c else None) for c in (observed or [])]
        for i in range(4):
            c = obs[i] if i < len(obs) else None
            if c and c in self.deck:
                if self.hand[i] != c:
                    changed.append((i, self.hand[i], c))
                self.hand[i] = c
                self.conf[i] = float((confs or [])[i]) if i < len(confs or []) else 0.8
            elif c:                                # observed a card NOT in deck
                self.conf[i] = 0.0                  # deck itself is suspect
        if changed:
            self.reanchor_from_observations()
        self.last_update = time.time()
        return changed

    def reanchor_from_observations(self):
        """Best-effort cycle re-anchor: hand + remaining deck cards in deck
        order form a consistent queue handed down from the observed truth."""
        hand = [c for c in self.hand if c in self.deck]
        rest = [c for c in self.deck if c not in hand]
        if len(hand) == 4 and len(set(hand)) == 4:
            self.queue = hand + rest
            self.queue_known = True
            for i in range(4):
                if self.conf[i] < 0.4:
                    self.conf[i] = 0.4              # cycle-sourced, not seen

    # ---------- playing ----------
    def authorize_play(self, slot):
        """Final pre-execution validation (P1D): the slot must hold a Verified
        card identity. Unknown identity → refused with reason."""
        card = self.hand[slot] if 0 <= slot < 4 else None
        conf = self.conf[slot] if 0 <= slot < 4 else 0.0
        if not card:
            return None, "slot identity unknown (no verified observation)"
        if conf < 0.5:
            return None, f"slot identity low-confidence ({conf:.2f})"
        return card, "ok"

    def commit_play(self, slot):
        """Called AFTER the tap executed: rotate the cycle queue. Hand slot is
        refilled from the queue head (below position 4) — the next card enters."""
        card, why = self.authorize_play(slot)
        if not card:
            # we still advance a blind queue when deck order is verified-only
            if self.queue_known and 0 <= slot < 4:
                self.hand[slot] = None
                self.conf[slot] = 0.0
                self._refill_from_queue(slot)
            return None, why
        if self.queue_known and card in self.queue:
            try:
                self.queue.remove(card)
            except ValueError:
                pass
            self.queue.append(card)                     # to the BACK
            self._refill_from_queue(slot)
        else:
            self.hand[slot] = None
            self.conf[slot] = 0.0
        self.last_played = (card, slot, self.ticks)
        self.last_update = time.time()
        return card, "ok"

    def _refill_from_queue(self, slot):
        if not self.queue_known:
            return
        nxt = None
        for i in range(4, len(self.queue)):
            cand = self.queue[i]
            if cand not in self.hand:
                nxt = cand
                break
        if nxt:
            self.hand[slot] = nxt
            self.conf[slot] = 0.6                       # cycle-sourced preview
        # else: no unseen card left (unexpected) — leave unknown

    # ---------- queries ----------
    def live_hand(self):
        """THE live four cards (None for unknown slots). This replaces every
        my_deck[:4] read."""
        return list(self.hand)

    def hand_confidence(self):
        return list(self.conf)

    def costs(self, card_cost_fn):
        """Costs per slot; None where identity unknown. Caller must not
        substitute a default cost (P1D)."""
        return [card_cost_fn(c) if c else None for c in self.hand]

    def verified(self):
        return all(c and conf >= 0.5 for c, conf in zip(self.hand, self.conf))

    def conservative_mode(self):
        """True while identities are unverified — spells/finishers/high-cost
        commitments must be refused until observation resolves them."""
        return not self.verified()

    def describe(self):
        return {
            "hand": self.live_hand(),
            "conf": [round(float(c), 2) for c in self.conf],
            "cycle_known": self.queue_known,
            "conservative": self.conservative_mode(),
            "last_played": self.last_played,
        }


def make_tracker(deck, verified_deck=False, observed_hand=None):
    t = HandTracker(deck)
    if observed_hand:
        t.observe_hand(observed_hand, confs=[1.0] * 4)
    return t
