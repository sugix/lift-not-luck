"""Chapter 8: bandits in production at Parlo.

Real reminder systems differ from the textbook bandit in four ways, and this
simulator models each one:

* **Batches.** The bandit's beliefs update once a day, not after every send.
* **Delays.** A push is opened within minutes, but whether the learner finishes
  a lesson is only known a day later.
* **Proxies.** Opens are fast and plentiful, but they are not lessons.
* **Rules.** Some messages only make sense for some learners (no "streak"
  message without a streak), and any message that drives unsubscribes must be
  switched off.

Everything is vectorised over independent runs: arrays are (runs, sends, arms)
within a day, and we loop only over days.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Message:
    name: str
    open_rate: float            # P(push is opened)
    complete_given_open: float  # P(lesson finished within 24h | opened)
    unsub_rate: float           # P(learner turns off notifications)
    needs_streak: bool = False  # only eligible for learners with a streak

    @property
    def lesson_rate(self) -> float:
        return self.open_rate * self.complete_given_open


MESSAGES = [
    Message("Plain nudge", 0.30, 0.300, 0.002),
    Message("Streak at risk", 0.35, 0.300, 0.002, needs_streak=True),
    Message("Juno says hi", 0.32, 0.375, 0.002),
    Message("Progress recap", 0.25, 0.400, 0.002),
    Message("Streak DYING!!", 0.50, 0.180, 0.012, needs_streak=True),
]
NAMES = [m.name for m in MESSAGES]


@dataclass
class Rates:
    """True per-message rates, as arrays of shape (K,)."""
    open: np.ndarray
    complete_given_open: np.ndarray
    unsub: np.ndarray

    @classmethod
    def of(cls, messages=MESSAGES) -> "Rates":
        return cls(np.array([m.open_rate for m in messages]),
                   np.array([m.complete_given_open for m in messages]),
                   np.array([m.unsub_rate for m in messages]))

    @property
    def lesson(self) -> np.ndarray:
        return self.open * self.complete_given_open


# --------------------------------------------------------------------------
# What the bandit has seen so far (per run, per message)
# --------------------------------------------------------------------------


@dataclass
class Seen:
    sends: np.ndarray            # pushes sent
    opens: np.ndarray            # pushes opened (known the same day)
    sends_resolved: np.ndarray   # sends whose lesson outcome is now known
    lessons_resolved: np.ndarray # lessons among those
    opens_resolved: np.ndarray   # opens whose lesson outcome is now known
    unsubs: np.ndarray           # unsubscribes (known the same day)
    active: np.ndarray           # bool: message not switched off by a guardrail

    @classmethod
    def empty(cls, runs: int, k: int) -> "Seen":
        z = lambda: np.zeros((runs, k))
        return cls(z(), z(), z(), z(), z(), z(), np.ones((runs, k), bool))

    def decay(self, factor: float) -> None:
        """Forget old evidence a little each day (for changing worlds)."""
        for name in ("sends", "opens", "sends_resolved", "lessons_resolved",
                     "opens_resolved", "unsubs"):
            setattr(self, name, getattr(self, name) * factor)


# --------------------------------------------------------------------------
# Policies: (seen, eligible[runs, sends, K], rng) -> arm[runs, sends]
# --------------------------------------------------------------------------


def _pick(scores: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    return np.argmax(np.where(eligible, scores, -np.inf), axis=-1)


def _beta(rng, wins, trials, shape):
    """One Beta(1+wins, 1+trials-wins) draw per send: shape (runs, sends, K)."""
    a = np.broadcast_to((1 + wins)[:, None, :], shape)
    b = np.broadcast_to((1 + trials - wins)[:, None, :], shape)
    return rng.beta(a, b)


def fixed_split(seen, eligible, rng):
    return _pick(rng.random(eligible.shape), eligible)


def thompson_opens(seen, eligible, rng):
    """Optimise the fast proxy: P(open)."""
    return _pick(_beta(rng, seen.opens, seen.sends, eligible.shape), eligible)


def thompson_lessons(seen, eligible, rng):
    """Optimise the true outcome directly, using only resolved sends."""
    return _pick(_beta(rng, seen.lessons_resolved, seen.sends_resolved,
                       eligible.shape), eligible)


def thompson_two_stage(seen, eligible, rng):
    """P(lesson) = P(open) x P(lesson | open): fast part from today's opens,
    slow part from opens whose outcome has resolved."""
    p_open = _beta(rng, seen.opens, seen.sends, eligible.shape)
    p_finish = _beta(rng, seen.lessons_resolved, seen.opens_resolved, eligible.shape)
    return _pick(p_open * p_finish, eligible)


def ucb_lessons(seen, eligible, rng):
    """UCB1 on resolved lessons. Deterministic: in a daily batch every
    eligible learner gets the same top message."""
    n = seen.sends_resolved
    t = np.maximum(n.sum(axis=1, keepdims=True), 1.0)   # sends so far, per run
    mean = np.divide(seen.lessons_resolved, n, out=np.zeros_like(n), where=n > 0)
    score = np.where(n > 0, mean + np.sqrt(2 * np.log(t) / np.maximum(n, 1)), np.inf)
    scores = np.broadcast_to(score[:, None, :], eligible.shape)
    tiny = rng.random(eligible.shape) * 1e-12  # break exact ties only
    return _pick(scores + tiny, eligible)


# --------------------------------------------------------------------------
# Guardrail
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UnsubGuardrail:
    """Switch a message off once we're `confidence` sure its unsubscribe rate
    is above `limit`, under a Beta(1, 1) prior."""
    limit: float = 0.006
    confidence: float = 0.95

    def update(self, seen: Seen) -> None:
        p_above = stats.beta.sf(self.limit, 1 + seen.unsubs,
                                1 + seen.sends - seen.unsubs)
        seen.active &= ~(p_above > self.confidence)


# --------------------------------------------------------------------------
# The simulation
# --------------------------------------------------------------------------


@dataclass
class Journey:
    regret: np.ndarray        # (runs, days) cumulative lessons lost vs best eligible
    lessons: np.ndarray       # (runs, days) lessons per day
    share: np.ndarray         # (days, K) share of sends per message, all runs
    unsubs: np.ndarray        # (runs,) total unsubscribes
    switched_off_day: np.ndarray = field(default=None)  # (runs, K) day a message was disabled, -1 if never


def simulate(policy: Callable, days: int = 30, per_day: int = 1_000,
             runs: int = 200, delay: int = 1, streak_share: float = 0.6,
             rates_on_day: Callable[[int], Rates] | None = None,
             guardrail: UnsubGuardrail | None = None,
             discount: float = 1.0, seed: int = 0,
             messages=MESSAGES) -> Journey:
    """Run a reminder bandit for `days`, `per_day` sends a day, in `runs` worlds.

    * Opens and unsubscribes are seen at the end of the same day.
    * Lesson outcomes are seen `delay` days later (0 means the same day).
    * `rates_on_day(d)` lets the true rates change over time.
    """
    rng = np.random.default_rng(seed)
    k = len(messages)
    base = Rates.of(messages)
    needs = np.array([m.needs_streak for m in messages])
    seen = Seen.empty(runs, k)
    pending = []  # (reveal_day, sends_per_arm, lessons_per_arm, opens_per_arm)

    regret = np.zeros((runs, days))
    lessons = np.zeros((runs, days))
    share = np.zeros((days, k))
    unsubs = np.zeros(runs)
    off_day = np.full((runs, k), -1)

    for d in range(days):
        rates = rates_on_day(d) if rates_on_day else base
        has_streak = rng.random((runs, per_day)) < streak_share
        eligible = ~needs[None, None, :] | has_streak[..., None]
        eligible = eligible & seen.active[:, None, :]
        # A learner always has at least one eligible message (non-streak ones).
        arm = policy(seen, eligible, rng)

        opened = rng.random((runs, per_day)) < rates.open[arm]
        finished = opened & (rng.random((runs, per_day)) < rates.complete_given_open[arm])
        unsub = rng.random((runs, per_day)) < rates.unsub[arm]

        onehot = arm[..., None] == np.arange(k)
        n_send = onehot.sum(axis=1)
        n_open = (onehot & opened[..., None]).sum(axis=1)
        n_fin = (onehot & finished[..., None]).sum(axis=1)
        n_unsub = (onehot & unsub[..., None]).sum(axis=1)

        # Regret against the best message each learner was eligible for
        # (ignoring guardrails: switching off a harmful message is a choice).
        elig_all = ~needs[None, None, :] | has_streak[..., None]
        best = np.where(elig_all, rates.lesson, -np.inf).max(axis=-1)
        gap = best - rates.lesson[arm]
        regret[:, d] = gap.sum(axis=1)
        lessons[:, d] = finished.sum(axis=1)
        share[d] = n_send.sum(axis=0) / n_send.sum()
        unsubs += unsub.sum(axis=1)

        # End of day: learn what we can see.
        seen.decay(discount)
        seen.sends += n_send
        seen.opens += n_open
        seen.unsubs += n_unsub
        pending.append((d + delay, n_send, n_fin, n_open))
        still = []
        for reveal, s, f, o in pending:
            if reveal <= d:
                seen.sends_resolved += s
                seen.lessons_resolved += f
                seen.opens_resolved += o
            else:
                still.append((reveal, s, f, o))
        pending = still

        if guardrail is not None:
            before = seen.active.copy()
            guardrail.update(seen)
            off_day[before & ~seen.active] = d

    return Journey(np.cumsum(regret, axis=1), lessons, share, unsubs, off_day)
