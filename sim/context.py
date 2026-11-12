"""Chapter 9: contextual bandits, off-policy evaluation and journeys at Parlo.

The world: each learner has a context (level, streak, time of day, phone), and
the best reminder depends on it. Beginners love a friendly note from Juno;
advanced learners respond to a progress recap; long streaks respond to a
streak warning. Phone platform does nothing at all: it's there to show that a
good method copes with features that don't matter.

All rates are known to the simulator and hidden from the policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

MESSAGES = ["Plain nudge", "Streak at risk", "Juno says hi", "Progress recap"]
K = len(MESSAGES)

LEVELS = ["beginner", "intermediate", "advanced"]
STREAKS = ["none", "short", "long"]
TIMES = ["morning", "evening"]
PHONES = ["ios", "android"]
FEATURES = {"level": LEVELS, "streak": STREAKS, "time": TIMES, "phone": PHONES}

# Every combination of features is one segment: 3 * 3 * 2 * 2 = 36.
SEGMENTS = list(product(range(3), range(3), range(2), range(2)))
N_SEG = len(SEGMENTS)
SEG_SHARE = np.full(N_SEG, 1 / N_SEG)  # learners spread evenly across segments


def true_rate(message: int, level: int, streak: int, time: int, phone: int) -> float:
    """True P(lesson within 24h | message, context). Hidden from policies."""
    evening = 0.01 * time
    if message == 0:   # Plain nudge: fine for everyone, great for no one
        p = 0.09
    elif message == 1:  # Streak at risk: only means something with a streak
        p = [0.05, 0.10, 0.15][streak]
    elif message == 2:  # Juno says hi: a friendly face matters most early on
        p = [0.14, 0.11, 0.07][level]
    else:               # Progress recap: advanced learners like seeing progress
        p = [0.07, 0.10, 0.14][level]
    return p + evening


RATES = np.array([[true_rate(m, *s) for m in range(K)] for s in SEGMENTS])  # (36, K)


def features(seg_idx: np.ndarray) -> np.ndarray:
    """One-hot features plus an intercept, shape (..., D)."""
    s = np.asarray(SEGMENTS)[seg_idx]  # (..., 4)
    cols = [np.ones(s.shape[:-1])]
    for j, values in enumerate(FEATURES.values()):
        for v in range(1, len(values)):  # drop the first level of each feature
            cols.append((s[..., j] == v).astype(float))
    return np.stack(cols, axis=-1)


D = features(np.array([0])).shape[-1]  # 1 + 2 + 2 + 1 + 1 = 7


# --------------------------------------------------------------------------
# Policies. Each sees (state, segment index per send, rng) -> arm per send.
# --------------------------------------------------------------------------


@dataclass
class TableState:
    """Wins and sends per (run, segment, arm), for tabular Thompson."""
    wins: np.ndarray
    sends: np.ndarray


@dataclass
class LinearState:
    """Bayesian linear regression per (run, arm): precision A and vector b."""
    A: np.ndarray  # (runs, K, D, D)
    b: np.ndarray  # (runs, K, D)


def thompson_one_size(state: TableState, seg, rng):
    """Ignore context: one Beta per arm, pooled over all segments."""
    w = state.wins.sum(axis=1)[:, None, :]
    n = state.sends.sum(axis=1)[:, None, :]
    shape = seg.shape + (K,)
    return np.argmax(rng.beta(np.broadcast_to(1 + w, shape),
                              np.broadcast_to(1 + n - w, shape)), axis=-1)


def thompson_per_segment(state: TableState, seg, rng):
    """One Beta per (segment, arm): learns each of the 36 segments separately."""
    runs = np.arange(seg.shape[0])[:, None]
    w = state.wins[runs, seg]   # (runs, sends, K)
    n = state.sends[runs, seg]
    return np.argmax(rng.beta(1 + w, 1 + n - w), axis=-1)


def linear_thompson(noise: float = 0.3):
    """Linear Thompson sampling: share strength across segments through features.

    Each arm has weights w with a Gaussian posterior N(A^-1 b, noise^2 A^-1).
    For every send we draw fresh weights and pick the arm with the highest
    predicted lesson rate for that learner's features.
    """
    def policy(state: LinearState, seg, rng):
        runs, sends = seg.shape
        x = features(seg)                                   # (runs, sends, D)
        cov = np.linalg.inv(state.A)                        # (runs, K, D, D)
        mean = np.einsum("rkij,rkj->rki", cov, state.b)     # (runs, K, D)
        chol = np.linalg.cholesky(noise ** 2 * cov)         # (runs, K, D, D)
        z = rng.standard_normal((runs, sends, K, D))
        w = mean[:, None] + np.einsum("rkij,rskj->rski", chol, z)
        scores = np.einsum("rsd,rskd->rsk", x, w)
        return np.argmax(scores, axis=-1)
    policy.__name__ = "linear_thompson"
    return policy


# --------------------------------------------------------------------------
# Simulation with daily batches (Chapter 8 showed why batches matter)
# --------------------------------------------------------------------------


@dataclass
class Run:
    regret: np.ndarray   # (runs, days) cumulative lessons lost vs per-learner best
    lessons: np.ndarray  # (runs, days)
    picks: np.ndarray    # (N_SEG, K) share of sends per segment and arm, last 5 days


def simulate(policy, days: int = 30, per_day: int = 500, runs: int = 100,
             seed: int = 0, linear: bool = False, prior: float = 1.0) -> Run:
    rng = np.random.default_rng(seed)
    if linear:
        state = LinearState(np.broadcast_to(prior * np.eye(D), (runs, K, D, D)).copy(),
                            np.zeros((runs, K, D)))
    else:
        state = TableState(np.zeros((runs, N_SEG, K)), np.zeros((runs, N_SEG, K)))
    best = RATES.max(axis=1)
    regret = np.zeros((runs, days))
    lessons = np.zeros((runs, days))
    picks = np.zeros((N_SEG, K))
    run_idx = np.arange(runs)[:, None]

    for d in range(days):
        seg = rng.choice(N_SEG, size=(runs, per_day), p=SEG_SHARE)
        arm = policy(state, seg, rng)
        p = RATES[seg, arm]
        y = rng.random((runs, per_day)) < p
        regret[:, d] = (best[seg] - p).sum(axis=1)
        lessons[:, d] = y.sum(axis=1)
        if d >= days - 5:
            np.add.at(picks, (seg.ravel(), arm.ravel()), 1)

        if linear:
            x = features(seg)
            onehot = (arm[..., None] == np.arange(K)).astype(float)   # (runs, sends, K)
            state.A += np.einsum("rsk,rsi,rsj->rkij", onehot, x, x)
            state.b += np.einsum("rsk,rsi,rs->rki", onehot, x, y.astype(float))
        else:
            np.add.at(state.sends, (run_idx, seg, arm), 1)
            np.add.at(state.wins, (run_idx, seg, arm), y)

    picks = picks / picks.sum(axis=1, keepdims=True)
    return Run(np.cumsum(regret, axis=1), lessons, picks)


# --------------------------------------------------------------------------
# Off-policy evaluation
# --------------------------------------------------------------------------


@dataclass
class Log:
    seg: np.ndarray        # (n,) segment of each learner
    arm: np.ndarray        # (n,) message sent
    reward: np.ndarray     # (n,) lesson or not
    propensity: np.ndarray # (n,) P(logging policy picks the arm it picked)


def log_from(policy_probs: np.ndarray, n: int, rng) -> Log:
    """Collect n sends from a stochastic logging policy given as (N_SEG, K) probs."""
    seg = rng.choice(N_SEG, size=n, p=SEG_SHARE)
    u = rng.random(n)[:, None]
    arm = (u > np.cumsum(policy_probs[seg], axis=1)).sum(axis=1)
    reward = rng.random(n) < RATES[seg, arm]
    return Log(seg, arm, reward.astype(float), policy_probs[seg, arm])


def true_value(policy_probs: np.ndarray) -> float:
    """Exact expected lesson rate of a policy, from the hidden truth."""
    return float((SEG_SHARE[:, None] * policy_probs * RATES).sum())


def ips(log: Log, target: np.ndarray) -> float:
    """Inverse propensity scoring: reweight logged rewards by target/logging odds."""
    w = target[log.seg, log.arm] / log.propensity
    return float(np.mean(w * log.reward))


def snips(log: Log, target: np.ndarray) -> float:
    """Self-normalised IPS: divide by the sum of weights instead of n."""
    w = target[log.seg, log.arm] / log.propensity
    return float(np.sum(w * log.reward) / np.sum(w))


def direct_model(log: Log, prior_rate: float = 0.1, prior_n: float = 2.0) -> np.ndarray:
    """Reward model: shrunken mean lesson rate per (segment, arm) from the log."""
    wins = np.zeros((N_SEG, K))
    sends = np.zeros((N_SEG, K))
    np.add.at(wins, (log.seg, log.arm), log.reward)
    np.add.at(sends, (log.seg, log.arm), 1)
    return (wins + prior_rate * prior_n) / (sends + prior_n)


def direct_method(log: Log, target: np.ndarray) -> float:
    q = direct_model(log)
    return float((SEG_SHARE[:, None] * target * q).sum())


def doubly_robust(log: Log, target: np.ndarray) -> float:
    """Model prediction for every arm, plus an IPS correction for the model's error."""
    q = direct_model(log)
    model_part = (target[log.seg] * q[log.seg]).sum(axis=1)
    w = target[log.seg, log.arm] / log.propensity
    correction = w * (log.reward - q[log.seg, log.arm])
    return float(np.mean(model_part + correction))


# --------------------------------------------------------------------------
# Journeys: three reminders in a row
# --------------------------------------------------------------------------

ORDERINGS = [(a, b, c) for a in range(K) for b in range(K) for c in range(K)
             if len({a, b, c}) == 3]  # 24 orderings without repeats


def step_rate(step: int, msg: int, prev: int | None) -> float:
    """P(lesson) for one reminder in a 3-step journey (average learner).

    Each message has its own appeal, every extra reminder is a little less
    effective, and Juno's invitation works much better the day after a
    progress recap (the learner has just been reminded how far they've come).
    """
    base = [0.09, 0.10, 0.12, 0.10][msg]
    fatigue = -0.01 * step
    synergy = 0.05 if (prev == 3 and msg == 2) else 0.0
    return base + fatigue + synergy


JOURNEY_VALUE = np.array([
    sum(step_rate(t, m, o[t - 1] if t else None) for t, m in enumerate(o)) for o in ORDERINGS
])
# Lookup tables so the simulations stay vectorised.
STEP_RATE = np.array([[[step_rate(t, m, None if pv == K else pv) for pv in range(K + 1)]
                       for m in range(K)] for t in range(3)])  # (step, msg, prev)
VALUE_OF = np.full((K, K, K), np.nan)
for _o, _v in zip(ORDERINGS, JOURNEY_VALUE):
    VALUE_OF[_o] = _v


def journey_orderings_as_arms(days=30, per_day=200, runs=100, seed=0):
    """Treat each of the 24 orderings as one arm; reward = lessons over 3 steps."""
    rng = np.random.default_rng(seed)
    n_arms = len(ORDERINGS)
    probs = np.array([[step_rate(t, m, o[t - 1] if t else None) for t, m in enumerate(o)]
                      for o in ORDERINGS])  # (24, 3)
    wins = np.zeros((runs, n_arms))
    trials = np.zeros((runs, n_arms))  # step-level trials, 3 per journey
    regret = np.zeros((runs, days))
    run_idx = np.arange(runs)[:, None]
    for d in range(days):
        draws = rng.beta(1 + wins[:, None], 1 + trials[:, None] - wins[:, None],
                         size=(runs, per_day, n_arms))
        arm = draws.argmax(axis=-1)
        outcome = rng.random((runs, per_day, 3)) < probs[arm]
        regret[:, d] = (JOURNEY_VALUE.max() - JOURNEY_VALUE[arm]).sum(axis=1)
        np.add.at(wins, (run_idx, arm), outcome.sum(axis=-1))
        np.add.at(trials, (run_idx, arm), 3)
    return np.cumsum(regret, axis=1)


def journey_step_by_step(days=30, per_day=200, runs=100, seed=0):
    """Choose each step's message given (step, previous message): a small
    contextual bandit with 3 * 5 contexts instead of 24 whole orderings.

    It's myopic: each step maximises its own lesson rate, not the journey's.
    """
    rng = np.random.default_rng(seed)
    n_ctx = 3 * (K + 1)  # step x previous message (or none)
    wins = np.zeros((runs, n_ctx, K))
    sends = np.zeros((runs, n_ctx, K))
    regret = np.zeros((runs, days))
    run_idx = np.arange(runs)[:, None]
    for d in range(days):
        prev = np.full((runs, per_day), K)  # K means "no previous message"
        used = np.zeros((runs, per_day, K), bool)
        chosen = np.zeros((runs, per_day, 3), int)
        for t in range(3):
            ctx = t * (K + 1) + prev
            w = wins[run_idx, ctx]
            n = sends[run_idx, ctx]
            draws = rng.beta(1 + w, 1 + n - w)
            draws[used] = -np.inf  # no repeats within a journey
            msg = draws.argmax(axis=-1)
            p = STEP_RATE[t, msg, prev]
            y = rng.random((runs, per_day)) < p
            np.add.at(sends, (run_idx, ctx, msg), 1)
            np.add.at(wins, (run_idx, ctx, msg), y)
            used[np.arange(runs)[:, None], np.arange(per_day)[None, :], msg] = True
            chosen[..., t] = msg
            prev = msg
        values = VALUE_OF[chosen[..., 0], chosen[..., 1], chosen[..., 2]]
        regret[:, d] = (JOURNEY_VALUE.max() - values).sum(axis=1)
    return np.cumsum(regret, axis=1)
