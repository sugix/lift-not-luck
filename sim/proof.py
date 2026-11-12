"""Chapter 10: proving a bandit worked.

Two questions, two tools:

* "How good is each message?" A bandit's own per-arm averages are biased,
  because it stops sending messages that got unlucky early. Inverse
  propensity weighting (IPW) with logged propensities fixes the bias.
* "How much did the bandit add?" Compare the bandit against a randomly held
  out group that keeps the old rule. Before/after comparisons get fooled by
  anything else that changes over time.

The world is Chapter 7's four reminders, with daily batches (Chapter 8) and an
exploration floor (Chapter 9: never explore at 0%).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import parlo

NAMES = list(parlo.REMINDERS)
RATES = np.array(list(parlo.REMINDERS.values()))
K = len(RATES)


def thompson_probs(wins, sends, rng, floor=0.05, draws=2000):
    """P(Thompson picks each arm), estimated by Monte Carlo, mixed with a
    uniform exploration floor. Shapes: wins/sends (runs, K) -> (runs, K)."""
    samples = rng.beta(1 + wins[:, None, :], 1 + sends[:, None, :] - wins[:, None, :],
                       size=(wins.shape[0], draws, K))
    p = (samples.argmax(axis=-1)[..., None] == np.arange(K)).mean(axis=1)
    return (1 - floor) * p + floor / K


@dataclass
class Logged:
    arm: np.ndarray         # (runs, n) message sent
    reward: np.ndarray      # (runs, n) lesson or not
    propensity: np.ndarray  # (runs, n, K) P(each arm) at decision time
    days: int
    per_day: int


def run_logged(days=20, per_day=500, runs=500, floor=0.05, seed=0) -> Logged:
    """Batched Thompson with a floor, logging every propensity."""
    rng = np.random.default_rng(seed)
    wins = np.zeros((runs, K))
    sends = np.zeros((runs, K))
    arms, rewards, props = [], [], []
    rows = np.arange(runs)[:, None]
    for _ in range(days):
        p = thompson_probs(wins, sends, rng, floor)                       # (runs, K)
        u = rng.random((runs, per_day))[..., None]
        arm = (u > np.cumsum(p, axis=1)[:, None, :]).sum(axis=-1)         # (runs, per_day)
        y = rng.random((runs, per_day)) < RATES[arm]
        np.add.at(sends, (rows, arm), 1)
        np.add.at(wins, (rows, arm), y)
        arms.append(arm)
        rewards.append(y)
        props.append(np.broadcast_to(p[:, None, :], (runs, per_day, K)))
    return Logged(np.concatenate(arms, 1), np.concatenate(rewards, 1).astype(float),
                  np.concatenate(props, 1), days, per_day)


def naive_means(log: Logged):
    """Per-arm sample mean and Wald 95% CI, as a dashboard would show them."""
    onehot = log.arm[..., None] == np.arange(K)
    n = onehot.sum(axis=1)
    m = (onehot * log.reward[..., None]).sum(axis=1) / np.maximum(n, 1)
    se = np.sqrt(m * (1 - m) / np.maximum(n, 1))
    return m, m - 1.96 * se, m + 1.96 * se, n


def ipw_means(log: Logged):
    """Per-arm IPW mean: every send counts, each weighted by 1 / P(that arm)."""
    onehot = log.arm[..., None] == np.arange(K)
    terms = onehot * log.reward[..., None] / log.propensity                # (runs, n, K)
    m = terms.mean(axis=1)
    se = terms.std(axis=1) / np.sqrt(terms.shape[1])
    return m, m - 1.96 * se, m + 1.96 * se


# --------------------------------------------------------------------------
# Holdouts
# --------------------------------------------------------------------------

OLD_RULE = NAMES.index("Plain nudge")  # what Parlo sent before the bandit


def summer_slump(day: int, days: int, depth: float = 0.25) -> float:
    """Everyone studies less as the summer goes on: a multiplier on all rates."""
    return 1 - depth * day / max(days - 1, 1)


@dataclass
class HoldoutRun:
    bandit_sends: np.ndarray    # (runs, days)
    bandit_lessons: np.ndarray
    holdout_sends: np.ndarray
    holdout_lessons: np.ndarray
    before_rate: np.ndarray     # (runs,) old rule's rate in the week before launch


def run_with_holdout(days=30, per_day=1000, runs=300, holdout=0.1, floor=0.05,
                     slump=0.25, seed=0) -> HoldoutRun:
    """A week of the old rule for everyone, then launch: a random `holdout`
    share keeps the old rule, the rest gets the bandit. All rates drift down
    with a summer slump that has nothing to do with the bandit."""
    rng = np.random.default_rng(seed)
    total = days + 7
    wins = np.zeros((runs, K))
    sends = np.zeros((runs, K))
    rows = np.arange(runs)[:, None]
    out = {k: np.zeros((runs, days)) for k in ("bs", "bl", "hs", "hl")}
    before = np.zeros(runs)

    for d in range(total):
        rates = RATES * summer_slump(d, total, slump)
        if d < 7:  # before launch: everyone gets the old rule
            y = rng.random((runs, per_day)) < rates[OLD_RULE]
            before += y.sum(axis=1)
            continue
        t = d - 7
        # no learner ids in this sim, so a per-send draw stands in for a sticky learner-level holdout
        in_holdout = rng.random((runs, per_day)) < holdout
        p = thompson_probs(wins, sends, rng, floor)
        u = rng.random((runs, per_day))[..., None]
        arm = (u > np.cumsum(p, axis=1)[:, None, :]).sum(axis=-1)
        arm = np.where(in_holdout, OLD_RULE, arm)
        y = rng.random((runs, per_day)) < rates[arm]
        b = ~in_holdout
        np.add.at(sends, (rows, arm), b)
        np.add.at(wins, (rows, arm), y & b)
        out["bs"][:, t] = b.sum(axis=1)
        out["bl"][:, t] = (y & b).sum(axis=1)
        out["hs"][:, t] = in_holdout.sum(axis=1)
        out["hl"][:, t] = (y & in_holdout).sum(axis=1)
    return HoldoutRun(out["bs"], out["bl"], out["hs"], out["hl"], before / (7 * per_day))
