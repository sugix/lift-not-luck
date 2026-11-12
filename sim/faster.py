"""Chapter 5: faster answers.

* CUPED: subtract the part of each learner's outcome that their pre-test
  behaviour already predicts. Strong for a habit metric like lessons per
  week; modest for a yes/no metric like day-30 retention.
* Peeking: checking a fixed-horizon p-value every day and stopping at the
  first p < 0.05 inflates false positives.
* Always-valid inference: a mixture sequential probability ratio test
  (mSPRT) and its confidence sequence, built from scratch, can be checked
  as often as you like.
* Juno latency: a guardrail watched continuously, with CUPED on each
  learner's pre-test latency (mostly their network), stops a bad rollout fast.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# Weekly Goals: existing learners, lessons in the two weeks after enrolling
# --------------------------------------------------------------------------

GOALS_LIFT = 0.03       # Weekly Goals raises lessons by 3%


def weekly_goals(n, lift=0.0, seed=0, reps=None):
    """Pre-period and test-period lessons for `n` learners.

    Each learner has a persistent habit (their usual lessons per fortnight).
    Both periods are noisy draws around it, so the pre-period predicts the
    test period well, but not perfectly. Shapes (n,) or (reps, n).
    """
    rng = np.random.default_rng(seed)
    shape = (n,) if reps is None else (reps, n)
    habit = np.exp(rng.normal(2.0, 0.75, shape))                # about 10 lessons a fortnight
    pre = rng.poisson(habit * rng.gamma(12, 1 / 12, shape))
    post = rng.poisson(habit * (1 + lift) * rng.gamma(12, 1 / 12, shape))
    return pre.astype(float), post.astype(float)


def cuped(y, x):
    """CUPED-adjusted outcome: y - theta * (x - mean(x)), theta = cov/var.

    Use theta estimated on both arms pooled, so it doesn't depend on the arm.
    """
    theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    return y - theta * (x - x.mean()), theta


# --------------------------------------------------------------------------
# Sequential monitoring
# --------------------------------------------------------------------------

def daily_estimates(days=28, per_day=500, lift=0.0, reps=500, use_cuped=False, seed=0,
                    sampler=None):
    """Run `reps` experiments, enrolling `per_day` learners per arm each day.

    Returns (estimate, se) arrays of shape (reps, days): the difference in
    means (treatment - control) using everyone enrolled so far, with or
    without CUPED. `sampler(n, lift, rng_seed, reps)` returns (pre, post).
    """
    sampler = sampler or weekly_goals
    # running sums per arm: n, x, y, x^2, y^2, xy
    acc = {arm: np.zeros((6, reps)) for arm in ("c", "t")}
    est, se = np.zeros((reps, days)), np.zeros((reps, days))
    for d in range(days):
        for k, (arm, l) in enumerate([("c", 0.0), ("t", lift)]):
            x, y = sampler(per_day, l, seed=seed * 1000 + 2 * d + k, reps=reps)
            a = acc[arm]
            a[0] += per_day
            a[1] += x.sum(1); a[2] += y.sum(1)
            a[3] += (x * x).sum(1); a[4] += (y * y).sum(1); a[5] += (x * y).sum(1)
        n = acc["c"][0] + acc["t"][0]
        sx = acc["c"][1] + acc["t"][1]
        sy = acc["c"][2] + acc["t"][2]
        theta = 0.0
        if use_cuped:
            cov = ((acc["c"][5] + acc["t"][5]) - sx * sy / n) / (n - 1)
            var = ((acc["c"][3] + acc["t"][3]) - sx ** 2 / n) / (n - 1)
            theta = cov / var
        means, varis = {}, {}
        for arm in ("c", "t"):
            a = acc[arm]
            m = a[0]
            mx, my = a[1] / m, a[2] / m
            vx = (a[3] - m * mx ** 2) / (m - 1)
            vy = (a[4] - m * my ** 2) / (m - 1)
            cxy = (a[5] - m * mx * my) / (m - 1)
            means[arm] = my - theta * (mx - sx / n)
            varis[arm] = (vy - 2 * theta * cxy + theta ** 2 * vx) / m
        est[:, d] = means["t"] - means["c"]
        se[:, d] = np.sqrt(varis["t"] + varis["c"])
    return est, se


def naive_p(est, se):
    return 2 * stats.norm.sf(np.abs(est) / se)


def msprt(est, se, tau):
    """Always-valid p-value from a normal-mixture SPRT.

    With the estimate ~ N(theta, V) and a N(0, tau^2) mixture over theta,
    the likelihood ratio against theta = 0 is
        sqrt(V / (V + tau^2)) * exp(tau^2 est^2 / (2 V (V + tau^2))).
    The always-valid p-value is the running minimum of 1 / ratio.
    """
    V = se ** 2
    log_lr = 0.5 * np.log(V / (V + tau ** 2)) + tau ** 2 * est ** 2 / (2 * V * (V + tau ** 2))
    p = np.minimum(1, np.exp(-log_lr))
    return np.minimum.accumulate(p, axis=-1)


def confidence_sequence(est, se, tau, alpha=0.05):
    """Half-width of the always-valid (1 - alpha) interval at each look."""
    V = se ** 2
    return np.sqrt(V * (V + tau ** 2) / tau ** 2 * (np.log((V + tau ** 2) / V) - 2 * np.log(alpha)))


# --------------------------------------------------------------------------
# Juno latency
# --------------------------------------------------------------------------

LATENCY_SLOWDOWN = 0.25   # v2 adds 250 ms per response (an extra retrieval call)
LATENCY_MARGIN = 0.10     # guardrail: no more than 100 ms slower


def juno_latency(n, slowdown=0.0, seed=0, reps=None):
    """Mean response latency (seconds) per learner, before and during the test.

    A learner's latency is mostly their network and device, which persists;
    each period's mean is that plus noise from a handful of responses.
    """
    rng = np.random.default_rng(seed)
    shape = (n,) if reps is None else (reps, n)
    base = rng.lognormal(np.log(1.6), 0.45, shape)          # the learner's usual latency
    pre = base + rng.normal(0, 0.25, shape)
    post = base + slowdown + rng.normal(0, 0.25, shape)
    return pre, post


def streak_with_pre(seed=7):
    """Chapter 4's clean Streak Shield test, with each learner's first-week
    lessons (measured before the shield is offered on day 8)."""
    from . import foundations as F
    from . import readout as R
    S = R.streak_clean(seed=seed)
    S["lessons_wk1"] = F.learners(len(S), seed=seed).lessons_wk1.values
    return S
