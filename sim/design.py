"""Chapter 2: designing the test.

Three design questions, each answered by simulating thousands of tests:

* How many learners? Power and minimum detectable effect (MDE) for the
  Streak Shield retention test from Chapter 1.
* What is the unit? Juno's completion rate is measured per lesson, but the
  coin is flipped per learner. Treating lessons as independent understates
  the noise; the delta method gets it right.
* Is "no significant harm" the same as "no harm"? A guardrail needs its own
  power, which a non-inferiority test makes explicit.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from . import foundations as F

# Chapter 1's world: day-30 retention without the shield, and the true lift.
_L = F.learners()
BASELINE = float(_L.p0.mean())
TRUE_LIFT = float((_L.p1 - _L.p0).mean())

NEW_LEARNERS_PER_DAY = 2_000   # learners entering the test each day, both arms together


def z_crit(alpha=0.05, power=0.8, sides=2):
    return stats.norm.ppf(1 - alpha / sides) + stats.norm.ppf(power)


def n_per_arm(p, delta, alpha=0.05, power=0.8):
    """Learners per arm to detect an absolute lift `delta` on a rate `p`."""
    return int(np.ceil(2 * p * (1 - p) * z_crit(alpha, power) ** 2 / delta ** 2))


def mde(p, n, alpha=0.05, power=0.8):
    """Smallest absolute lift detectable with `n` learners per arm."""
    return z_crit(alpha, power) * np.sqrt(2 * p * (1 - p) / n)


def simulate_power(n, p=BASELINE, lift=TRUE_LIFT, reps=4000, alpha=0.05, seed=0):
    """Share of `reps` simulated two-arm tests that come out significant."""
    rng = np.random.default_rng(seed)
    c = rng.binomial(n, p, reps) / n
    t = rng.binomial(n, p + lift, reps) / n
    pooled = (c + t) / 2
    z = (t - c) / np.sqrt(2 * pooled * (1 - pooled) / n)
    return float((np.abs(z) > stats.norm.ppf(1 - alpha / 2)).mean()), t - c


# --------------------------------------------------------------------------
# Juno: a per-lesson metric in a per-learner test
# --------------------------------------------------------------------------

# Juno's completion with its current prompt, and a trimmed system prompt that cuts
# token cost by 30% but quietly costs one point of completion.
JUNO_COMPLETION = {"current prompt": 0.78, "short prompt": 0.77}
LEARNER_SPREAD = 4.0   # Beta concentration: smaller = learners differ more
MEAN_LESSONS = 5.0     # lessons per learner during the test


def juno_learners(n, rate, reps=1, seed=0):
    """Per-learner lesson and completion counts, shape (reps, n).

    Each learner has their own completion propensity (some find Juno easy,
    some don't) and does a varying number of lessons, heavy users many.
    """
    rng = np.random.default_rng(seed)
    a, b = rate * LEARNER_SPREAD, (1 - rate) * LEARNER_SPREAD
    p = rng.beta(a, b, (reps, n))
    lessons = 1 + rng.negative_binomial(2, 2 / (2 + MEAN_LESSONS - 1), (reps, n))
    done = rng.binomial(lessons, p)
    return lessons.astype(float), done.astype(float)


def naive_ratio(lessons, done):
    """Completion rate and SE treating every lesson as independent."""
    x, y = lessons.sum(-1), done.sum(-1)
    r = y / x
    return r, np.sqrt(r * (1 - r) / x)


def delta_ratio(lessons, done):
    """Completion rate and delta-method SE, treating the learner as the unit.

    r = mean(done) / mean(lessons). Linearising around the means gives
    Var(r) ~ Var(done - r * lessons) / (n * mean(lessons)^2).
    """
    n = lessons.shape[-1]
    mx = lessons.mean(-1)
    r = done.sum(-1) / lessons.sum(-1)
    resid = done - r[..., None] * lessons
    return r, np.sqrt(resid.var(-1, ddof=1) / n) / mx


def aa_false_positive(n=2_000, reps=2_000, seed=0):
    """A/A tests on Juno's completion rate: how often each SE cries wolf."""
    lx, ly = juno_learners(n, JUNO_COMPLETION["current prompt"], reps, seed)
    rx, ry = juno_learners(n, JUNO_COMPLETION["current prompt"], reps, seed + 1)
    out = {}
    for name, fn in [("lessons as independent", naive_ratio), ("delta method (per learner)", delta_ratio)]:
        a, sa = fn(lx, ly)
        b, sb = fn(rx, ry)
        z = (a - b) / np.sqrt(sa ** 2 + sb ** 2)
        out[name] = float((np.abs(z) > 1.96).mean())
    return out


def guardrail_test(n, margin=0.02, reps=1, seed=0):
    """Current prompt (control) vs short prompt (treatment), per learner.

    Returns arrays over `reps` simulated tests: the observed difference
    (short - current), its delta-method 95% CI, the two-sided p-value for
    "any difference" and the one-sided p-value for the non-inferiority
    question "is the short prompt worse by more than `margin`?".
    """
    cx, cy = juno_learners(n, JUNO_COMPLETION["current prompt"], reps, seed)
    tx, ty = juno_learners(n, JUNO_COMPLETION["short prompt"], reps, seed + 1)
    rc, sc = delta_ratio(cx, cy)
    rt, st = delta_ratio(tx, ty)
    d, se = rt - rc, np.sqrt(sc ** 2 + st ** 2)
    return {
        "difference": d, "ci_lo": d - 1.96 * se, "ci_hi": d + 1.96 * se,
        "p_any_difference": 2 * stats.norm.sf(np.abs(d) / se),
        "p_non_inferior": stats.norm.sf((d + margin) / se),
    }


def per_learner_sd(rate, n=200_000, seed=9):
    """SD of the per-learner linearised residual, divided by mean lessons:
    the sigma to plug into a sample-size formula for the ratio metric."""
    x, y = juno_learners(n, rate, 1, seed)
    r = y.sum() / x.sum()
    return float((y - r * x).std() / x.mean())
