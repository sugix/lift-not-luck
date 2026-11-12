"""Chapter 4: reading the result.

* Streak Shield's clean, properly sized test, read three ways: p-value and
  confidence interval, Bayesian chance to win, and expected loss.
* Juno-Lite: a cheaper model that's slightly worse at tutoring. The test
  isn't significant, yet the decision is easy once you price the downside.
* A portfolio of Parlo experiments, to show the winner's curse: significant
  results overstate their lifts, and a prior learned from past tests fixes it.
"""

from __future__ import annotations

import numpy as np

from . import design as D
from . import parlo
from . import trust as T


def streak_clean(seed=7):
    """The Streak Shield test with no crash: one row per assigned learner."""
    return T.streak_test(crash=False, seed=seed)


def beta_posteriors(successes, trials, draws=200_000, seed=0, prior=(1, 1)):
    """Posterior draws of each arm's rate under a Beta prior. Shape (draws, arms)."""
    rng = np.random.default_rng(seed)
    a, b = prior
    s, n = np.asarray(successes), np.asarray(trials)
    return rng.beta(a + s, b + n - s, size=(draws, len(s)))


def chance_and_loss(diff_draws):
    """Chance the treatment wins, and the expected loss (in the units of the
    difference) of shipping it and of keeping control."""
    return {
        "chance to win": float((diff_draws > 0).mean()),
        "expected loss if we ship": float(np.maximum(-diff_draws, 0).mean()),
        "expected loss if we don't": float(np.maximum(diff_draws, 0).mean()),
    }


# --------------------------------------------------------------------------
# Juno-Lite: cheaper, slightly worse
# --------------------------------------------------------------------------

MEDIUM = {"completion": 0.74, "cost": 0.006}    # Juno's current model (Chapter 7)
LITE = {"completion": 0.725, "cost": 0.002}     # a distilled model: cheaper, a bit weaker


def value_per_lesson(model):
    return parlo.LESSON_VALUE * model["completion"] - model["cost"]


def juno_lite_test(n_per_arm=600, seed=4):
    """Learner-randomised test of Lite vs Medium on value per lesson.

    Returns the difference in dollars per lesson (Lite - Medium), its
    delta-method SE, and the same for completion alone.
    """
    out = {}
    for k, (name, m) in enumerate([("medium", MEDIUM), ("lite", LITE)]):
        lessons, done = D.juno_learners(n_per_arm, m["completion"], seed=seed + k)
        lessons, done = lessons[0], done[0]
        value = parlo.LESSON_VALUE * done - m["cost"] * lessons
        for metric, y in [("value", value), ("completion", done)]:
            r = y.sum() / lessons.sum()
            se = (y - r * lessons).std(ddof=1) / np.sqrt(n_per_arm) / lessons.mean()
            out[(name, metric)] = (r, se)
    res = {}
    for metric in ["value", "completion"]:
        (a, sa), (b, sb) = out[("lite", metric)], out[("medium", metric)]
        res[metric] = (a - b, float(np.hypot(sa, sb)))
    return res


# --------------------------------------------------------------------------
# A portfolio of experiments: the winner's curse
# --------------------------------------------------------------------------

def portfolio(n_tests=2_000, true_sd=0.6, se=0.7, seed=11):
    """True lifts (points) of many past experiments, mostly small, and the
    noisy estimate each test reported. Returns (truth, estimate, se)."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(0.2, true_sd, n_tests)
    estimate = truth + rng.normal(0, se, n_tests)
    return truth, estimate, se


def shrink(estimate, se, prior_mean, prior_sd):
    """Posterior mean under a normal prior: pull each estimate toward the
    prior mean, more strongly the noisier it is."""
    w = prior_sd ** 2 / (prior_sd ** 2 + se ** 2)
    return prior_mean + w * (estimate - prior_mean)
