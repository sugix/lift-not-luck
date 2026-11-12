"""Chapter 6: many metrics, long-run truth.

* Many metrics: Juno v3 is scored on 25 rubric metrics. Only 3 really move.
  Uncorrected p-values find extra "wins"; Benjamini-Hochberg keeps them honest.
* Simpson's paradox in a ramp-up: the treatment share grew while the kind of
  learner arriving changed, so the pooled comparison points the wrong way.
* Novelty: a livelier Juno persona gets a burst of attention that fades.
* Surrogates: predicting day-30 retention from week-one behaviour works for
  a feature that builds a habit, and fails for one that just nags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------------------------------
# Many metrics
# --------------------------------------------------------------------------

RUBRIC = ["accuracy", "clarity", "encouragement", "pacing", "follow-up questions",
          "grammar feedback", "vocabulary level", "cultural notes", "humour", "brevity",
          "patience", "error correction", "examples", "memory of past lessons", "tone",
          "politeness", "structure", "confidence", "relevance", "personalisation",
          "pronunciation tips", "recap quality", "goal tracking", "motivation", "safety"]
REAL_GAINS = {"clarity": 0.12, "examples": 0.12, "recap quality": 0.12}   # in SDs per learner


def rubric_z(n_per_arm=1_500, reps=1, corr=0.3, seed=0):
    """z-scores of v3 - v2 on every rubric metric, shape (reps, 25).

    Metrics share a common 'overall impression' factor, so they're
    positively correlated, as LLM-judge scores usually are.
    """
    rng = np.random.default_rng(seed)
    k = len(RUBRIC)
    cov = np.full((k, k), corr) + (1 - corr) * np.eye(k)
    noise = rng.multivariate_normal(np.zeros(k), cov, size=reps)
    shift = np.array([REAL_GAINS.get(m, 0.0) for m in RUBRIC]) * np.sqrt(n_per_arm / 2)
    return noise + shift


# --------------------------------------------------------------------------
# Simpson's paradox in a ramp-up
# --------------------------------------------------------------------------

RAMP = pd.DataFrame({
    "day": range(1, 8),
    "treatment share": [0.05, 0.10, 0.20, 0.50, 0.50, 0.50, 0.50],
    # launch-week sign-ups came from a marketing push: keener learners
    "baseline retention": [0.45, 0.45, 0.45, 0.30, 0.30, 0.30, 0.30],
})
RAMP_LIFT = 0.01


def ramp_test(per_day=4_000, seed=0):
    """Daily counts and day-30 retention for a test whose split ramped up."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in RAMP.iterrows():
        n_t = int(per_day * r["treatment share"])
        n_c = per_day - n_t
        rows.append({"day": int(r.day), "control learners": n_c, "treatment learners": n_t,
                     "control retained": rng.binomial(n_c, r["baseline retention"]),
                     "treatment retained": rng.binomial(n_t, r["baseline retention"] + RAMP_LIFT)})
    return pd.DataFrame(rows)


def stratified_lift(df):
    """Per-day differences, combined with weights n_c n_t / (n_c + n_t)."""
    nc, nt = df["control learners"], df["treatment learners"]
    d = df["treatment retained"] / nt - df["control retained"] / nc
    w = nc * nt / (nc + nt)
    return float((w * d).sum() / w.sum())


# --------------------------------------------------------------------------
# Novelty
# --------------------------------------------------------------------------

def novelty_lift(day):
    """True lift (%) in daily lessons from 'Coach Juno', by days since launch:
    a 7-point burst of curiosity fading to a lasting 1 point."""
    return 1 + 7 * np.exp(-np.asarray(day) / 4)


def novelty_test(days=28, n_per_arm=40_000, seed=0):
    """Daily lessons per learner in each arm, existing learners, launch on day 0."""
    rng = np.random.default_rng(seed)
    habit = rng.lognormal(np.log(0.8), 0.6, 2 * n_per_arm)
    arm = np.repeat([0, 1], n_per_arm)
    rows = []
    for t in range(days):
        rate = habit * (1 + arm * novelty_lift(t) / 100)
        y = rng.poisson(rate)
        c, tr = y[arm == 0], y[arm == 1]
        se = np.sqrt(c.var() / c.size + tr.var() / tr.size) / c.mean()
        rows.append({"day": t, "lift %": 100 * (tr.mean() / c.mean() - 1), "se %": 100 * se,
                     "true lift %": float(novelty_lift(t))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Surrogates
# --------------------------------------------------------------------------

def surrogate_world(n, engagement_shift=0.0, nag=0.0, seed=0):
    """Learners with week-one behaviour and day-30 retention.

    engagement_shift: a feature that builds a real habit (shifts the latent
        engagement that drives everything).
    nag: a feature that pushes extra quick lessons in week one (more
        notifications) without changing engagement.
    Retention depends on engagement and on days active in week one; the
    number of week-one lessons has no effect of its own.
    """
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n) + engagement_shift
    active = rng.binomial(7, _sigmoid(-0.2 + 1.0 * e))
    lessons = rng.poisson(np.exp(1.0 + 0.5 * e) * (1 + nag))
    retained = (rng.random(n) < _sigmoid(-0.9 + 0.7 * e + 0.15 * (active - 3.5))).astype(int)
    return pd.DataFrame({"active_days_wk1": active, "lessons_wk1": lessons, "retained_d30": retained})


# --------------------------------------------------------------------------
# A quarter of launches, and a holdout that adds them up honestly
# --------------------------------------------------------------------------

def quarter_of_launches(n_tests=40, se=0.7, holdout_se=0.35, seed=8):
    """A quarter's experiments at Parlo: true lifts (% lessons per learner),
    what each test reported, which shipped (significantly positive), and what
    a quarter-long holdout, kept off every launch, measures for the total."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(0.2, 0.6, n_tests)
    reported = truth + rng.normal(0, se, n_tests)
    shipped = reported / se > 1.96
    holdout_measured = truth[shipped].sum() + rng.normal(0, holdout_se)
    return pd.DataFrame({"true_lift": truth, "reported_lift": reported, "shipped": shipped}), holdout_measured
