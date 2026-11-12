"""Chapter 14: who benefits?

* Premium discount: a randomised test of 30% off Parlo Premium. Some
  learners buy only with the discount (persuadable), some buy anyway (the
  discount just costs money), some never buy, and a few are put off.
  Outcome: revenue per learner.
* Juno's proactive review offer: a randomised test of Juno offering a short
  review session. It helps learners who are struggling or drifting, mildly
  annoys confident regulars, and costs tokens. Outcome: day-30 retention.

Both are randomised 50/50, so the propensity is known (0.5) and the god
view knows every learner's true effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


FULL_PRICE, DISCOUNT_PRICE = 60.0, 42.0


def discount_test(n=80_000, seed=0):
    rng = np.random.default_rng(seed)
    lessons = rng.poisson(rng.gamma(2.0, 2.5, n))
    active = np.minimum(7, rng.poisson(1 + 0.5 * lessons))
    tier = rng.choice([1, 2, 3], n, p=[0.35, 0.4, 0.25])           # 1 = high-income market
    trial_over = (rng.random(n) < 0.5).astype(int)
    engaged = (lessons - 5) / 4
    base = -3.0 + 0.9 * engaged + 0.6 * (tier == 1) - 0.5 * (tier == 3) + 0.4 * trial_over
    # The discount helps price-sensitive, moderately engaged learners most,
    # does little for disengaged ones, and slightly puts off keen high-tier ones.
    tau = (0.9 * (tier >= 2) + 0.5 * trial_over) * np.exp(-0.5 * engaged ** 2) \
        - 0.25 * ((tier == 1) & (engaged > 1))
    t = (rng.random(n) < 0.5).astype(int)
    p0, p1 = _sigmoid(base), _sigmoid(base + tau)
    u = rng.random(n)
    buy0, buy1 = u < p0, u < p1
    rev0, rev1 = FULL_PRICE * buy0, DISCOUNT_PRICE * buy1
    return pd.DataFrame({
        "lessons_wk1": lessons, "active_days_wk1": active, "tier": tier, "trial_over": trial_over,
        "offer": t, "revenue": np.where(t == 1, rev1, rev0),
        "true_uplift": DISCOUNT_PRICE * p1 - FULL_PRICE * p0,
        "realised_uplift": rev1 - rev0,     # what the offer did for this learner, in the god view
        "segment": np.select([buy0 & buy1, ~buy0 & buy1, buy0 & ~buy1], ["buys anyway", "persuadable", "put off"],
                             "never buys"),
    })


REVIEW_COST = 0.03          # dollars of tokens per review session
RETAINED_VALUE = 4.0        # dollars a retained learner is worth over the next month


def review_test(n=80_000, seed=1):
    rng = np.random.default_rng(seed)
    error_rate = rng.beta(2, 5, n)
    days_since = rng.exponential(2.0, n).round(1)
    lessons = rng.poisson(rng.gamma(2.0, 2.5, n))
    level = rng.choice([0, 1, 2], n, p=[0.5, 0.35, 0.15])
    base = -0.6 + 0.12 * lessons - 0.25 * days_since + 0.2 * level
    tau = 1.4 * error_rate + 0.1 * np.minimum(days_since, 6) - 0.35 * (lessons > 8) - 0.2 * (level == 2) - 0.45
    t = (rng.random(n) < 0.5).astype(int)
    p0, p1 = _sigmoid(base), _sigmoid(base + tau)
    u = rng.random(n)
    y = np.where(t == 1, u < p1, u < p0).astype(int)
    return pd.DataFrame({"error_rate": error_rate, "days_since_last": days_since, "lessons_wk1": lessons,
                         "level": level, "offered": t, "retained": y, "true_effect": p1 - p0})


# --------------------------------------------------------------------------
# Meta-learners for heterogeneous effects
# --------------------------------------------------------------------------

def _gbm(seed):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.06, min_samples_leaf=80,
                                         random_state=seed)


def t_learner(X_train, t_train, y_train, X_test, seed=0):
    """Two outcome models, one per arm; the uplift is their difference."""
    m1 = _gbm(seed).fit(X_train[t_train == 1], y_train[t_train == 1])
    m0 = _gbm(seed).fit(X_train[t_train == 0], y_train[t_train == 0])
    return m1.predict(X_test) - m0.predict(X_test)


def dr_learner(X_train, t_train, y_train, X_test, e=0.5, folds=5, seed=0):
    """Doubly robust learner: build an unbiased per-learner pseudo-outcome
    (Chapter 11's AIPW term), then regress it on the features."""
    X_train = np.asarray(X_train)
    n = len(y_train)
    m0, m1 = np.zeros(n), np.zeros(n)
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X_train):
        for arm, store in [(0, m0), (1, m1)]:
            idx = tr[t_train[tr] == arm]
            store[te] = _gbm(seed).fit(X_train[idx], y_train[idx]).predict(X_train[te])
    pseudo = m1 - m0 + t_train * (y_train - m1) / e - (1 - t_train) * (y_train - m0) / (1 - e)
    final = _gbm(seed).fit(X_train, pseudo)
    return final.predict(np.asarray(X_test)), pseudo


def uplift_curve(score, t, y, e=0.5, points=21):
    """Incremental outcome per learner for each share of learners treated, best score first,
    estimated from randomised data with inverse propensity weights."""
    order = np.argsort(-score)
    t, y = t[order], y[order]
    n = len(y)
    contrib = t * y / e - (1 - t) * y / (1 - e)
    shares = np.linspace(0, 1, points)
    gains = [contrib[: int(round(s * n))].sum() / n for s in shares]
    return shares, np.array(gains)


def policy_value(treat, t, y, e=0.5):
    """IPW estimate of the mean outcome if we followed `treat` (0/1 per learner)."""
    match = (treat == t)
    w = np.where(t == 1, 1 / e, 1 / (1 - e))
    v = match * w * y
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))
