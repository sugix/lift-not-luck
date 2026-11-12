"""Chapter 11: when you can't randomise.

* "Learners who use Juno retain better." Juno is free and optional, so
  learners choose it, and keener learners choose it more. We observe
  first-week behaviour that reflects keenness; we never observe keenness.
  The god view knows each learner's two potential outcomes.
* Juno's dictionary tool. Juno decides when to look a word up, and it looks
  up words in harder lessons. Every input to that decision is logged, which
  makes this the friendliest kind of observational data there is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


LEVELS = ["beginner", "intermediate", "advanced"]
COVARIATES = ["lessons_wk1", "active_days_wk1", "level", "on_ios", "evening_learner"]


def juno_users(n=40_000, hidden=0.0, seed=0):
    """Learners, their first-week behaviour, whether they used Juno in
    weeks 2 to 4, and day-30 retention.

    `hidden` is how strongly Juno use depends on keenness *beyond* what the
    first week shows. At 0, the observed covariates are enough to adjust for.
    Also returns `lessons_wk2_4`, a mediator: Juno raises it, and it raises
    retention.
    """
    rng = np.random.default_rng(seed)
    keen = rng.standard_normal(n)
    lessons = rng.poisson(np.exp(1.1 + 0.5 * keen))
    active = rng.binomial(7, _sigmoid(-0.1 + 0.9 * keen))
    level = rng.choice(3, n, p=[0.5, 0.35, 0.15])
    ios = (rng.random(n) < 0.45).astype(int)
    evening = (rng.random(n) < 0.4).astype(int)

    # Who chooses Juno: driven by what the first week shows, plus (optionally)
    # by keenness the first week doesn't show.
    resid = keen - 0.6 * (active - 3.5) / 1.6     # the part of keenness the first week misses
    resid = (resid - resid.mean()) / resid.std()  # one unit = one standard deviation
    p_juno = _sigmoid(-2.4 + 0.25 * lessons + 0.3 * active + 0.5 * (level == 2)
                      - 0.3 * (level == 0) + 0.3 * ios + hidden * resid)
    juno = (rng.random(n) < p_juno).astype(int)

    # Outcomes in both worlds. Juno's true effect: +0.15 on the log-odds of
    # retention directly, plus a little more through 25% more lessons in weeks 2 to 4.
    base = -1.3 + 0.9 * keen + 0.2 * (level == 1) + 0.1 * evening
    mediated = rng.poisson(np.exp(1.5 + 0.4 * keen), size=(2, n)) * np.array([[1.0], [1.25]])
    lp0 = base + 0.03 * mediated[0] + 0.0
    lp1 = base + 0.03 * mediated[1] + 0.15
    u = rng.random(n)
    y0, y1 = (u < _sigmoid(lp0)).astype(int), (u < _sigmoid(lp1)).astype(int)

    return pd.DataFrame({
        "lessons_wk1": lessons, "active_days_wk1": active, "level": level,
        "on_ios": ios, "evening_learner": evening, "juno": juno,
        "lessons_wk2_4": np.where(juno == 1, mediated[1], mediated[0]).round(),
        "retained": np.where(juno == 1, y1, y0),
        "y0": y0, "y1": y1, "propensity_true": p_juno,
    })


def design_matrix(df, cols=COVARIATES):
    """One-hot the level; leave the rest as numbers."""
    X = df[cols].copy()
    if "level" in cols:
        X = pd.get_dummies(X, columns=["level"], drop_first=True, dtype=float)
    return X.astype(float)


def ipw(y, t, e):
    """Inverse propensity weighted ATE (normalised weights)."""
    w1, w0 = t / e, (1 - t) / (1 - e)
    return (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum()


def aipw(y, t, e, m0, m1):
    """Doubly robust (augmented IPW) ATE and its standard error."""
    psi = m1 - m0 + t * (y - m1) / e - (1 - t) * (y - m0) / (1 - e)
    return psi.mean(), psi.std(ddof=1) / np.sqrt(len(psi))


def cross_fit(X, t, y, folds=5, seed=0, clip=0.01):
    """Cross-fitted nuisance models for double machine learning.

    Propensity e(x) = P(T=1 | x) and outcome models m_t(x) = E[Y | T=t, x],
    each predicted on the fold it wasn't trained on.
    """
    X = np.asarray(X)
    e = np.zeros(len(y)); m0 = np.zeros(len(y)); m1 = np.zeros(len(y))
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(X):
        ps = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed)
        ps.fit(X[train], t[train])
        e[test] = ps.predict_proba(X[test])[:, 1]
        for arm, store in [(0, m0), (1, m1)]:
            idx = train[t[train] == arm]
            om = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed)
            om.fit(X[idx], y[idx])
            store[test] = om.predict_proba(X[test])[:, 1]
    return np.clip(e, clip, 1 - clip), m0, m1


def logistic_propensity(X, t):
    model = LogisticRegression(max_iter=2000)
    model.fit(X, t)
    return model.predict_proba(X)[:, 1]


def smd(x, t, w=None):
    """Standardised mean difference between groups, optionally weighted."""
    w = np.ones(len(x)) if w is None else w
    m1 = np.average(x[t == 1], weights=w[t == 1])
    m0 = np.average(x[t == 0], weights=w[t == 0])
    s = np.sqrt((x[t == 1].var() + x[t == 0].var()) / 2)
    return (m1 - m0) / s


# --------------------------------------------------------------------------
# Juno's dictionary tool
# --------------------------------------------------------------------------

def dictionary_tool(n=30_000, seed=0):
    """Lessons where Juno did or didn't call its dictionary tool.

    Juno calls the tool more on hard lessons with rare words. The decision
    depends only on logged inputs (difficulty, rare-word count, task type),
    and Juno samples it with temperature, so the logged probability of the
    call is the true propensity.
    """
    rng = np.random.default_rng(seed)
    difficulty = rng.beta(2, 2, n)
    rare = rng.poisson(1 + 4 * difficulty)
    task = rng.choice(["drill", "reading", "conversation"], n, p=[0.4, 0.35, 0.25])
    p_call = _sigmoid(-2.0 + 3.0 * difficulty + 0.35 * rare + 0.5 * (task == "reading"))
    called = (rng.random(n) < p_call).astype(int)
    lp = 2.2 - 3.2 * difficulty - 0.1 * rare + 0.3 * (task == "drill")
    u = rng.random(n)
    y0 = (u < _sigmoid(lp)).astype(int)
    y1 = (u < _sigmoid(lp + 0.25)).astype(int)
    return pd.DataFrame({"difficulty": difficulty, "rare_words": rare, "task": task,
                         "p_call_logged": p_call, "tool_called": called,
                         "completed": np.where(called == 1, y1, y0), "y0": y0, "y1": y1})
