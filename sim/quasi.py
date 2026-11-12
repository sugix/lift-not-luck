"""Chapter 12: natural experiments.

* Difference-in-differences and synthetic control: Parlo launches Offline
  Mode in Brazil only (a partner deal), in week 26 of the year. Every
  country shares the same seasons; each also has its own growth trend, and
  Brazil was already growing faster than most.
* Regression discontinuity: Juno's answers are scored by an LLM judge, and
  any answer scoring below 0.5 is rewritten by a larger model. Answers just
  below and just above the line are nearly identical, so comparing them
  isolates the rewrite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --------------------------------------------------------------------------
# Panel of countries
# --------------------------------------------------------------------------

COUNTRIES = ["Brazil", "Mexico", "India", "Indonesia", "Turkey", "Poland", "Spain",
             "Italy", "France", "Germany", "Japan", "Korea", "UK", "Canada"]
# Weekly growth in the share of learners active (points per week).
TRENDS = {"Brazil": 0.10, "Mexico": 0.12, "India": 0.08, "Indonesia": 0.06, "Turkey": 0.04,
          "Poland": 0.0, "Spain": -0.02, "Italy": -0.03, "France": -0.03, "Germany": -0.05,
          "Japan": 0.02, "Korea": 0.03, "UK": -0.03, "Canada": -0.01}
LAUNCH_WEEK = 26
OFFLINE_EFFECT = 1.5   # points of weekly active share


def season(week):
    """New-year resolutions, a summer slump, a September return."""
    w = np.asarray(week)
    return 2.0 * np.exp(-w / 4) - 2.5 * np.exp(-((w - 30) / 6) ** 2) + 1.0 * np.exp(-((w - 37) / 3) ** 2)


def country_panel(weeks=52, seed=5):
    """Weekly active share (%) by country. Brazil gets Offline Mode from LAUNCH_WEEK."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in COUNTRIES:
        level = rng.normal(40, 4)
        for w in range(weeks):
            treated = c == "Brazil" and w >= LAUNCH_WEEK
            y = level + TRENDS[c] * w + season(w) + OFFLINE_EFFECT * treated + rng.normal(0, 0.25)
            rows.append({"country": c, "week": w, "active_share": y,
                         "brazil": int(c == "Brazil"), "post": int(w >= LAUNCH_WEEK)})
    return pd.DataFrame(rows)


def _wide(panel, pre_weeks, demean):
    wide = panel.pivot(index="week", columns="country", values="active_share")
    if demean:   # match shapes, not levels: subtract each country's pre-launch mean
        wide = wide - wide.loc[: pre_weeks - 1].mean()
    return wide


def synth_weights(panel, treated="Brazil", pre_weeks=LAUNCH_WEEK, demean=True):
    """Non-negative weights summing to 1 that make the donor countries'
    pre-launch path match the treated country's as closely as possible."""
    wide = _wide(panel, pre_weeks, demean)
    donors = [c for c in wide.columns if c != treated]
    Y1 = wide.loc[: pre_weeks - 1, treated].values
    Y0 = wide.loc[: pre_weeks - 1, donors].values
    k = len(donors)
    res = minimize(lambda w: ((Y1 - Y0 @ w) ** 2).sum(), np.full(k, 1 / k),
                   bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-10})
    return pd.Series(res.x, index=donors)


def synth_path(panel, treated="Brazil", pre_weeks=LAUNCH_WEEK, demean=True):
    """Treated country's path, its synthetic twin, and the weights."""
    wide = _wide(panel, pre_weeks, demean)
    w = synth_weights(panel, treated, pre_weeks, demean)
    return wide[treated], wide[w.index] @ w, w


# --------------------------------------------------------------------------
# Regression discontinuity: Juno's rewrite threshold
# --------------------------------------------------------------------------

CUTOFF = 0.5
REWRITE_EFFECT = 0.06   # completion points added by rewriting an answer


def rewrite_log(n=120_000, seed=3):
    """Lessons with the judge's score of Juno's first answer, whether it was
    rewritten (score below the cutoff), the lesson's difficulty and whether
    the learner completed the lesson."""
    rng = np.random.default_rng(seed)
    difficulty = rng.beta(2, 2, n)
    score = np.clip(rng.beta(5, 3, n) - 0.15 * (difficulty - 0.5), 0, 1)
    rewritten = (score < CUTOFF).astype(int)
    p = 0.35 + 0.5 * score - 0.2 * difficulty + REWRITE_EFFECT * rewritten
    completed = (rng.random(n) < np.clip(p, 0, 1)).astype(int)
    return pd.DataFrame({"judge_score": score, "rewritten": rewritten,
                         "difficulty": difficulty, "completed": completed})
