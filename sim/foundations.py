"""Chapter 1: the lift that wasn't.

Parlo ships "Streak Shield", a feature that protects a streak for one missed
day. Learners opt in. We simulate the *god view*: for every learner, both
potential outcomes, retained if they get the feature and retained if they
don't. Real data only ever shows one of the two.

The hidden truth: the feature adds a little retention. But keen learners are
both more likely to opt in and more likely to stay anyway, so a naive
comparison of opted-in vs not opted-in learners looks spectacular.

The Juno twin: a new tutor prompt, "v2", is rolled out first on easy
vocabulary drills. Its success rate looks far higher than v1's, mostly
because of which lessons it was given.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


TRUE_EFFECT_LOGIT = 0.08  # Streak Shield's true effect on the log-odds of retention


def learners(n: int = 20_000, seed: int = 0) -> pd.DataFrame:
    """Learners with a hidden 'keenness' and both potential outcomes.

    Columns:
      keenness      latent motivation (never observed in real life)
      lessons_wk1   lessons in the first week (observed, correlated with keenness)
      p0, p1        P(retained at day 30) without / with Streak Shield
      y0, y1        the two potential outcomes (only one is ever observed)
    """
    rng = np.random.default_rng(seed)
    keen = rng.standard_normal(n)
    lessons_wk1 = rng.poisson(np.exp(1.2 + 0.5 * keen))
    base = -0.6 + 1.1 * keen
    p0 = _sigmoid(base)
    p1 = _sigmoid(base + TRUE_EFFECT_LOGIT)
    u = rng.random(n)            # shared noise: the same learner, two worlds
    return pd.DataFrame({
        "keenness": keen,
        "lessons_wk1": lessons_wk1,
        "p0": p0, "p1": p1,
        "y0": (u < p0).astype(int),
        "y1": (u < p1).astype(int),
    })


def opt_in(df: pd.DataFrame, seed: int = 1) -> pd.DataFrame:
    """Learners choose the feature; keen learners choose it far more often."""
    rng = np.random.default_rng(seed)
    p_opt = _sigmoid(-1.0 + 1.6 * df.keenness)
    out = df.copy()
    out["shield"] = (rng.random(len(df)) < p_opt).astype(int)
    out["retained"] = np.where(out.shield == 1, out.y1, out.y0)
    return out


def randomise(df: pd.DataFrame, seed: int = 2) -> pd.DataFrame:
    """A coin decides who gets the feature."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["shield"] = (rng.random(len(df)) < 0.5).astype(int)
    out["retained"] = np.where(out.shield == 1, out.y1, out.y0)
    return out


# --------------------------------------------------------------------------
# Juno: prompt v2 vs v1
# --------------------------------------------------------------------------

TASKS = {"Vocabulary drill": 0.15, "Grammar explanation": 0.5, "Free conversation": 0.85}
V2_EFFECT_LOGIT = 0.06


def juno_lessons(n: int = 30_000, rollout: str = "drills first", seed: int = 3) -> pd.DataFrame:
    """Lessons with a difficulty, a prompt version and a completion outcome.

    rollout="drills first": v2 was switched on for most drills, few conversations.
    rollout="random": a coin picks the prompt for every lesson.
    """
    rng = np.random.default_rng(seed)
    task = rng.choice(list(TASKS), size=n, p=[0.5, 0.3, 0.2])
    difficulty = np.array([TASKS[t] for t in task])
    if rollout == "random":
        p_v2 = np.full(n, 0.5)
    else:
        p_v2 = np.select([task == "Vocabulary drill", task == "Grammar explanation"], [0.8, 0.4], 0.1)
    v2 = (rng.random(n) < p_v2).astype(int)
    p = _sigmoid(2.4 - 3.0 * difficulty + V2_EFFECT_LOGIT * v2)
    done = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"task": task, "prompt": np.where(v2 == 1, "v2", "v1"), "completed": done})
