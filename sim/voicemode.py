"""Chapter 15's demo experiment: Juno Voice Mode.

Learners in the treatment arm can talk to Juno instead of typing. One row
per learner, with the per-user sums an experimentation pipeline would
compute from event logs.

True effects (relative, treatment vs control):
  lessons per learner      +4%     (decision metric, CUPED on pre-period lessons)
  completion per lesson    +2%     (ratio)
  latency per turn         +6%     (guardrail, margin 10%)
  cost per lesson          +15%    (guardrail, margin 10%: fails)
  minutes studied          +5%     (secondary)
  every other secondary    0
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRUE = {"lessons": 0.04, "completion": 0.02, "latency": 0.06, "cost": 0.15, "minutes": 0.05}
SECONDARY_NULL = ["streak_days", "vocab_reviews", "shares", "ratings_given",
                  "settings_changes", "support_tickets", "notifications_opened"]


def simulate(n=30_000, effects=None, arm=None, crash_share=0.0, seed=0):
    """One row per learner. `arm` can be passed in (e.g. from hash
    assignment); otherwise a fair coin. `effects=None` uses TRUE; pass {} for
    an A/A test. `crash_share` drops that share of Android treatment learners
    from the logs (Chapter 3's bug)."""
    rng = np.random.default_rng(seed)
    eff = TRUE if effects is None else effects
    if arm is None:
        arm = np.where(rng.random(n) < 0.5, "treatment", "control")
    t = (np.asarray(arm) == "treatment").astype(float)
    habit = rng.lognormal(np.log(6), 0.7, n)
    platform = rng.choice(["iOS", "Android", "web"], n, p=[0.45, 0.4, 0.15])
    pre = rng.poisson(habit * rng.gamma(10, 0.1, n))
    lessons = rng.poisson(habit * (1 + eff.get("lessons", 0) * t) * rng.gamma(10, 0.1, n)) + 1
    p_done = np.clip(rng.beta(7, 2.5, n) * (1 + eff.get("completion", 0) * t), 0, 1)
    done = rng.binomial(lessons, p_done)
    turns = rng.poisson(lessons * 6.0) + 1
    lat_user = rng.lognormal(np.log(1.4), 0.3, n) * (1 + eff.get("latency", 0) * t)
    latency_total = lat_user * turns * rng.gamma(20, 0.05, n)
    cost = lessons * 0.006 * (1 + eff.get("cost", 0) * t) * rng.gamma(8, 1 / 8, n)
    minutes = rng.gamma(4, habit * 2.5 / 4) * (1 + eff.get("minutes", 0) * t)
    df = pd.DataFrame({
        "user_id": [f"u{i:06d}" for i in range(n)], "arm": arm, "platform": platform,
        "pre_lessons": pre, "lessons": lessons, "lessons_completed": done,
        "turns": turns, "latency_total_s": latency_total, "cost_usd": cost, "minutes": minutes,
    })
    for k, name in enumerate(SECONDARY_NULL):
        df[name] = rng.poisson(habit * (0.2 + 0.1 * k)) * (1 + eff.get(name, 0) * t)
    if crash_share:
        lost = (df.arm == "treatment") & (df.platform == "Android") & (rng.random(n) < crash_share)
        df = df[~lost].reset_index(drop=True)
    return df
