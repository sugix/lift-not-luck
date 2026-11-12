"""Chapter 3: don't trust it yet.

Two broken pipelines that produce confident, wrong results:

* Streak Shield, properly sized this time (Chapter 2). The shield's new
  onboarding screen crashes on old Android phones before the "exposure"
  event is logged, so some treatment learners silently vanish from the
  analysis table. They're less engaged than average, so the treatment arm
  looks better than it is. A sample ratio mismatch (SRM) check catches it.
* Juno prompt v2. It's slower, so more calls time out and get retried, and
  every attempt is logged as a session. A response cache shared by both arms
  serves v1's answers to v2 learners. Retries distort the per-session
  numbers; the cache quietly shrinks the effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import foundations as F


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------------------------------
# Streak Shield with a crash
# --------------------------------------------------------------------------

CRASH_RATE = 0.3   # share of old-Android treatment learners whose onboarding crashes


def streak_test(n=31_400, crash=True, seed=32):
    """Assignment table for every learner, plus whether they reached the
    exposure log. Returns one row per learner.

    Columns: arm, platform, logged (reached the analysis table), retained.
    Old-Android learners are less keen on average. In the treatment arm, a
    crash stops some of them from ever being logged, and they never get the
    shield either.
    """
    rng = np.random.default_rng(seed)
    L = F.learners(n, seed=seed)
    old = rng.random(n) < _sigmoid(-2.4 - 1.2 * L.keenness)
    arm = np.where(rng.random(n) < 0.5, "shield", "control")
    crashed = crash & (arm == "shield") & old & (rng.random(n) < CRASH_RATE)
    got_shield = (arm == "shield") & ~crashed
    retained = np.where(got_shield, L.y1, L.y0)
    return pd.DataFrame({
        "arm": arm,
        "platform": np.where(old, "old Android", "other"),
        "logged": ~crashed,
        "retained": retained,
        "true_effect": L.y1 - L.y0,
    })


# --------------------------------------------------------------------------
# Juno: retries and a shared cache
# --------------------------------------------------------------------------

V1_COMPLETION = 0.74
V2_LIFT = 0.04          # v2's true lift in completion per lesson
CACHED_SHARE = 0.4      # lessons whose answer comes from the cache
TIMEOUT = {"v1": 0.02, "v2": 0.08}


def juno_test(n_per_arm=4_000, shared_cache=True, seed=20):
    """Session-level log of a learner-randomised prompt test.

    One row per *attempt*. A timed-out attempt is logged as not completed and
    retried once as a new session. If `shared_cache`, cached lessons are
    served the v1 answer whatever the learner's arm, and are never slow.
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_arm
    arm = np.repeat(["v1", "v2"], n_per_arm)
    p_learner = rng.beta(0.74 * 4, 0.26 * 4, n)
    lessons = 1 + rng.negative_binomial(2, 2 / (2 + 4), n)

    learner = np.repeat(np.arange(n), lessons)
    lesson_id = np.arange(len(learner))
    a = arm[learner]
    cached = rng.random(len(learner)) < CACHED_SHARE
    served = np.where(cached & shared_cache, "v1", a)
    p = np.clip(p_learner[learner] + V2_LIFT * (served == "v2"), 0, 1)
    slow = np.where(cached, 0.0, np.where(served == "v2", TIMEOUT["v2"], TIMEOUT["v1"]))
    timed_out = rng.random(len(learner)) < slow

    first = pd.DataFrame({"learner": learner, "arm": a, "lesson": lesson_id, "attempt": 1,
                          "served": served,
                          "completed": np.where(timed_out, 0, rng.random(len(learner)) < p).astype(int)})
    retry = first[timed_out].copy()
    retry["attempt"] = 2
    retry["completed"] = (rng.random(len(retry)) < p[timed_out]).astype(int)
    return pd.concat([first, retry], ignore_index=True).sort_values(["lesson", "attempt"],
                                                                    ignore_index=True)


def per_learner(log: pd.DataFrame) -> pd.DataFrame:
    """Lessons and completions per learner, using each lesson's final attempt."""
    final = log.groupby("lesson").tail(1)
    return final.groupby(["learner", "arm"]).completed.agg(lessons="size", done="sum").reset_index()


def delta_diff(per: pd.DataFrame, a="v2", b="v1"):
    """Difference in completion rate (a - b) with a delta-method 95% CI."""
    out = {}
    for name in (a, b):
        g = per[per.arm == name]
        r = g.done.sum() / g.lessons.sum()
        se = (g.done - r * g.lessons).std(ddof=1) / np.sqrt(len(g)) / g.lessons.mean()
        out[name] = (r, se)
    d = out[a][0] - out[b][0]
    se = np.hypot(out[a][1], out[b][1])
    return d, d - 1.96 * se, d + 1.96 * se


# --------------------------------------------------------------------------
# Multiple exposures: one learner, two devices
# --------------------------------------------------------------------------

DEVICE_LIFT = 0.06     # the new lesson screen's true effect on lessons (+6%)


def two_devices(n=40_000, two_device_share=0.3, assign_by="device", seed=40):
    """Learners on one or two devices (say, a phone and a tablet).

    assign_by="device": each device flips its own coin, as happens when
    bucketing uses a device id. Two-device learners can see both versions.
    assign_by="learner": one coin per learner, applied to all their devices.
    The logged arm is the arm of the first device the learner used.
    """
    rng = np.random.default_rng(seed)
    two = rng.random(n) < two_device_share
    habit = rng.lognormal(np.log(5), 0.6, n) * np.where(two, 1.5, 1.0)   # multi-device learners study more
    if assign_by == "learner":
        d1 = rng.random(n) < 0.5
        d2 = d1
    else:
        d1 = rng.random(n) < 0.5
        d2 = np.where(two, rng.random(n) < 0.5, d1)
    exposure = (d1.astype(float) + d2) / 2           # share of their study time on the new screen
    lessons = rng.poisson(habit * (1 + DEVICE_LIFT * exposure))
    return pd.DataFrame({"two_devices": two, "arm": np.where(d1, "treatment", "control"),
                         "saw_both": d1 != d2, "lessons": lessons})
