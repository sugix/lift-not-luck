"""Chapter 13: when the coin isn't obeyed, or isn't contained.

* Encouragement (instrumental variables): Parlo can't force anyone to use
  Juno, but it can randomise a "Try Juno" card. Keen learners use Juno
  anyway, for reasons no log captures; the card is a clean coin flip that
  nudges some learners into using it.
* Interference between learners: a Friends Leaderboard helps the learner who
  has it and also lifts their friends, so individual randomisation leaks.
* Interference through shared infrastructure: a routing policy that sends
  more lessons to Juno's large model slows the shared GPU pool for
  everyone. Learner-level A/B tests can't see that; switchbacks can.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


# --------------------------------------------------------------------------
# Encouragement design
# --------------------------------------------------------------------------

JUNO_EFFECT_LOGIT = 0.25


def encouragement(n=60_000, seed=0):
    """Learners randomised to see a 'Try Juno' card (nudge = 1) or not.

    Juno use depends strongly on hidden keenness and on the card. Retention
    depends on keenness and Juno. Both potential outcomes are kept.
    """
    rng = np.random.default_rng(seed)
    keen = rng.standard_normal(n)
    lessons_wk1 = rng.poisson(np.exp(1.1 + 0.35 * keen))
    nudge = (rng.random(n) < 0.5).astype(int)
    u_use = rng.random(n)
    use0 = (u_use < _sigmoid(-1.2 + 1.3 * keen)).astype(int)            # without the card
    use1 = (u_use < _sigmoid(-1.2 + 1.3 * keen + 1.4)).astype(int)      # with the card
    juno = np.where(nudge == 1, use1, use0)
    base = -1.0 + 0.9 * keen
    u = rng.random(n)
    y0 = (u < _sigmoid(base)).astype(int)
    y1 = (u < _sigmoid(base + JUNO_EFFECT_LOGIT)).astype(int)
    complier = (use1 == 1) & (use0 == 0)
    return pd.DataFrame({"nudge": nudge, "juno": juno, "lessons_wk1": lessons_wk1,
                         "retained": np.where(juno == 1, y1, y0), "y0": y0, "y1": y1,
                         "complier": complier.astype(int),
                         "type": np.select([use0 == 1, complier], ["always", "complier"], "never")})


def wald(y, d, z):
    """IV (Wald) estimate of the effect of d on y using instrument z, with an
    influence-function standard error."""
    y, d, z = map(np.asarray, (y, d, z))
    zc = z - z.mean()
    beta = (zc * y).sum() / (zc * d).sum()
    resid = (y - y.mean()) - beta * (d - d.mean())
    se = np.sqrt(((resid * zc) ** 2).sum()) / abs((zc * d).sum())
    return beta, se


# --------------------------------------------------------------------------
# Friends Leaderboard
# --------------------------------------------------------------------------

OWN_LIFT = 0.04        # lessons +4% for a learner who has the leaderboard
SPILL_PER_FRIEND = 0.02  # +2% per friend who has it


def friend_groups(groups=6_000, size=5, design="individual", share=0.5, seed=0):
    """Weekly lessons for learners in friend groups.

    design="individual": each learner flips their own coin.
    design="cluster":    each friend group flips one coin.
    design="all" / "none": everyone / no one gets it (the god view's two worlds).
    """
    rng = np.random.default_rng(seed)
    n = groups * size
    group = np.repeat(np.arange(groups), size)
    habit = rng.lognormal(np.log(4), 0.6, n)
    if design == "individual":
        t = (rng.random(n) < share).astype(int)
    elif design == "cluster":
        t = np.repeat((rng.random(groups) < share).astype(int), size)
    else:
        t = np.full(n, int(design == "all"))
    treated_friends = np.bincount(group, weights=t)[group] - t
    rate = habit * (1 + OWN_LIFT * t + SPILL_PER_FRIEND * treated_friends)
    lessons = rng.poisson(rate)
    return pd.DataFrame({"group": group, "leaderboard": t, "treated_friends": treated_friends,
                         "lessons": lessons})


# --------------------------------------------------------------------------
# Switchbacks: Juno's model routing and a shared GPU pool
# --------------------------------------------------------------------------

QUALITY = {"medium": 0.74, "large": 0.78}
POLICIES = {"current": 0.3, "aggressive": 0.7}   # share of lessons sent to the large model
LATENCY_COST = 0.04                               # completion lost per extra second
SMOOTHING_MIN = 15                                # how fast the GPU queue reacts (minutes)


def latency(load):
    """Seconds per response given the share of traffic on the large model."""
    return 1.0 + 2.5 * load ** 2


def completion_rate(large_share, load):
    q = QUALITY["medium"] + (QUALITY["large"] - QUALITY["medium"]) * large_share
    return q - LATENCY_COST * (latency(load) - latency(0))


def global_effect():
    """Aggressive everywhere minus current everywhere: what launching does."""
    a, c = POLICIES["aggressive"], POLICIES["current"]
    return completion_rate(a, a) - completion_rate(c, c)


def ab_effect():
    """A 50/50 learner-level A/B: both arms share one pool at the average load."""
    a, c = POLICIES["aggressive"], POLICIES["current"]
    load = (a + c) / 2
    return completion_rate(a, load) - completion_rate(c, load)


def switchback(days=14, block_min=60, lessons_per_min=40, seed=0):
    """Minute-by-minute simulation. Each block of `block_min` minutes runs
    one policy for all traffic, chosen by a coin flip. The GPU load follows
    the policy with exponential smoothing, so each block starts with the
    previous block's congestion (carryover). Traffic and baseline completion
    vary with the hour of day."""
    rng = np.random.default_rng(seed)
    minutes = days * 24 * 60
    blocks = minutes // block_min
    policy = rng.random(blocks) < 0.5                     # True = aggressive
    share = np.where(np.repeat(policy, block_min), POLICIES["aggressive"], POLICIES["current"])
    load = np.empty(minutes)
    level = POLICIES["current"]
    alpha = 1 - np.exp(-1 / SMOOTHING_MIN)
    for m in range(minutes):
        level += alpha * (share[m] - level)
        load[m] = level
    hour = (np.arange(minutes) // 60) % 24
    daily = 0.03 * np.sin(2 * np.pi * (hour - 8) / 24)   # learners do better in the evening
    traffic = rng.poisson(lessons_per_min * (1 + 0.5 * np.sin(2 * np.pi * (hour - 14) / 24)))
    p = completion_rate(share, load) + daily
    done = rng.binomial(traffic, np.clip(p, 0, 1))
    return pd.DataFrame({"minute": np.arange(minutes), "block": np.arange(minutes) // block_min,
                         "minute_in_block": np.arange(minutes) % block_min,
                         "aggressive": share == POLICIES["aggressive"], "load": load,
                         "lessons": traffic, "completed": done, "hour": hour})


def switchback_estimate(log, burn_in=0):
    """Block-level difference in completion (aggressive - current), dropping
    the first `burn_in` minutes of every block. Blocks are the unit."""
    kept = log[log.minute_in_block >= burn_in]
    b = kept.groupby("block").agg(aggressive=("aggressive", "first"),
                                  lessons=("lessons", "sum"), completed=("completed", "sum"))
    b["rate"] = b.completed / b.lessons
    a, c = b.rate[b.aggressive], b.rate[~b.aggressive]
    d = a.mean() - c.mean()
    se = np.sqrt(a.var() / len(a) + c.var() / len(c))
    return d, se
