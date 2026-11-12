"""Parlo: a fictional language-learning app, and Juno, its AI tutor.

This module is the book's simulator. Every number in the chapters comes from
code here, with a fixed seed, so the results are the same on every build.

Chapter 7 uses two pieces:

* ``REMINDERS``: four streak-reminder push messages with *true* return rates.
  In real life nobody knows these; the simulator knows them so we can measure
  how much each strategy loses while it learns (regret).
* ``JUNO_MODELS``: three LLMs Juno can route a lesson to, each with a true
  lesson-completion rate and a cost per lesson.

Policies are vectorised across many independent runs: arrays have shape
``(n_runs, n_arms)``, and we loop only over time. That keeps a full comparison
inside a few seconds on a laptop CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# The world (only the simulator knows these numbers)
# --------------------------------------------------------------------------

REMINDERS = {
    "Plain nudge": 0.090,       # "Time for your Spanish lesson."
    "Streak at risk": 0.105,    # "Your 12-day streak ends tonight."
    "Juno says hi": 0.120,      # "Juno saved a 3-minute lesson for you."
    "Progress recap": 0.100,    # "You learned 48 words this week."
}


@dataclass(frozen=True)
class Model:
    name: str
    completion_rate: float   # chance a lesson routed to this model is completed
    cost: float              # dollars per lesson


JUNO_MODELS = [
    Model("large", 0.78, 0.020),
    Model("medium", 0.74, 0.006),
    Model("small", 0.62, 0.001),
]

LESSON_VALUE = 0.10  # dollars of value we assign to one completed lesson


# --------------------------------------------------------------------------
# Policies
# --------------------------------------------------------------------------
# Each policy is a function (state, rng) -> chosen arm per run, shape (n_runs,).
# ``state`` holds per-run counts: pulls[n_runs, K], wins[n_runs, K], t.


@dataclass
class State:
    pulls: np.ndarray
    wins: np.ndarray
    t: int = 0

    @classmethod
    def empty(cls, n_runs: int, k: int) -> "State":
        return cls(np.zeros((n_runs, k)), np.zeros((n_runs, k)))

    @property
    def means(self) -> np.ndarray:
        """Observed success rate per arm (0 where an arm was never tried)."""
        return np.divide(self.wins, self.pulls, out=np.zeros_like(self.wins),
                         where=self.pulls > 0)


def _argmax_random_ties(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row-wise argmax that breaks ties at random instead of always picking 0."""
    noise = rng.random(x.shape) * 1e-9
    return np.argmax(x + noise, axis=1)


def fixed_split(state: State, rng: np.random.Generator) -> np.ndarray:
    """The A/B/n test: every send goes to a uniformly random message."""
    n_runs, k = state.pulls.shape
    return rng.integers(0, k, size=n_runs)


def epsilon_greedy(epsilon: float = 0.1):
    """Exploit the best-looking arm, but explore at random with prob epsilon."""
    def policy(state: State, rng: np.random.Generator) -> np.ndarray:
        n_runs, k = state.pulls.shape
        greedy = _argmax_random_ties(state.means, rng)
        explore = rng.random(n_runs) < epsilon
        return np.where(explore, rng.integers(0, k, size=n_runs), greedy)
    policy.__name__ = f"epsilon_greedy({epsilon})"
    return policy


def ucb1(state: State, rng: np.random.Generator) -> np.ndarray:
    """Optimism in the face of uncertainty: mean plus a confidence bonus."""
    untried = state.pulls == 0
    t = max(state.t, 1)
    bonus = np.sqrt(2 * np.log(t) / np.maximum(state.pulls, 1))
    score = np.where(untried, np.inf, state.means + bonus)
    return _argmax_random_ties(score, rng)


def thompson(state: State, rng: np.random.Generator) -> np.ndarray:
    """Thompson sampling with a Beta(1, 1) prior on each arm's success rate."""
    draws = rng.beta(1 + state.wins, 1 + state.pulls - state.wins)
    return np.argmax(draws, axis=1)


# --------------------------------------------------------------------------
# Running a policy against the world
# --------------------------------------------------------------------------


@dataclass
class Result:
    choices: np.ndarray   # (n_runs, horizon) arm chosen at each send
    rewards: np.ndarray   # (n_runs, horizon) 1 if the user came back
    rates: np.ndarray     # true rates, (K,)

    @property
    def regret(self) -> np.ndarray:
        """Cumulative expected regret per run, shape (n_runs, horizon).

        At each send, regret is how much lower the chosen message's *true*
        rate was than the best message's. Summing it gives the expected number
        of returns we gave up while learning.
        """
        gap = self.rates.max() - self.rates[self.choices]
        return np.cumsum(gap, axis=1)

    def share_of_best(self) -> np.ndarray:
        """Fraction of runs sending the best arm at each step, shape (horizon,)."""
        return (self.choices == self.rates.argmax()).mean(axis=0)


def simulate(policy, rates, horizon: int = 10_000, n_runs: int = 200,
             seed: int = 7) -> Result:
    """Run ``policy`` for ``horizon`` sends in each of ``n_runs`` worlds."""
    rates = np.asarray(rates, dtype=float)
    rng = np.random.default_rng(seed)
    k = len(rates)
    state = State.empty(n_runs, k)
    choices = np.empty((n_runs, horizon), dtype=np.int8)
    rewards = np.empty((n_runs, horizon), dtype=np.int8)
    rows = np.arange(n_runs)
    for t in range(horizon):
        state.t = t
        arm = policy(state, rng)
        reward = rng.random(n_runs) < rates[arm]
        state.pulls[rows, arm] += 1
        state.wins[rows, arm] += reward
        choices[:, t] = arm
        rewards[:, t] = reward
    return Result(choices, rewards, rates)


def trace_thompson(rates, steps: int = 20, seed: int = 3):
    """One Thompson run, step by step, for the 'watch it think' table.

    Returns a list of dicts: the sampled guess for every arm, the arm picked,
    and whether the user came back.
    """
    rates = np.asarray(rates, dtype=float)
    rng = np.random.default_rng(seed)
    k = len(rates)
    wins = np.zeros(k)
    pulls = np.zeros(k)
    rows = []
    for t in range(steps):
        draws = rng.beta(1 + wins, 1 + pulls - wins)
        arm = int(np.argmax(draws))
        reward = bool(rng.random() < rates[arm])
        rows.append({"step": t + 1, "draws": draws.round(3), "picked": arm,
                     "came_back": reward})
        pulls[arm] += 1
        wins[arm] += reward
    return rows


def juno_rewards(value: float = LESSON_VALUE):
    """Expected net value per lesson for each Juno model: value * p - cost."""
    return np.array([value * m.completion_rate - m.cost for m in JUNO_MODELS])


def thompson_with_cost(costs, value: float = LESSON_VALUE):
    """Thompson sampling on completion rate, then subtract each model's known cost.

    The uncertain part (does the learner finish?) is learned; the known part
    (what the model costs) is just arithmetic.
    """
    costs = np.asarray(costs, dtype=float)

    def policy(state: State, rng: np.random.Generator) -> np.ndarray:
        p = rng.beta(1 + state.wins, 1 + state.pulls - state.wins)
        return np.argmax(value * p - costs, axis=1)
    policy.__name__ = "thompson_with_cost"
    return policy
