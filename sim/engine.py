"""Chapter 15: a small lift engine.

Everything an experiment readout needs, in one place, built from the
earlier chapters:

* deterministic hash assignment with a salt per experiment (consistent and independent)
* sample ratio checks, overall and by segment (Chapter 3)
* per-user means with CUPED (Chapter 5) and ratio metrics with the delta
  method (Chapter 2), both on the unit you randomised
* guardrails as non-inferiority tests (Chapter 2)
* Benjamini-Hochberg across secondary metrics (Chapter 6)
* chance to win and expected loss for the decision (Chapter 4)

It's deliberately small: about the size of what a team could own and test.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------

def bucket(user_id, salt: str) -> float:
    """A stable number in [0, 1) for this user in this experiment."""
    h = hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()
    return int(h[:15], 16) / 16 ** 15


def assign(user_ids, salt: str, treatment_share: float = 0.5) -> np.ndarray:
    """'treatment' or 'control' for each user, the same every time it's asked."""
    u = np.array([bucket(uid, salt) for uid in user_ids])
    return np.where(u < treatment_share, "treatment", "control")


# --------------------------------------------------------------------------
# Metric definitions
# --------------------------------------------------------------------------

@dataclass
class Metric:
    name: str
    numerator: str                    # column summed per user
    denominator: str | None = None    # for ratio metrics (per-lesson, per-turn)
    covariate: str | None = None      # pre-period column for CUPED
    role: str = "secondary"           # "decision", "guardrail" or "secondary"
    higher_is_better: bool = True
    margin: float | None = None       # guardrails: largest acceptable harm (relative)


@dataclass
class Experiment:
    name: str
    metrics: list
    treatment_share: float = 0.5
    segments: list = field(default_factory=list)
    alpha: float = 0.05
    threshold_of_caring: float = 0.002    # relative expected loss we'd shrug at


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def srm_table(df: pd.DataFrame, exp: Experiment) -> pd.DataFrame:
    """Sample ratio checks overall and inside each segment value."""
    rows = []

    def check(sub, label):
        n_t = int((sub.arm == "treatment").sum())
        n = len(sub)
        p = stats.chisquare([n_t, n - n_t], [n * exp.treatment_share, n * (1 - exp.treatment_share)]).pvalue
        rows.append({"slice": label, "users": n, "treatment share": n_t / n, "p_value": p})

    check(df, "overall")
    for seg in exp.segments:
        for value, sub in df.groupby(seg):
            check(sub, f"{seg} = {value}")
    out = pd.DataFrame(rows).set_index("slice")
    out["fails (p < 0.001)"] = out.p_value < 0.001
    return out


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------

def _arm_stats(df, m: Metric, theta: float, x_mean: float):
    """Mean of the metric for one arm (per user) and the variance of that mean."""
    y = df[m.numerator].to_numpy(float)
    if m.denominator is None:
        if m.covariate is not None:
            y = y - theta * (df[m.covariate].to_numpy(float) - x_mean)
        return y.mean(), y.var(ddof=1) / len(y)
    d = df[m.denominator].to_numpy(float)
    r = y.sum() / d.sum()
    resid = y - r * d
    return r, resid.var(ddof=1) / len(y) / d.mean() ** 2


def estimate(df: pd.DataFrame, m: Metric) -> dict:
    """Relative lift of treatment over control, with its standard error."""
    theta, x_mean = 0.0, 0.0
    if m.covariate is not None and m.denominator is None:
        x, y = df[m.covariate].to_numpy(float), df[m.numerator].to_numpy(float)
        theta = np.cov(x, y)[0, 1] / x.var(ddof=1)
        x_mean = x.mean()
    mt, vt = _arm_stats(df[df.arm == "treatment"], m, theta, x_mean)
    mc, vc = _arm_stats(df[df.arm == "control"], m, theta, x_mean)
    lift = mt / mc - 1
    se = np.sqrt(vt / mc ** 2 + vc * mt ** 2 / mc ** 4)        # delta method for a ratio of means
    return {"control": mc, "treatment": mt, "lift": lift, "se": se}


def _chance_and_loss(good_lift, se):
    """Normal posterior (flat prior): chance the change is good, and the
    expected loss of shipping it, both in relative terms."""
    z = good_lift / se
    return stats.norm.cdf(z), se * stats.norm.pdf(z) - good_lift * stats.norm.cdf(-z)


def readout(df: pd.DataFrame, exp: Experiment) -> pd.DataFrame:
    rows = []
    for m in exp.metrics:
        e = estimate(df, m)
        sign = 1 if m.higher_is_better else -1
        good = sign * e["lift"]                         # positive = the change is good
        z = e["lift"] / e["se"]
        row = {"metric": m.name, "role": m.role, "control": e["control"], "treatment": e["treatment"],
               "lift": e["lift"], "ci_low": e["lift"] - 1.96 * e["se"], "ci_high": e["lift"] + 1.96 * e["se"],
               "p_value": 2 * stats.norm.sf(abs(z))}
        if m.role == "guardrail":
            # non-inferiority: can we reject "the harm is at least the margin"?
            row["p_non_inferior"] = stats.norm.sf((good + m.margin) / e["se"])
            # one-sided at alpha / 2, so "pass" agrees with the 95% interval we report
            row["guardrail"] = "pass" if row["p_non_inferior"] < exp.alpha / 2 else "fail"
        if m.role == "decision":
            row["chance_to_win"], row["expected_loss"] = _chance_and_loss(good, e["se"])
        rows.append(row)
    out = pd.DataFrame(rows).set_index("metric")
    sec = out.role == "secondary"
    if sec.any():
        out.loc[sec, "p_bh"] = _bh(out.loc[sec, "p_value"].to_numpy())
    return out


def _bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank, i in reversed(list(enumerate(order, start=1))):
        running = min(running, p[i] * n / rank)
        adj[i] = running
    return adj


# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------

def decide(df: pd.DataFrame, exp: Experiment) -> str:
    """A short, plain-English decision memo."""
    srm = srm_table(df, exp)
    if srm["fails (p < 0.001)"].any():
        bad = ", ".join(srm.index[srm["fails (p < 0.001)"]])
        return f"STOP: sample ratio mismatch in {bad}. Find the cause before reading any metric."
    r = readout(df, exp)
    lines = [f"Experiment: {exp.name}"]
    d = r[r.role == "decision"].iloc[0]
    lines.append(f"Decision metric '{r[r.role == 'decision'].index[0]}': {d.lift:+.1%} "
                 f"(95% CI {d.ci_low:+.1%} to {d.ci_high:+.1%}), chance to win {d.chance_to_win:.0%}, "
                 f"expected loss if shipped {d.expected_loss:.2%}.")
    failed = r[(r.role == "guardrail") & (r.guardrail == "fail")]
    for name, g in r[r.role == "guardrail"].iterrows():
        lines.append(f"Guardrail '{name}': {g.lift:+.1%} (95% CI {g.ci_low:+.1%} to {g.ci_high:+.1%}), {g.guardrail}.")
    sec = r[r.role == "secondary"]
    sig = sec.index[sec.p_bh < exp.alpha].tolist() if len(sec) else []
    lines.append(f"Secondary metrics moving after FDR correction: {', '.join(sig) if sig else 'none'}.")
    if len(failed):
        lines.append("RECOMMENDATION: do not ship as is. Guardrail(s) not shown to be safe: "
                     + ", ".join(failed.index) + ".")
    elif d.expected_loss < exp.threshold_of_caring:
        lines.append("RECOMMENDATION: ship. The expected loss is below the threshold of caring.")
    else:
        lines.append("RECOMMENDATION: keep running or redesign. The downside is still too uncertain.")
    return "\n".join(lines)
