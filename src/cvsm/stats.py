"""Statistical analysis (Section V-G): query-level bootstrap, McNemar, Holm-Bonferroni, power analysis."""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy import stats as st


def bootstrap_ci(per_query: Sequence, statistic: Callable[[Sequence], float], n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap with the query as the resampling unit."""
    rng = np.random.default_rng(seed)
    data = list(per_query)
    n = len(data)
    point = statistic(data)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        v = statistic([data[i] for i in idx])
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def hr_of(pairs: Sequence[tuple[int, int]]) -> float | None:
    """HR (Eq. 19) as a ratio of sums over queries: pairs are (n_hallucinated, n_claims)."""
    tot = sum(n for _, n in pairs)
    return None if tot == 0 else sum(h for h, _ in pairs) / tot


def paired_delta_hr(a: Sequence[tuple[int, int]], b: Sequence[tuple[int, int]]) -> Callable:
    """Statistic for paired bootstrap of HR(a) - HR(b) over the same queries (indices)."""
    def f(idx_list):
        ha = hr_of([a[i] for i in idx_list]); hb = hr_of([b[i] for i in idx_list])
        return None if ha is None or hb is None else ha - hb
    return f


def mcnemar(x: Sequence[int], y: Sequence[int]) -> dict:
    """Exact McNemar test on paired binary outcomes."""
    b = sum(1 for i, j in zip(x, y) if i == 1 and j == 0)
    c = sum(1 for i, j in zip(x, y) if i == 0 and j == 1)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * st.binom.cdf(min(b, c), n, 0.5))
    return {"b": b, "c": c, "p_value": float(p)}


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running, stop = {}, 0.0, False
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        reject = (not stop) and p <= alpha / (m - i)
        stop = stop or not reject
        out[k] = {"p": p, "p_adj": adj, "reject": reject}
    return out


def required_n_from_se(se_at_n0: float, n0: int, mde: float, alpha: float = 0.05, power: float = 0.8,
                       two_sided: bool = True) -> int:
    """Sample size for a paired, query-level test of an HR difference.

    The standard error of the ratio-of-sums difference scales as 1/sqrt(n), so with the pilot bootstrap SE
    `se_at_n0` at n0 queries: n = n0 * ((z_{1-a/2} + z_{power}) * se_at_n0 / mde)^2."""
    za = st.norm.ppf(1 - alpha / 2) if two_sided else st.norm.ppf(1 - alpha)
    zb = st.norm.ppf(power)
    return int(math.ceil(n0 * ((za + zb) * se_at_n0 / mde) ** 2))


def bootstrap_se(per_query: Sequence, statistic: Callable, n_boot: int = 10_000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    data = list(per_query)
    n = len(data)
    vals = []
    for _ in range(n_boot):
        v = statistic([data[i] for i in rng.integers(0, n, n)])
        if v is not None:
            vals.append(v)
    return float(np.std(vals, ddof=1))


def mcnemar_required_n(p10: float, p01: float, alpha: float = 0.05, power: float = 0.8) -> int | None:
    """Connor (1987) sample size for McNemar's test from discordant-pair probabilities."""
    pd_, diff = p10 + p01, abs(p10 - p01)
    if diff == 0:
        return None
    za, zb = st.norm.ppf(1 - alpha / 2), st.norm.ppf(power)
    n = (za * math.sqrt(pd_) + zb * math.sqrt(pd_ - diff ** 2)) ** 2 / diff ** 2
    return int(math.ceil(n))
