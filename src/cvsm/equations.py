"""Pure implementations of the equations in research.md Part B. Each function cites its equation."""
from __future__ import annotations

import math
from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

CONTESTED, CORROBORATED, SINGLETON, RESOLVED = "Contested", "Corroborated", "Singleton", "Resolved"


# ---------- Eq. (2): support sets ----------
def support_sets(cluster_agents: Iterable[str], exposed: set[str]) -> tuple[set[str], set[str]]:
    """S(kappa) = agents asserting a member; S_ind = those not exposed to kappa."""
    S = set(cluster_agents)
    return S, {a for a in S if a not in exposed}


# ---------- Eq. (3): consensus laundering rate ----------
def clr(clusters: Sequence[dict]) -> float | None:
    """clusters: dicts with keys accepted(bool), S(int), S_ind(int), y(0/1). K_acc = accepted with S>=2."""
    false_acc = [c for c in clusters if c["accepted"] and c["S"] >= 2 and c["y"] == 0]
    if not false_acc:
        return None
    return sum(1 for c in false_acc if c["S_ind"] <= 1) / len(false_acc)


# ---------- Eq. (4): phi coefficient ----------
def phi(e_i: Sequence[int], e_l: Sequence[int]) -> float | None:
    e_i, e_l = np.asarray(e_i, dtype=int), np.asarray(e_l, dtype=int)
    n = len(e_i)
    if n == 0 or n != len(e_l):
        raise ValueError("error vectors must be non-empty and equal length")
    p11 = np.mean((e_i == 1) & (e_l == 1)); p00 = np.mean((e_i == 0) & (e_l == 0))
    p10 = np.mean((e_i == 1) & (e_l == 0)); p01 = np.mean((e_i == 0) & (e_l == 1))
    p1_, p0_ = p11 + p10, p01 + p00
    p_1, p_0 = p11 + p01, p10 + p00
    den = math.sqrt(p1_ * p0_ * p_1 * p_0)
    if den == 0:
        return None          # undefined when an agent is always right or always wrong
    return float((p11 * p00 - p10 * p01) / den)


def mean_pairwise_rho(errors: dict[str, Sequence[int]]) -> tuple[float | None, dict]:
    """rho_hat = mean of defined pairwise phi values (Section III-D)."""
    pairs = {}
    for a, b in combinations(sorted(errors), 2):
        pairs[f"{a}|{b}"] = phi(errors[a], errors[b])
    vals = [v for v in pairs.values() if v is not None]
    return (float(np.mean(vals)) if vals else None), pairs


# ---------- Eq. (5) and (11) ----------
def n_eff(s: float, rho: float) -> float:
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho_hat must lie in [0, 1] (Eq. 5)")
    if s <= 0:
        return 0.0
    return s / (1.0 + (s - 1.0) * rho)


def agreement(s_ind: int, k: int, rho: float) -> float:
    """Eq. (11): A = n_eff(|S_ind|) / k."""
    return n_eff(s_ind, rho) / k


# ---------- Eq. (12) ----------
def memory_term(items: Sequence[tuple[float, float, float, float]]) -> float:
    """items: (Conf(e), sim(c,e), P_ent(e=>c), P_con(e,c)) for e in R(c)."""
    if not items:
        return 0.0
    w = [conf * sim for conf, sim, _, _ in items]
    tot = sum(w)
    if tot <= 0:
        return 0.0
    return sum(wi * (pe - pc) for wi, (_, _, pe, pc) in zip(w, items)) / tot


# ---------- Eq. (13) ----------
def grounding_term(scores: Sequence[tuple[float, float]]) -> float:
    """scores: (P_ent(psi=>c), P_con(psi,c)) for psi in Psi(c)."""
    if not scores:
        return 0.0
    return max(s[0] for s in scores) - max(s[1] for s in scores)


# ---------- Eq. (14), (15), (16), (17) ----------
def confidence(A: float, M: float, G: float, alpha: float, beta: float, gamma: float) -> float:
    if min(alpha, beta, gamma) < 0 or abs(alpha + beta + gamma - 1.0) > 1e-9:
        raise ValueError("weights must be non-negative and sum to 1 (Eq. 14)")
    return min(1.0, max(0.0, alpha * A + beta * M + gamma * G))


def conf_star(conf: float, premise_confs: Sequence[float]) -> float:
    return min([conf, *premise_confs])


def write_rule(conf_star_: float, M: float, G: float, label: str, tau: float, delta: float) -> bool:
    return conf_star_ >= tau and min(M, G) > -delta and label in (CORROBORATED, SINGLETON, RESOLVED)


def supersession(conf_star_c: float, conf_e: float, Delta: float) -> str:
    """Eq. (17), applied when c contradicts active entry e."""
    return "supersede" if conf_star_c >= conf_e + Delta else "arbiter"


# ---------- Eq. (9) and (10) ----------
def equivalent(p_ent_ab: float, p_ent_ba: float, theta_e: float) -> bool:
    return p_ent_ab >= theta_e and p_ent_ba >= theta_e


def contradicts(p_con_ab: float, p_con_ba: float, theta_c: float) -> bool:
    return max(p_con_ab, p_con_ba) >= theta_c


def cluster_label(contradicted: bool, s_ind: int) -> str:
    if contradicted:
        return CONTESTED
    return CORROBORATED if s_ind >= 2 else SINGLETON


def connected_components(n: int, edges: Iterable[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


# ---------- Eq. (6): propagation depth ----------
def shortest_paths_from(src: str, edges: dict[str, set[str]]) -> dict[str, int]:
    """BFS over the dependency relation; edges[u] = steps that depend on u."""
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in edges.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def propagation_depth(origin: str, later_steps: Iterable[str], edges: dict[str, set[str]]) -> int:
    """PD(h) = max over later asserting steps of d(o(h), sigma); 0 if none reachable.

    Steps that re-assert h without a dependency path from o(h) are independent re-assertions, not
    propagation, so they do not contribute (DESIGN_NOTES, PD)."""
    dist = shortest_paths_from(origin, edges)
    ds = [dist[s] for s in later_steps if s in dist and s != origin]
    return max(ds) if ds else 0


# ---------- Eq. (7) ----------
def mcr(active_labels: Sequence[int]) -> float | None:
    if not active_labels:
        return None
    return sum(1 for y in active_labels if y == 0) / len(active_labels)


# ---------- Eq. (19) ----------
def hallucination_rate(per_query: Sequence[tuple[int, int]]) -> float | None:
    """per_query: (n_hallucinated, n_claims) for D(y_hat_q)."""
    tot = sum(n for _, n in per_query)
    return None if tot == 0 else sum(h for h, _ in per_query) / tot


# ---------- Eq. (20) ----------
def response_consistency(per_query_answers: Sequence[Sequence[str]], equiv=lambda a, b: a == b) -> float:
    vals = []
    for answers in per_query_answers:
        U = len(answers)
        if U < 2:
            raise ValueError("RC needs U >= 2 runs per query")
        agree = sum(1 for a, b in combinations(answers, 2) if equiv(a, b))
        vals.append(2.0 * agree / (U * (U - 1)))
    return float(np.mean(vals))


# ---------- Eq. (21) ----------
def vsr_frr(claims: Sequence[tuple[int, bool]]) -> tuple[float | None, float | None]:
    """claims: (y, blocked) for c in C_int."""
    neg = [b for y, b in claims if y == 0]
    pos = [b for y, b in claims if y == 1]
    vsr = sum(neg) / len(neg) if neg else None
    frr = sum(pos) / len(pos) if pos else None
    return vsr, frr


# ---------- Eq. (22) and Brier ----------
def ece(conf: Sequence[float], correct: Sequence[int], bins: int = 10) -> float | None:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = len(conf)
    if n == 0:
        return None
    idx = np.minimum((conf * bins).astype(int), bins - 1)   # equal-width bins, 1.0 in the last bin
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(conf: Sequence[float], correct: Sequence[int]) -> float | None:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return None if len(conf) == 0 else float(np.mean((conf - correct) ** 2))
