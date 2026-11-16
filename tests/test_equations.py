"""Hand-checked examples for every equation in research.md Part B (values computed by hand in comments)."""
import math

import pytest

from cvsm import equations as eq


def test_eq2_support_sets():
    S, S_ind = eq.support_sets(["a1", "a2", "a3"], exposed={"a2"})
    assert S == {"a1", "a2", "a3"} and S_ind == {"a1", "a3"}


def test_eq3_clr():
    clusters = [
        {"accepted": True, "S": 3, "S_ind": 1, "y": 0},    # laundered
        {"accepted": True, "S": 2, "S_ind": 2, "y": 0},    # shared misconception: denominator only
        {"accepted": True, "S": 3, "S_ind": 0, "y": 1},    # true: excluded
        {"accepted": False, "S": 3, "S_ind": 0, "y": 0},   # not accepted: excluded
        {"accepted": True, "S": 1, "S_ind": 1, "y": 0},    # |S| < 2: not in K_acc
    ]
    assert eq.clr(clusters) == pytest.approx(0.5)
    assert eq.clr([{"accepted": True, "S": 2, "S_ind": 2, "y": 1}]) is None


def test_eq4_phi():
    # p11=.4 p10=.2 p01=0 p00=.4 ; num=.16 ; den=sqrt(.6*.4*.4*.6)=.24 -> 2/3
    assert eq.phi([1, 1, 1, 0, 0], [1, 1, 0, 0, 0]) == pytest.approx(2 / 3)
    assert eq.phi([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert eq.phi([1, 0, 1, 0], [1, 1, 0, 0]) == pytest.approx(0.0)
    assert eq.phi([1, 1, 1], [0, 1, 0]) is None          # agent always wrong -> undefined
    rho, pairs = eq.mean_pairwise_rho({"a": [1, 1, 0, 0], "b": [1, 1, 0, 0], "c": [1, 0, 1, 0]})
    # pairs: a|b = 1, a|c = 0, b|c = 0 -> mean 1/3
    assert rho == pytest.approx(1 / 3) and pairs["a|b"] == pytest.approx(1.0)


def test_eq5_eq11_effective_support_and_agreement():
    assert eq.n_eff(3, 0.0) == 3
    assert eq.n_eff(3, 1.0) == 1
    assert eq.n_eff(3, 0.5) == pytest.approx(1.5)        # 3 / (1 + 2*0.5)
    assert eq.agreement(3, 3, 0.5) == pytest.approx(0.5)
    assert eq.agreement(1, 3, 0.5) == pytest.approx(1 / 3)
    assert eq.agreement(0, 3, 0.5) == 0.0
    with pytest.raises(ValueError):
        eq.n_eff(2, 1.5)
    # Remark 1: agreement increases strictly in s for rho < 1
    vals = [eq.agreement(s, 5, 0.3) for s in range(1, 6)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_eq12_memory_term():
    # w = 1.0*0.8 = .8 and 0.5*0.6 = .3 ; num = .8*(.9-.05) + .3*(.1-.7) = .68 - .18 = .5 ; M = .5/1.1
    items = [(1.0, 0.8, 0.9, 0.05), (0.5, 0.6, 0.1, 0.7)]
    assert eq.memory_term(items) == pytest.approx(0.5 / 1.1)
    assert eq.memory_term([]) == 0.0


def test_eq13_grounding_term():
    assert eq.grounding_term([(0.9, 0.05), (0.2, 0.3)]) == pytest.approx(0.6)   # .9 - .3
    assert eq.grounding_term([]) == 0.0


def test_eq14_confidence():
    # .4*.5 + .2*(.5/1.1) + .4*.6 = .2 + .090909 + .24
    assert eq.confidence(0.5, 0.5 / 1.1, 0.6, 0.4, 0.2, 0.4) == pytest.approx(0.2 + 0.2 * 0.5 / 1.1 + 0.24)
    assert eq.confidence(1, 1, 1, 0.4, 0.2, 0.4) == 1.0
    assert eq.confidence(0, -1, -1, 0.4, 0.2, 0.4) == 0.0          # clipped
    with pytest.raises(ValueError):
        eq.confidence(0.5, 0, 0, 0.5, 0.5, 0.5)                    # weights must sum to 1


def test_eq15_conf_star():
    assert eq.conf_star(0.8, [0.6, 0.9]) == 0.6
    assert eq.conf_star(0.5, []) == 0.5


def test_eq16_write_rule():
    assert eq.write_rule(0.6, 0.1, 0.2, eq.CORROBORATED, tau=0.5, delta=0.5)
    assert not eq.write_rule(0.49, 0.1, 0.2, eq.CORROBORATED, 0.5, 0.5)       # below tau
    assert not eq.write_rule(0.9, -0.5, 0.9, eq.CORROBORATED, 0.5, 0.5)       # veto: min(M,G) = -.5 not > -.5
    assert eq.write_rule(0.9, -0.49, 0.9, eq.CORROBORATED, 0.5, 0.5)
    assert not eq.write_rule(0.9, 0.0, 0.9, eq.CONTESTED, 0.5, 0.5)           # label not admissible
    assert eq.write_rule(0.9, 0.0, 0.9, eq.RESOLVED, 0.5, 0.5)
    assert eq.write_rule(0.9, 0.0, 0.9, eq.SINGLETON, 0.5, 0.5)


def test_remark1_bounds():
    alpha, beta, gamma, k, rho = 0.4, 0.2, 0.4, 3, 0.5
    conf_all = eq.confidence(eq.agreement(k, k, rho), 0, 0, alpha, beta, gamma)
    assert conf_all == pytest.approx(alpha / (1 + (k - 1) * rho))             # = 0.2
    single = eq.confidence(eq.agreement(1, k, rho), 0, 0.5, alpha, beta, gamma)
    assert single == pytest.approx(alpha / k + gamma * 0.5)


def test_eq17_supersession():
    assert eq.supersession(0.8, 0.6, 0.1) == "supersede"
    assert eq.supersession(0.65, 0.6, 0.1) == "arbiter"


def test_eq9_eq10_alignment_and_labels():
    assert eq.equivalent(0.8, 0.75, 0.7) and not eq.equivalent(0.8, 0.6, 0.7)
    assert eq.contradicts(0.1, 0.71, 0.7) and not eq.contradicts(0.6, 0.69, 0.7)
    assert eq.cluster_label(True, 3) == eq.CONTESTED
    assert eq.cluster_label(False, 2) == eq.CORROBORATED
    assert eq.cluster_label(False, 1) == eq.SINGLETON
    assert eq.cluster_label(False, 0) == eq.SINGLETON
    assert eq.connected_components(5, [(0, 1), (1, 2), (3, 4)]) == [[0, 1, 2], [3, 4]]


def test_eq6_propagation_depth():
    edges = {"a": {"b", "d"}, "b": {"c"}}
    assert eq.propagation_depth("a", ["c", "d"], edges) == 2
    assert eq.propagation_depth("a", ["e"], edges) == 0          # independent re-assertion, unreachable
    assert eq.propagation_depth("a", [], edges) == 0


def test_eq7_mcr():
    assert eq.mcr([1, 0, 0, 1]) == 0.5
    assert eq.mcr([]) is None


def test_eq19_hr():
    assert eq.hallucination_rate([(1, 4), (0, 0), (2, 6)]) == pytest.approx(0.3)
    assert eq.hallucination_rate([(0, 0)]) is None


def test_eq20_rc():
    assert eq.response_consistency([["a", "a", "a"], ["a", "b", "a"]]) == pytest.approx((1 + 1 / 3) / 2)
    with pytest.raises(ValueError):
        eq.response_consistency([["a"]])


def test_eq21_vsr_frr():
    items = [(0, True), (0, False), (0, True), (1, False), (1, True), (1, False), (1, False)]
    vsr, frr = eq.vsr_frr(items)
    assert vsr == pytest.approx(2 / 3) and frr == pytest.approx(1 / 4)


def test_eq22_ece_and_brier():
    conf, correct = [0.15, 0.15, 0.85, 0.85], [0, 1, 1, 1]
    # bin [.1,.2): acc .5 conf .15 -> .5*.35 ; bin [.8,.9): acc 1 conf .85 -> .5*.15 ; total .25
    assert eq.ece(conf, correct) == pytest.approx(0.25)
    assert eq.brier(conf, correct) == pytest.approx((0.15**2 + 0.85**2 + 0.15**2 + 0.15**2) / 4)
    assert eq.ece([1.0], [1]) == pytest.approx(0.0)          # conf = 1.0 falls in the last bin
