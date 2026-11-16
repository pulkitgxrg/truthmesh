"""Trace metrics on a hand-built trace whose answers are worked out in the comments."""
import json

import pytest

from cvsm.evaluate import MetricEquiv, aggregate, dependency_edges, query_metrics
from cvsm.labelling import Labeller
from cvsm.llm import LLMClient
from cvsm.llm.base import FakeProvider
from cvsm.types import Example, ModelSpec

from fakes import FakeEncoder, FakeNLI

JOHN = "Film X was directed by John Roe."
BORN = "John Roe was born in 1950."
TRUE = "Film X is a 1990 film."


def _step(sid, phase, sub, agent, refs, claims):
    return {"step_id": sid, "phase": phase, "subtask": sub, "agent_id": agent, "role": "r", "model": "m",
            "context_refs": refs, "output_text": "", "claim_ids": claims, "llm_call_keys": []}


def _claim(cid, text, agent, sub, step, phase):
    return {"cid": cid, "text": text, "agent_id": agent, "subtask": sub, "step_id": step, "phase": phase,
            "qualifiers": [], "premises": []}


RECORD = {
    "qid": "q", "system": "cvsm",
    "trace": {
        "steps": [_step("s1", "draft", "t1", "a1", [], ["c1", "c5"]), _step("s2", "draft", "t1", "a2", [], ["c2"]),
                  _step("s3", "draft", "t2", "a1", ["entry:m1"], ["c3"]),
                  _step("s4", "synthesis", "final", "synthesizer", ["entry:m1", "entry:m2", "step:s3"], ["c4"])],
        "claims": {"c1": _claim("c1", JOHN, "a1", "t1", "s1", "draft"), "c2": _claim("c2", JOHN, "a2", "t1", "s2", "draft"),
                   "c3": _claim("c3", BORN, "a1", "t2", "s3", "draft"),
                   "c4": _claim("c4", BORN, "synthesizer", "final", "s4", "synthesis"),
                   "c5": _claim("c5", TRUE, "a1", "t1", "s1", "draft")},
        "events": [{"kind": "memory_write", "t": 1, "active_ids": ["m1"]},
                   {"kind": "memory_write", "t": 2, "active_ids": ["m1", "m2"]}],
    },
    "output": {"answer": "1950", "final_text": BORN, "abstained": False,
               "memory_log": [{"id": "m1", "text": JOHN, "source_claims": ["c1", "c2"], "source_steps": ["s1", "s2"]},
                              {"id": "m2", "text": BORN, "source_claims": ["c3"], "source_steps": ["s3"]}],
               "clusters": [{"conf_star": 0.7, "rep_text": JOHN, "label": "Corroborated", "status": "written"}]},
    "cost": {"system": {"input_tokens": 100, "output_tokens": 10, "llm_calls": 5, "nli_calls": 20, "cost_usd": 0.001},
             "instrumentation": {"cost_usd": 0.0}, "wall_time_s": 2.0},
}
Y = {JOHN: 0, BORN: 0, TRUE: 1}


def test_dependency_edges():
    e = dependency_edges(RECORD["trace"]["steps"], RECORD["output"]["memory_log"])
    assert e["s1"] == {"s3", "s4"} and e["s2"] == {"s3", "s4"} and e["s3"] == {"s4"}


def test_query_metrics_hand_checked():
    equiv = MetricEquiv(FakeNLI(), FakeEncoder())
    q = query_metrics(RECORD, [BORN], Y, equiv)
    assert (q["n_final_claims"], q["n_final_halluc"]) == (1, 1)
    # class {c1,c2}: origin s1, s2 not reachable from s1 -> PD 0, not in final
    # class {c3,c4}: origin s3, s4 depends on s3 -> PD 1, escapes into the final answer
    assert q["pd"] == [0, 1] and q["escaped"] == [False, True]
    # blocked = neither written nor in the final answer: only the true claim c5
    assert sorted(q["vsr_items"]) == sorted([(0, False), (0, False), (0, False), (0, False), (1, True)])
    assert q["mcr_curve"] == [(1, 1.0), (2, 1.0)]
    assert q["clr_clusters"] == [{"subtask": "t1", "S": 2, "S_ind": 2, "accepted": True, "y": 0, "texts": [JOHN, JOHN]}]
    assert q["calib"] == [(0.7, 0)]


def test_aggregate():
    equiv = MetricEquiv(FakeNLI(), FakeEncoder())
    q = query_metrics(RECORD, [BORN], Y, equiv)
    ex = {"q": Example("q", "hotpotqa", "?", "1950", [])}
    m = aggregate([q], [RECORD], ex)
    assert m["HR"] == 1.0 and m["coverage"] == 1.0 and m["EM"] == 1.0
    assert m["VSR"] == 0.0 and m["FRR"] == 1.0
    assert m["PD_mean"] == 0.5 and m["escape_rate"] == 0.5
    assert m["CLR"] == 0.0                     # the false accepted cluster had two independent supporters
    assert m["ECE"] == pytest.approx(0.7) and m["Brier"] == pytest.approx(0.49)
    assert m["cost"]["llm_calls"]["mean"] == 5


def test_labeller_stages(tmp_path):
    ex = Example("q", "hotpotqa", "Who directed Film X?", "Jane Doe", ["Film X: Film X was directed by Jane Doe."])
    nli = FakeNLI(contradictions=[("directed by jane doe", "directed by john roe")])
    lab = Labeller(nli, None, None)
    assert lab.label(ex, "Film X was directed by Jane Doe.")["y"] == 1
    r = lab.label(ex, JOHN)
    assert r["y"] == 0 and r["verdict"] == "contradicted" and r["stage"] == "nli"
    assert lab.label(ex, "Jane Doe likes tea.")["stage"] == "undecided"       # no judge configured: not imputed
    judge = ModelSpec("fakejudge", "j")
    llm = LLMClient(tmp_path, providers={"fakejudge": FakeProvider(lambda s, u: json.dumps({"verdict": "not_supported"}))})
    lab2 = Labeller(nli, llm, judge, agent_families={"anthropic"})
    r = lab2.label(ex, "Jane Doe likes tea.")
    assert r["stage"] == "judge" and r["y"] == 0 and r["verdict"] == "not_supported"
    with pytest.raises(ValueError):
        Labeller(nli, llm, ModelSpec("anthropic", "x"), agent_families={"anthropic"})


def test_judge_family_through_gateway():
    from cvsm.types import ModelSpec
    assert ModelSpec("aicredits", "openai/gpt-4.1-nano").family() == "openai"
    assert ModelSpec("aicredits", "google/gemini-2.5-flash-lite").family() == "google"
    assert ModelSpec("anthropic", "claude-haiku-4-5").family() == "anthropic"
    nli = FakeNLI()
    Labeller(nli, None, ModelSpec("aicredits", "google/gemini-2.5-flash-lite"), agent_families={"openai"})
    with pytest.raises(ValueError):
        Labeller(nli, None, ModelSpec("aicredits", "openai/gpt-4o-mini"), agent_families={"openai"})


def test_cohen_kappa_and_labeller_pr():
    from cvsm.agreement import cohen_kappa, precision_recall
    # po = 0.8 ; pe = (0.6*0.4 + 0.4*0.6)... a: S S S N N, b: S S N N N -> counts a S3 N2, b S2 N3
    # po = 4/5 = .8 ; pe = (3*2 + 2*3)/25 = .48 ; kappa = (.8-.48)/(.52) = 0.6154
    assert cohen_kappa(list("SSSNN"), list("SSNNN")) == pytest.approx(0.32 / 0.52)
    assert cohen_kappa(list("SN"), list("SN")) == pytest.approx(1.0)
    r = precision_recall([0, 0, 1, 1, 0], [0, 1, 1, 0, 0])
    assert (r["tp"], r["fp"], r["fn"]) == (2, 1, 1)
    assert r["precision"] == pytest.approx(2 / 3) and r["recall"] == pytest.approx(2 / 3)
