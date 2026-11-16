"""End-to-end tests of Algorithm 1 and the Table IV switches on a small synthetic world with fakes."""
import json
import re

import pytest

from cvsm import equations as eq
from cvsm import prompts as P
from cvsm.config import Params, RunConfig, SYSTEM_PRESETS, Switches
from cvsm.llm import LLMClient
from cvsm.llm.base import FakeProvider
from cvsm.memory import ACTIVE, RETRACTED, SUPERSEDED
from cvsm.runner import Resources, run_query
from cvsm.types import AgentSpec, Example, ModelSpec, Passage

from fakes import FakeEncoder, FakeNLI, extraction_response, _norm

JANE = "Film X was directed by Jane Doe."
JOHN = "Film X was directed by John Roe."
BORN = "Jane Doe was born in 1970."

PASSAGES = [Passage("p1", "Film X", JANE), Passage("p2", "Jane Doe", BORN),
            Passage("p3", "Film Y", "Film Y was released in 1999.")]


class World:
    """Scripted LLM behaviour. Records every prompt so tests can check what each agent saw."""

    def __init__(self):
        self.prompts: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        if system == P.DECOMPOSE_SYSTEM:
            return json.dumps({"subtasks": [
                {"id": "t1", "description": "Who directed Film X?", "depends_on": []},
                {"id": "t2", "description": "When was the director identified in t1 born?", "depends_on": ["t1"]}]})
        if system == P.EXTRACT_SYSTEM:
            return extraction_response(user)
        if system == P.ARBITER_SYSTEM:
            opts = dict(re.findall(r"^\[([A-Z])\] (.*)$", user.split("Evidence passages:")[0], re.M))
            pas = dict(re.findall(r"^\[(P\d+)\] (.*)$", user.split("Evidence passages:")[1], re.M))
            for letter, text in opts.items():
                for pid, ptxt in pas.items():
                    if _norm(text) in _norm(ptxt):
                        return json.dumps({"verdict": letter, "passage": pid})
            return json.dumps({"verdict": "insufficient", "passage": None})
        if system == P.SYNTH_SYSTEM:
            return json.dumps({"answer": "1970", "answer_claim": BORN,
                               "statement": f"{JANE} Jane Doe won an Oscar."})
        if system.startswith("You write concise summaries"):
            return "Earlier work found that " + JANE
        if "Write up to 5 short verification questions" in user:             # B4 plan
            return json.dumps({"questions": ["Who directed Film X?", "When was Jane Doe born?"]})
        if "Answer briefly and factually" in user:                            # B4 factored answers
            return JANE if "directed" in user else BORN
        if "Verification questions and independently obtained answers" in user:
            return f"ANSWER: 1970\nEXPLANATION: {JANE} {BORN}"
        # agents
        m = re.search(r"Your sub-task: (.*)", user)
        sub = m.group(1) if m else user
        critic = system == P.ROLE_PROMPTS["critic"]
        revising = "Your previous draft" in user
        if "born" in sub:
            return f"ANSWER: 1970\nEXPLANATION: {BORN}"
        name = "John Roe" if critic and not revising else "Jane Doe"
        return f"ANSWER: {name}\nEXPLANATION: Film X was directed by {name}."


def make_cfg(preset="cvsm", tmp_path=None, **params) -> RunConfig:
    m = ModelSpec("fake", "fake-model")
    agents = [AgentSpec("a1", "retriever", m), AgentSpec("a2", "reasoner", m), AgentSpec("a3", "critic", m)]
    sw = Switches(**SYSTEM_PRESETS[preset])
    return RunConfig(name="test", preset=preset, dataset="hotpotqa", split="validation", n=None, seed=13,
                     run_index=1, k=3, agents=agents, utility_model=m, arbiter_model=m, eval_decomposer_model=m,
                     judge_model=None, nli_checkpoint="fake", label_nli_checkpoint="fake", encoder="fake",
                     params=Params(**params), switches=sw, cache_dir=str(tmp_path / "cache"), concurrency=1)


EX = Example(qid="q1", dataset="hotpotqa", question="When was the director of Film X born?", answer="1970",
             gold_evidence=[JANE, BORN], passages=PASSAGES)


def resources(tmp_path, world=None, offline=False):
    world = world or World()
    llm = LLMClient(tmp_path / "cache", providers={"fake": FakeProvider(world)}, offline=offline)
    nli = FakeNLI(contradictions=[("directed by jane doe", "directed by john roe"),
                                  ("born in 1970", "born in 1971")])
    return Resources.__new__(Resources), world, llm, nli


def build(tmp_path, offline=False, world=None):
    res, world, llm, nli = resources(tmp_path, world, offline)
    res.llm, res.nli, res.encoder = llm, nli, FakeEncoder()
    return res, world


def test_cvsm_end_to_end(tmp_path):
    cfg = make_cfg(tmp_path=tmp_path)
    res, world = build(tmp_path)
    rec = run_query(cfg, res, EX)
    assert rec["status"] == "ok", rec["error"]
    out = rec["output"]
    assert out["answer"] == "1970" and not out["abstained"]
    assert "Oscar" not in out["final_text"]                         # removed by the final entailment filter
    assert BORN in out["final_text"] and JANE in out["final_text"]
    active = [e for e in out["memory_log"] if e["status"] == ACTIVE and e["version"] == 1]
    texts = {e["text"] for e in active}
    assert texts == {JANE, BORN} and JOHN not in texts

    # R1: first-round drafts see only the query, sub-task, role prompt and I_j
    steps = rec["trace"]["steps"]
    d1 = [s for s in steps if s["phase"] == "draft" and s["subtask"] == "t1"]
    d2 = [s for s in steps if s["phase"] == "draft" and s["subtask"] == "t2"]
    # R1: no peer steps and no memory outside I_j; retrieved passages are allowed (DESIGN_NOTES #51)
    assert all(not any(r.startswith(("step:", "entry:")) for r in s["context_refs"]) for s in d1)
    assert all(any(r.startswith("passage:") for r in s["context_refs"]) for s in d1)
    m1 = next(e["id"] for e in active if e["text"] == JANE)
    assert all([r for r in s["context_refs"] if not r.startswith("passage:")] == [f"entry:{m1}"] for s in d2)
    draft_prompts = [u for sy, u in world.prompts if sy in P.ROLE_PROMPTS.values() and "Your previous draft" not in u]
    assert not any("John Roe" in u for u in draft_prompts)         # no peer output in any draft prompt

    # Contested -> evidence arbitration, NLI-checked
    arb = [e for e in rec["trace"]["events"] if e["kind"] == "arbitration"]
    assert arb and arb[0]["accepted"] and arb[0]["outcome"].count("resolved") == 1

    # Eq. (11)-(15): numbers on the written entries
    jane = next(e for e in active if e["text"] == JANE)
    assert jane["label"] == "Resolved"
    expected_jane = eq.confidence(eq.agreement(2, 3, 0.0), 0.0, 0.95 - 0.01, 0.4, 0.2, 0.4)
    assert jane["conf"] == pytest.approx(expected_jane)
    born = next(e for e in active if e["text"] == BORN)
    assert born["label"] == "Corroborated" and born["premises"] == [m1]
    assert born["conf"] == pytest.approx(expected_jane)             # Eq. (15): capped by its premise
    assert born["A"] == pytest.approx(1.0)                          # all three drafts independent

    # revision claims entailed by memory are accepted without a new write
    assert any(e["kind"] == "revision_claim_accepted" for e in rec["trace"]["events"])
    # cost accounting
    c = rec["cost"]["system"]
    assert c["llm_calls"] > 0 and c["nli_calls"] > 0


def test_cache_makes_runs_reproducible(tmp_path):
    cfg = make_cfg(tmp_path=tmp_path)
    res, _ = build(tmp_path)
    r1 = run_query(cfg, res, EX)
    res2, world2 = build(tmp_path, offline=True)                    # cache only: any miss would raise
    r2 = run_query(cfg, res2, EX)
    assert r2["status"] == "ok" and world2.prompts == []
    assert r1["output"]["final_text"] == r2["output"]["final_text"]
    assert r1["output"]["clusters"] == r2["output"]["clusters"]
    assert [s["output_text"] for s in r1["trace"]["steps"]] == [s["output_text"] for s in r2["trace"]["steps"]]


def _cvsm(tmp_path, **params):
    from cvsm.components import QueryContext
    from cvsm.llm import Ledger
    from cvsm.nli import CountingNLI
    from cvsm.pipeline import CVSM
    from cvsm.retrieval import Retriever
    from cvsm.trace import Trace
    cfg = make_cfg(tmp_path=tmp_path, **params)
    res, _ = build(tmp_path)
    led = Ledger()
    ctx = QueryContext(cfg, res.llm, CountingNLI(res.nli, led), res.encoder, Retriever(PASSAGES), EX,
                       Trace("q1"), led, Ledger())
    return CVSM(ctx)


def test_supersession_and_transitive_retraction(tmp_path):
    sys = _cvsm(tmp_path)
    kw = dict(A=0.5, M=0.0, G=0.5, support=[], evidence=[], qualifiers=[], subtask="t1", source_claims=[],
              source_steps=[])
    old = sys.write("Jane Doe was born in 1971.", 0.6, "Corroborated", premises=[], **kw)
    child = sys.write("Jane Doe was 30 years old in 2001.", 0.6, "Corroborated", premises=[old], **kw)
    new = sys.write(BORN, 0.9, "Corroborated", premises=[], **kw)       # 0.9 >= 0.6 + Delta -> supersede
    assert new is not None
    assert sys.mem.get(old).status == SUPERSEDED
    assert sys.mem.get(child).status == RETRACTED                     # premise gone -> Conf* = 0 < tau
    assert sys.mem.get(new).status == ACTIVE


def test_supersession_margin_sends_to_arbiter(tmp_path):
    sys = _cvsm(tmp_path)
    kw = dict(A=0.5, M=0.0, G=0.5, support=[], evidence=[], qualifiers=[], subtask="t1", source_claims=[],
              source_steps=[], premises=[])
    old = sys.write(BORN, 0.6, "Corroborated", **kw)
    new = sys.write("Jane Doe was born in 1971.", 0.65, "Corroborated", **kw)   # margin < Delta -> Arbiter
    assert new is None                                                # passages support the existing entry
    assert sys.mem.get(old).status == ACTIVE
    assert any(e["kind"] == "arbitration" and e["context"] == "supersession" for e in sys.ctx.trace.events)


def test_premise_cap_disabled_in_a8(tmp_path):
    cfg = make_cfg("a8", tmp_path=tmp_path)
    res, _ = build(tmp_path)
    rec = run_query(cfg, res, EX)
    born = next(e for e in rec["output"]["memory_log"] if e["text"] == BORN)
    # G = max ent (.95, passage p2) - max con over all retrieved passages (.02, neutral) = .93 ; uncapped
    assert born["conf"] == pytest.approx(eq.confidence(1.0, 0.0, 0.93, 0.4, 0.2, 0.4))


@pytest.mark.parametrize("preset", sorted(SYSTEM_PRESETS))
def test_every_table_iv_configuration_runs(tmp_path, preset):
    cfg = make_cfg(preset, tmp_path=tmp_path)
    res, world = build(tmp_path)
    rec = run_query(cfg, res, EX)
    assert rec["status"] == "ok", rec["error"]
    out = rec["output"]
    steps = rec["trace"]["steps"]
    if preset == "a1":
        later = [s for s in steps if s["phase"] == "draft" and s["subtask"] == "t1"][1:]
        assert all(any(r.startswith("step:") for r in s["context_refs"]) for s in later)   # read peer drafts
    if preset == "a2":
        assert any(e["text"] == JOHN for e in out["memory_log"])                          # everything written
    if preset == "a4":
        assert all(e["label"] != "Singleton-grounded" for e in out["memory_log"])
    if preset == "b6":
        assert any(e["text"] == JOHN for e in out["memory_log"])
        assert all(e["label"] == "Unverified" for e in out["memory_log"])
    if preset == "b5":
        assert {s["phase"] for s in steps} == {"debate_r1", "debate_r2"}
        assert rec["cost"]["instrumentation"]["llm_calls"] > 0                           # extraction not billed
        assert all(c["component"] != "cvm:extract" for c in [])
    if preset in ("b1", "b2", "b3", "b4"):
        assert rec["cost"]["system"]["by_component"].get("cvm:extract") is None


@pytest.mark.parametrize("shown,gold_h,expect_pred", [(JANE, False, False), (JOHN, True, True)])
def test_halueval_detector(tmp_path, shown, gold_h, expect_pred):
    from cvsm.runner import run_query
    cfg = make_cfg(tmp_path=tmp_path)
    cfg.dataset = "halueval"
    ex = Example(qid="h1", dataset="halueval", question="Who directed Film X?", answer="Jane Doe",
                 gold_evidence=[JANE], passages=[Passage("h1:k", "", JANE)],
                 meta={"shown_answer": f"ANSWER: x\nEXPLANATION: {shown}", "is_hallucinated": gold_h})
    res, _ = build(tmp_path)
    rec = run_query(cfg, res, ex)
    assert rec["status"] == "ok", rec["error"]
    # JANE: A=1/3, G=.94 -> Conf = .4/3 + .4*.94 = .509 >= tau -> written -> predicted correct
    # JOHN: G = .01 - .95 < -delta -> vetoed -> blocked -> predicted hallucinated
    assert rec["output"]["predicted_hallucinated"] is expect_pred


def test_strict_r1_closed_book(tmp_path):
    cfg = make_cfg(tmp_path=tmp_path)
    cfg.switches.draft_retrieval = False
    res, _ = build(tmp_path)
    rec = run_query(cfg, res, EX)
    d1 = [s for s in rec["trace"]["steps"] if s["phase"] == "draft" and s["subtask"] == "t1"]
    assert all(s["context_refs"] == [] for s in d1)


def test_multi_entry_synthesis(tmp_path):
    """The answer claim is entailed only by the two cited entries together (FakeNLI: substring of the
    concatenation), not by either one, so it passes only with multi_entry_synthesis."""
    combined = f"{JANE} {BORN}"
    for multi, expect_abstain in ((True, False), (False, True)):
        world = World()
        orig = world.__call__
        def fn(system, user, orig=orig):
            if system == P.SYNTH_SYSTEM:
                return json.dumps({"answer": "1970", "answer_claim": combined, "statement": "",
                                   "support": ["M1", "M2"]})
            return orig(system, user)
        cfg = make_cfg(tmp_path=tmp_path / str(multi))
        cfg.switches.multi_entry_synthesis = multi
        res, _ = build(tmp_path / str(multi), world=fn)
        rec = run_query(cfg, res, EX)
        assert rec["status"] == "ok", rec["error"]
        assert rec["output"]["abstained"] is expect_abstain
