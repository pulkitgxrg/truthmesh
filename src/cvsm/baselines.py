"""Baselines B1-B5 (Table IV). B6 is CV-SM with verification switched off (pipeline.py).

Claim extraction on baseline steps is instrumentation for the trace metrics (PD, CLR, VSR): it is charged
to the instrumentation ledger, not to the baseline's cost."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from . import prompts as P
from .components import QueryContext, agent_respond, extract_claims, render_passages
from .datasets import normalize_answer
from .jsonutil import parse_answer
from .pipeline import assign_agents, topo_order
from .components import decompose_task
from .types import AgentSpec, Step, SubTask


def _single_agent(ctx: QueryContext) -> AgentSpec:
    a = ctx.cfg.agents[0]
    return AgentSpec(a.agent_id, "assistant", a.model)


def _passages_block(ctx, query) -> tuple[str, list]:
    ps = ctx.retriever(query, ctx.cfg.params.p)
    return (f"\nRetrieved passages:\n{render_passages(ps)}\n" if ps else ""), ps


def _result(ctx, step: Step, answer: str, text: str, extra: dict | None = None) -> dict:
    claims = extract_claims(ctx, ctx.ex.question, [step], {}, "(none)", instrument=True)
    out = {"answer": answer, "final_text": text, "abstained": not answer.strip(),
           "final_claim_ids": [c.cid for c in claims], "memory_log": [], "clusters": []}
    out.update(extra or {})
    return out


def _answer_text(raw: str) -> tuple[str, str]:
    ans, exp = parse_answer(raw)
    text = (f"The answer is {ans}. " if ans else "") + exp
    return ans, text.strip()


def run_single(ctx: QueryContext, retrieval: bool) -> dict:
    """B1 (no retrieval) and B2 (retrieval in context)."""
    agent = _single_agent(ctx)
    st = SubTask("t1", ctx.ex.question, [])
    block, ps = _passages_block(ctx, ctx.ex.question) if retrieval else ("", [])
    step = agent_respond(ctx, agent, st, "answer", block, [f"passage:{p.pid}" for p in ps], "", ctx.trace.new_id("s"))
    ctx.trace.add_step(step)
    ans, text = _answer_text(step.output_text)
    return _result(ctx, step, ans, text)


def run_self_consistency(ctx: QueryContext) -> dict:
    """B3: k' independent samples, majority vote over normalised short answers."""
    agent = _single_agent(ctx)
    st = SubTask("t1", ctx.ex.question, [])
    ids = [ctx.trace.new_id("s") for _ in range(ctx.cfg.switches.sc_samples)]
    with ThreadPoolExecutor(max_workers=min(8, len(ids))) as ex:
        steps = list(ex.map(lambda i_s: agent_respond(ctx, agent, st, "sample", "", [], f"sample{i_s[0]}", i_s[1]),
                            enumerate(ids)))
    for s in steps:
        ctx.trace.add_step(s)
    parsed = [_answer_text(s.output_text) for s in steps]
    votes = Counter(normalize_answer(a) for a, _ in parsed if a)
    if not votes:
        return _result(ctx, steps[0], "", "")
    top = max(votes.values())
    winner = next(normalize_answer(a) for a, _ in parsed if a and votes[normalize_answer(a)] == top)
    i = next(i for i, (a, _) in enumerate(parsed) if a and normalize_answer(a) == winner)
    for s in steps:
        if s is not steps[i]:
            extract_claims(ctx, ctx.ex.question, [s], {}, "(none)", instrument=True)
    return _result(ctx, steps[i], parsed[i][0], parsed[i][1], {"votes": dict(votes)})


def run_cove(ctx: QueryContext) -> dict:
    """B4: Chain-of-Verification, factored variant; verification questions answered with retrieval and
    without the draft in context."""
    agent = _single_agent(ctx)
    st = SubTask("t1", ctx.ex.question, [])
    draft = agent_respond(ctx, agent, st, "draft", "", [], "", ctx.trace.new_id("s"))
    ctx.trace.add_step(draft)
    extract_claims(ctx, ctx.ex.question, [draft], {}, "(none)", instrument=True)
    plan = ctx.call_json(agent.model, P.ROLE_PROMPTS["assistant"],
                         P.COVE_PLAN_USER.format(question=ctx.ex.question, draft=draft.output_text), "cove:plan")
    questions = [str(q) for q in (plan or {}).get("questions", [])][:5] if isinstance(plan, dict) else []
    qa, refs = [], [f"step:{draft.step_id}"]
    for i, vq in enumerate(questions):
        block, ps = _passages_block(ctx, vq)
        text = ctx.call(agent.model, P.ROLE_PROMPTS["assistant"],
                        P.COVE_ANSWER_USER.format(passages=block, question=vq), "cove:verify")
        vstep = Step(step_id=ctx.trace.new_id("s"), phase="verify", subtask="t1", agent_id=agent.agent_id,
                     role="assistant", model=agent.model.key(), context_refs=[f"passage:{p.pid}" for p in ps],
                     output_text=text)
        ctx.trace.add_step(vstep)
        extract_claims(ctx, vq, [vstep], {}, "(none)", instrument=True)
        refs.append(f"step:{vstep.step_id}")
        qa.append(f"Q{i + 1}: {vq}\nA{i + 1}: {text}")
    final_raw = ctx.call(agent.model, P.ROLE_PROMPTS["assistant"],
                         P.COVE_FINAL_USER.format(question=ctx.ex.question, draft=draft.output_text,
                                                  qa="\n".join(qa) or "(none)", fmt=P.ANSWER_FORMAT), "cove:final")
    final = Step(step_id=ctx.trace.new_id("s"), phase="final", subtask="t1", agent_id=agent.agent_id,
                 role="assistant", model=agent.model.key(), context_refs=refs, output_text=final_raw)
    ctx.trace.add_step(final)
    ans, text = _answer_text(final_raw)
    return _result(ctx, final, ans, text)


def run_debate(ctx: QueryContext) -> dict:
    """B5: multi-agent debate on the same decomposition, roles and models as CV-SM; R rounds (R includes
    the independent first round), retrieval in context, final majority. Sub-task results are passed
    downstream as free-text messages."""
    cfg = ctx.cfg
    subtasks = topo_order(decompose_task(ctx))
    pool = cfg.agent_pool()
    winners: dict[str, Step] = {}
    final_answer, final_step = "", None
    for j, st in enumerate(subtasks):
        agents = assign_agents(pool, cfg.k, j)
        msgs = [winners[d] for d in st.depends_on if d in winners]
        inputs = ""
        if msgs:
            inputs = "\nResults from earlier sub-tasks:\n" + "\n\n".join(
                f"[{m.subtask}] {m.output_text}" for m in msgs) + "\n"
        pblock, ps = _passages_block(ctx, st.description)
        base_refs = [f"step:{m.step_id}" for m in msgs] + [f"passage:{p.pid}" for p in ps]
        prev: list[Step] = []
        for rnd in range(1, cfg.switches.debate_rounds + 1):
            ids = [ctx.trace.new_id("s") for _ in agents]
            def go(a_sid, prev=prev, rnd=rnd):
                a, sid = a_sid
                peers = [s for s in prev if s.agent_id != a.agent_id]
                own = [s for s in prev if s.agent_id == a.agent_id]
                block = inputs + pblock
                if prev:
                    block += ("\nYour previous answer:\n" + (own[0].output_text if own else "(none)") +
                              "\n\nOther agents' answers:\n" +
                              "\n\n".join(f"[Agent {i + 1}] {s.output_text}" for i, s in enumerate(peers)) +
                              "\n\nConsider the other agents' answers and give your updated answer.\n")
                refs = base_refs + [f"step:{s.step_id}" for s in prev]
                return agent_respond(ctx, a, st, f"debate_r{rnd}", block, refs, f"r{rnd}", sid)
            with ThreadPoolExecutor(max_workers=len(agents)) as ex:
                cur = list(ex.map(go, zip(agents, ids)))
            for s in cur:
                ctx.trace.add_step(s)
            extract_claims(ctx, st.description, cur, {}, "(none)", instrument=True)
            prev = cur
        parsed = [parse_answer(s.output_text)[0] for s in prev]
        votes = Counter(normalize_answer(a) for a in parsed if a)
        if votes:
            top = max(votes.values())
            win_i = next(i for i, a in enumerate(parsed) if a and votes[normalize_answer(a)] == top)
        else:
            win_i = 0
        winners[st.sid] = prev[win_i]
        ctx.trace.event("debate_majority", subtask=st.sid, votes=dict(votes), winner_step=prev[win_i].step_id)
        final_answer, final_step = parsed[win_i], prev[win_i]
    ans, text = _answer_text(final_step.output_text)
    out = {"answer": ans, "final_text": text, "abstained": not ans.strip(),
           "final_claim_ids": list(final_step.claim_ids), "memory_log": [], "clusters": [],
           "final_step": final_step.step_id}
    return out


HALU_JUDGE_USER = """Knowledge: {knowledge}
Question: {question}
Answer: {answer}

Does the answer contain hallucinated (unsupported or false) information? Return only JSON: {{"hallucinated": true | false}}"""


def run_halueval_judge(ctx: QueryContext) -> dict:
    """Single-LLM recognition baseline for the HaluEval detector comparison (Section VI-D)."""
    agent = _single_agent(ctx)
    obj = ctx.call_json(agent.model, P.ROLE_PROMPTS["assistant"],
                        HALU_JUDGE_USER.format(knowledge=ctx.ex.gold_evidence[0], question=ctx.ex.question,
                                               answer=ctx.ex.meta["shown_answer"]), "halueval:judge")
    pred = bool(obj.get("hallucinated")) if isinstance(obj, dict) else None
    return {"answer": str(pred), "final_text": "", "abstained": pred is None, "final_claim_ids": [],
            "predicted_hallucinated": pred, "gold_hallucinated": ctx.ex.meta["is_hallucinated"],
            "memory_log": [], "clusters": []}


def run_baseline(ctx: QueryContext) -> dict:
    s = ctx.cfg.switches.system
    if s == "b1":
        return run_single(ctx, retrieval=False)
    if s == "b2":
        return run_single(ctx, retrieval=True)
    if s == "b3":
        return run_self_consistency(ctx)
    if s == "b4":
        return run_cove(ctx)
    if s == "b5":
        return run_debate(ctx)
    raise ValueError(s)
