"""LLM-backed components: Task Decomposer, Agents, CVM claim extraction, Arbiter, Synthesizer.

`QueryContext` bundles per-query resources so every call is cached, charged and traced."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

from . import prompts as P
from .config import RunConfig
from .jsonutil import parse_answer, parse_json
from .llm import LLMClient, Ledger
from .nli import CountingNLI
from .trace import Trace
from .types import AgentSpec, Claim, Example, ModelSpec, Passage, Step, SubTask


@dataclass
class QueryContext:
    cfg: RunConfig
    llm: LLMClient
    nli: CountingNLI            # system NLI (counted towards cost)
    encoder: object
    retriever: object
    ex: Example
    trace: Trace
    ledger: Ledger              # system cost
    instr_ledger: Ledger        # measurement-only calls (e.g. claim extraction for baseline traces)
    errors: list[str] = field(default_factory=list)

    def rng(self, *parts) -> random.Random:
        h = hashlib.sha256("|".join([str(self.cfg.seed), str(self.cfg.run_index), self.ex.qid, *map(str, parts)]).encode())
        return random.Random(int(h.hexdigest()[:16], 16))

    def call(self, model: ModelSpec, system: str, user: str, component: str, *, stochastic: bool = False,
             tag_parts: tuple = (), instrument: bool = False, max_tokens: int | None = None) -> str:
        temp = self.cfg.draft_temperature if stochastic else 0.0
        tag = "|".join([f"seed{self.cfg.seed}", f"run{self.cfg.run_index}", self.ex.qid, *map(str, tag_parts)]) if stochastic else ""
        res = self.llm.complete(model, system, user, temperature=temp, max_tokens=max_tokens or self.cfg.max_tokens,
                                tag=tag, ledger=self.instr_ledger if instrument else self.ledger, component=component)
        return res.text

    def call_json(self, model, system, user, component, *, instrument=False, max_tokens=None):
        text = self.call(model, system, user, component, instrument=instrument, max_tokens=max_tokens)
        try:
            return parse_json(text)
        except ValueError:
            # One retry with an explicit reminder; it has its own cache key because the prompt differs.
            text2 = self.call(model, system, user + "\n\nReturn valid JSON only, with no other text.",
                              component + ":retry", instrument=instrument, max_tokens=max_tokens)
            try:
                return parse_json(text2)
            except ValueError:
                self.errors.append(f"json_parse_failed:{component}")
                self.trace.event("error", kind="json_parse_failed", component=component, text=text2[:500])
                return None


# ---------------- Task Decomposer ----------------
def decompose_task(ctx: QueryContext) -> list[SubTask]:
    q = ctx.ex.question
    obj = ctx.call_json(ctx.cfg.utility_model, P.DECOMPOSE_SYSTEM,
                        P.DECOMPOSE_USER.format(question=q, max_subtasks=ctx.cfg.max_subtasks), "decomposer")
    subtasks: list[SubTask] = []
    try:
        seen: dict[str, str] = {}
        for i, st in enumerate(obj["subtasks"][: ctx.cfg.max_subtasks]):
            sid = f"t{i + 1}"
            seen[str(st.get("id", sid))] = sid
            deps = [seen[str(d)] for d in st.get("depends_on", []) if str(d) in seen and seen[str(d)] != sid]
            desc = str(st["description"]).strip()
            if desc:
                subtasks.append(SubTask(sid, desc, sorted(set(deps))))
    except (TypeError, KeyError, AttributeError):
        subtasks = []
    if not subtasks:
        ctx.errors.append("decomposition_fallback")
        subtasks = [SubTask("t1", q, [])]
    ctx.trace.event("decomposition", subtasks=[st.__dict__ for st in subtasks])
    return subtasks


# ---------------- Agents ----------------
def render_entries(entries, prefix="E") -> tuple[str, dict[str, str]]:
    if not entries:
        return "(none)", {}
    lines, mapping = [], {}
    for i, e in enumerate(entries, 1):
        lab = f"{prefix}{i}"
        mapping[lab] = e.id
        q = f"; qualifiers: {', '.join(e.qualifiers)}" if e.qualifiers else ""
        if e.label == "Unverified":      # B6 / unguarded writes carry no confidence
            lines.append(f"[{lab}] {e.text}" + (f" ({q[2:]})" if q else ""))
        else:
            lines.append(f"[{lab}] {e.text} (confidence {e.conf:.2f}; {e.label}{q})")
    return "\n".join(lines), mapping


def render_passages(passages: list[Passage]) -> str:
    return "\n".join(f"[P{i}] {p.as_text()}" for i, p in enumerate(passages, 1))


def agent_respond(ctx: QueryContext, agent: AgentSpec, subtask: SubTask, phase: str, inputs_block: str,
                  context_refs: list[str], tag: str, step_id: str) -> Step:
    """Returns an un-registered Step; the caller registers steps in a fixed order (deterministic traces)."""
    user = P.DRAFT_USER.format(question=ctx.ex.question, subtask=subtask.description, inputs=inputs_block,
                               fmt=P.ANSWER_FORMAT)
    text = ctx.call(agent.model, P.ROLE_PROMPTS[agent.role], user, f"agent:{phase}", stochastic=True,
                    tag_parts=(subtask.sid, agent.agent_id, phase, tag))
    return Step(step_id=step_id, phase=phase, subtask=subtask.sid, agent_id=agent.agent_id,
                role=agent.role, model=agent.model.key(), context_refs=context_refs, output_text=text)


def agent_revise(ctx: QueryContext, agent: AgentSpec, subtask: SubTask, draft: Step, memory_block: str,
                 feedback: list[str], context_refs: list[str], step_id: str) -> Step:
    fb = "\n".join(f"- {f}" for f in feedback) if feedback else "(none)"
    user = P.REVISE_USER.format(question=ctx.ex.question, subtask=subtask.description, draft=draft.output_text,
                                memory=memory_block, feedback=fb, fmt=P.ANSWER_FORMAT)
    text = ctx.call(agent.model, P.ROLE_PROMPTS[agent.role], user, "agent:revision", stochastic=True,
                    tag_parts=(subtask.sid, agent.agent_id, "revision"))
    return Step(step_id=step_id, phase="revision", subtask=subtask.sid, agent_id=agent.agent_id,
                role=agent.role, model=agent.model.key(), context_refs=context_refs, output_text=text)


# ---------------- CVM: claim decomposition, Eq. (1) ----------------
def extract_claims(ctx: QueryContext, subtask_desc: str, steps: list[Step], premise_map: dict[str, str],
                   premises_block: str, *, instrument: bool = False, model: ModelSpec | None = None,
                   phase: str | None = None) -> list[Claim]:
    """Batched decomposition of several responses into atomic claims with qualifiers and premise links."""
    if not steps:
        return []
    labels = [f"R{i}" for i in range(1, len(steps) + 1)]
    responses = "\n\n".join(f"[{lab}] {s.output_text}" for lab, s in zip(labels, steps))
    schema = ", ".join(f'"{lab}": [{{"claim": "...", "qualifiers": [], "premises": []}}]' for lab in labels)
    obj = ctx.call_json(model or ctx.cfg.utility_model, P.EXTRACT_SYSTEM,
                        P.EXTRACT_USER.format(subtask=subtask_desc, premises=premises_block,
                                              responses=responses, schema=schema),
                        "instr:extract" if instrument else "cvm:extract", instrument=instrument,
                        max_tokens=max(1024, 700 * len(steps)))
    out: list[Claim] = []
    if not isinstance(obj, dict):
        return out
    for lab, step in zip(labels, steps):
        items = obj.get(lab, []) or []
        for it in items:
            if isinstance(it, str):
                it = {"claim": it}
            if not isinstance(it, dict) or not str(it.get("claim", "")).strip():
                continue
            prem = [premise_map[p] for p in it.get("premises", []) or [] if isinstance(p, str) and p in premise_map]
            c = Claim(cid=ctx.trace.new_id("c"), text=str(it["claim"]).strip(), agent_id=step.agent_id,
                      subtask=step.subtask, step_id=step.step_id, phase=phase or step.phase,
                      qualifiers=[str(q) for q in (it.get("qualifiers") or [])], premises=sorted(set(prem)))
            ctx.trace.add_claim(c)
            step.claim_ids.append(c.cid)
            out.append(c)
    return out


def extract_text_claims(llm: LLMClient, model: ModelSpec, question: str, text: str, ledger: Ledger,
                        component: str = "eval:extract") -> list[dict]:
    """Decompose a single text (used for the final output of every system, identically)."""
    if not text.strip():
        return []
    schema = '"R1": [{"claim": "...", "qualifiers": [], "premises": []}]'
    user = P.EXTRACT_USER.format(subtask=question, premises="(none)", responses=f"[R1] {text}", schema=schema)
    res = llm.complete(model, P.EXTRACT_SYSTEM, user, temperature=0.0, max_tokens=1024, ledger=ledger,
                       component=component)
    try:
        obj = parse_json(res.text)
    except ValueError:
        res = llm.complete(model, P.EXTRACT_SYSTEM, user + "\n\nReturn valid JSON only, with no other text.",
                           temperature=0.0, max_tokens=1024, ledger=ledger, component=component + ":retry")
        obj = parse_json(res.text)
    items = obj.get("R1", []) if isinstance(obj, dict) else []
    return [{"text": (it["claim"] if isinstance(it, dict) else str(it)).strip(),
             "qualifiers": (it.get("qualifiers") or []) if isinstance(it, dict) else []}
            for it in items if (isinstance(it, dict) and str(it.get("claim", "")).strip()) or isinstance(it, str)]


# ---------------- Arbiter (Section IV-G) ----------------
def arbitrate(ctx: QueryContext, options: list[str], passages: list[Passage], key: str) -> tuple[int | None, str, Passage | None]:
    """Returns (index of chosen option or None, verdict, cited passage). Options are shown in random order
    without agent identities, rationales or confidence."""
    rng = ctx.rng("arbiter", key)
    order = list(range(len(options)))
    rng.shuffle(order)
    passages = list(passages)
    rng.shuffle(passages)
    letters = [chr(ord("A") + i) for i in range(len(order))]
    claims_block = "\n".join(f"[{letters[i]}] {options[j]}" for i, j in enumerate(order))
    opt_str = " | ".join(f'"{l}"' for l in letters) + ' | "neither" | "insufficient"'
    obj = ctx.call_json(ctx.cfg.arbiter_model, P.ARBITER_SYSTEM,
                        P.ARBITER_USER.format(question=ctx.ex.question, claims=claims_block,
                                              passages=render_passages(passages) or "(no passages retrieved)",
                                              options=opt_str), "arbiter")
    if not isinstance(obj, dict):
        return None, "parse_error", None
    verdict = str(obj.get("verdict", "")).strip().strip('"')
    cited = None
    pid = str(obj.get("passage") or "")
    if pid.upper().startswith("P") and pid[1:].isdigit() and 1 <= int(pid[1:]) <= len(passages):
        cited = passages[int(pid[1:]) - 1]
    if verdict in letters:
        return order[letters.index(verdict)], verdict, cited
    return None, verdict.lower(), cited


# ---------------- Synthesizer ----------------
def synthesize(ctx: QueryContext, entries, responses: list[Step]) -> dict:
    mem_block, mem_map = render_entries(entries, prefix="M")
    resp_block = "\n\n".join(f"[R{i}] {s.output_text}" for i, s in enumerate(responses, 1)) or "(none)"
    obj = ctx.call_json(ctx.cfg.utility_model, P.SYNTH_SYSTEM,
                        P.SYNTH_USER.format(question=ctx.ex.question, memory=mem_block, responses=resp_block),
                        "synthesizer")
    if not isinstance(obj, dict):
        return {"answer": "UNRESOLVED", "answer_claim": "", "statement": "", "support": []}
    out = {k: str(obj.get(k, "") or "") for k in ("answer", "answer_claim", "statement")}
    sup = obj.get("support") or []
    out["support"] = [mem_map[x] for x in sup if isinstance(x, str) and x in mem_map] if isinstance(sup, list) else []
    return out


def summarize_entries(ctx: QueryContext, entries) -> str:
    """A3 (-SML): verified claims handed downstream as a free-text summary."""
    if not entries:
        return ""
    findings = "\n".join(f"- {e.text}" for e in entries)
    return ctx.call(ctx.cfg.utility_model, "You write concise summaries.", P.SUMMARY_USER.format(findings=findings),
                    "a3:summary")
