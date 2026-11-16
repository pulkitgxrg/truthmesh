"""CV-SM for a single query (Algorithm 1), with the Table IV switches (A1-A8, B6) on the same code path."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

import numpy as np

from . import equations as eq
from .components import (QueryContext, agent_respond, agent_revise, arbitrate, decompose_task, extract_claims,
                         render_entries, summarize_entries, synthesize)
from .memory import ACTIVE, RETRACTED, SUPERSEDED, Entry, SharedMemory
from .nli import cosine_matrix
from .types import AgentSpec, Claim, Step, SubTask

UNVERIFIED = "Unverified"


@dataclass
class ClusterRec:
    key: str
    subtask: str
    phase: str
    member_ids: list[str]
    rep_id: str
    rep_text: str
    S: list[str]
    S_ind: list[str]
    label: str
    status: str = "pending"          # pending | resolved | rejected | unresolved | written | not_written
    A: float = 0.0
    M: float = 0.0
    G: float = 0.0
    conf: float = 0.0
    conf_star: float = 0.0
    premises: list[str] = field(default_factory=list)
    entry_id: str | None = None
    evidence: list[dict] = field(default_factory=list)
    reason: str = ""


def topo_order(subtasks: list[SubTask]) -> list[SubTask]:
    done, out, pending = set(), [], list(subtasks)
    while pending:
        progressed = False
        for st in list(pending):
            if all(d in done for d in st.depends_on):
                out.append(st); done.add(st.sid); pending.remove(st); progressed = True
        if not progressed:      # cannot happen after decompose_task's validation; keep order as-is
            out.extend(pending)
            break
    return out


def assign_agents(pool: list[AgentSpec], k: int, j: int) -> list[AgentSpec]:
    if len(pool) == k:
        return list(pool)
    return [pool[(j + i) % len(pool)] for i in range(k)]


class CVSM:
    def __init__(self, ctx: QueryContext):
        self.ctx = ctx
        self.cfg, self.sw, self.p = ctx.cfg, ctx.cfg.switches, ctx.cfg.params
        self.mem = SharedMemory(ctx.trace)
        self.rho = ctx.cfg.effective_rho()
        self.clusters: list[ClusterRec] = []
        self.extra_context_text: dict[str, list[str]] = {}     # step_id -> texts not addressable by id (A3)
        self._entry_text: dict[str, str] = {}

    # ---------- helpers ----------
    def emb(self, texts):
        return self.ctx.encoder.embed(list(texts))

    def nli(self, pairs):
        return self.ctx.nli.scores(pairs)

    def context_texts(self, step: Step) -> list[str]:
        """X_ij: claims available to the agent apart from the query and sub-task (Section III-B)."""
        out = list(self.extra_context_text.get(step.step_id, []))
        for ref in step.context_refs:
            kind, rid = ref.split(":", 1)
            if kind == "entry":
                out.append(self._entry_text[rid])
            elif kind == "step":
                out.extend(self.ctx.trace.claims[c].text for c in self.ctx.trace.step(rid).claim_ids)
            # "passage:" refs are external evidence, not agent claims, so they are not exposure (III-B)
        return out

    def top_entries(self, text: str, r: int, min_sim: float | None) -> list[tuple[Entry, float]]:
        act = self.mem.active()
        if not act:
            return []
        sims = cosine_matrix(self.emb([text]), self.emb([e.text for e in act]))[0]
        cand = [(e, float(s)) for e, s in zip(act, sims) if min_sim is None or s > min_sim]
        cand.sort(key=lambda x: -x[1])
        return cand[:r]

    def memory_score(self, text: str) -> tuple[float, list[dict]]:
        """Eq. (12) over R(c): the r most similar active entries with similarity above theta_sim."""
        R = self.top_entries(text, self.p.r, self.p.theta_sim)
        if not R:
            return 0.0, []
        sc = self.nli([(e.text, text) for e, _ in R])
        items = [(e.conf, s, x.ent, x.con) for (e, s), x in zip(R, sc)]
        return eq.memory_term(items), [{"entry": e.id, "sim": s, "ent": x.ent, "con": x.con}
                                       for (e, s), x in zip(R, sc)]

    def grounding(self, text: str, passages=None) -> tuple[float, list[dict]]:
        """Eq. (13) over Psi(c), the top-p retrieved passages (or a cited passage for Resolved)."""
        psi = self.ctx.retriever(text, self.p.p) if passages is None else passages
        if not psi:
            return 0.0, []
        sc = self.nli([(ps.as_text(), text) for ps in psi])
        return eq.grounding_term([(x.ent, x.con) for x in sc]), [
            {"pid": ps.pid, "ent": x.ent, "con": x.con} for ps, x in zip(psi, sc)]

    def entailed_by_memory(self, text: str) -> str | None:
        R = self.top_entries(text, self.p.r, None)
        if not R:
            return None
        for (e, _), x in zip(R, self.nli([(e.text, text) for e, _ in R])):
            if x.ent >= self.p.theta_e:
                return e.id
        return None

    def premise_confs(self, members: list[Claim], I_j: list[Entry]) -> tuple[list[float], list[str]]:
        """Eq. (15): premises recorded by the claims; claims recording none in a dependent sub-task are
        capped over all of I_j. Premises that are no longer active contribute Conf = 0."""
        if not self.sw.premise_cap:
            return [], []
        ids_I = {e.id for e in I_j}
        prem = sorted({p for m in members for p in m.premises if p in ids_I})
        if not prem and I_j:
            prem = sorted(ids_I)
        confs = [self.mem.get(p).conf if self.mem.get(p).status == ACTIVE else 0.0 for p in prem]
        return confs, prem

    def _rule_label(self, label: str) -> str:
        return eq.SINGLETON if label == "Singleton-grounded" else label

    def reverify(self, e: Entry) -> bool:
        conf = eq.confidence(e.A, e.M, e.G, self.p.alpha, self.p.beta, self.p.gamma)
        confs = [self.mem.get(p).conf if self.mem.get(p).status == ACTIVE else 0.0 for p in e.premises]
        cs = eq.conf_star(conf, confs) if self.sw.premise_cap else conf
        ok = eq.write_rule(cs, e.M, e.G, self._rule_label(e.label), self.p.tau, self.p.delta)
        self.ctx.trace.event("reverify", entry_id=e.id, conf_star=cs, ok=ok)
        return ok

    def _retire(self, entry_id: str, status: str, note: str):
        self.mem.set_status(entry_id, status, note)
        if self.sw.retraction:
            self.mem.retract_transitively(entry_id, self.reverify)

    def write(self, text: str, conf_star: float, label: str, *, A, M, G, support, evidence, premises,
              qualifiers, subtask, source_claims, source_steps) -> str | None:
        """Append one entry, applying the supersession rule (Eq. 17) to contradicted active entries."""
        if self.sw.r2:
            act = self.mem.active()
            if act:
                sims = cosine_matrix(self.emb([text]), self.emb([e.text for e in act]))[0]
                cand = [e for e, s in zip(act, sims) if s >= self.p.theta_sim]
                pairs = [(e.text, text) for e in cand] + [(text, e.text) for e in cand]
                sc = self.nli(pairs)
                n = len(cand)
                contra = [e for i, e in enumerate(cand) if eq.contradicts(sc[i].con, sc[n + i].con, self.p.theta_c)]
                to_supersede = []
                for e in contra:
                    dec = eq.supersession(conf_star, e.conf, self.p.Delta)
                    self.ctx.trace.event("supersession_check", new_text=text, entry_id=e.id, decision=dec,
                                         conf_star_c=conf_star, conf_e=e.conf)
                    if dec == "supersede":
                        to_supersede.append(e)
                        continue
                    passages = _dedupe(self.ctx.retriever(text, self.p.p) + self.ctx.retriever(e.text, self.p.p))
                    idx, verdict, cited = arbitrate(self.ctx, [text, e.text], passages, key=f"sup|{e.id}|{text}")
                    ok = idx is not None and cited is not None and \
                        self.nli([(cited.as_text(), [text, e.text][idx])])[0].ent >= self.p.theta_e
                    self.ctx.trace.event("arbitration", context="supersession", options=[text, e.text],
                                         verdict=verdict, chosen=idx, cited=cited.pid if cited else None, accepted=ok)
                    if ok and idx == 0:
                        to_supersede.append(e)
                    else:
                        self.ctx.trace.event("write_blocked", text=text, reason="supersession arbitration", entry_id=e.id)
                        return None
                for e in to_supersede:
                    self._retire(e.id, SUPERSEDED, note=f"superseded by new claim: {text}")
        e = Entry(id=self.mem.new_id(), text=text, embedding=[float(x) for x in self.emb([text])[0]],
                  conf=conf_star, label=label, support=support, evidence=evidence, premises=premises,
                  qualifiers=qualifiers, subtask=subtask, A=A, M=M, G=G, source_claims=source_claims,
                  source_steps=source_steps)
        self._entry_text[e.id] = text
        self.mem.append(e)
        return e.id

    # ---------- alignment, Eq. (9), (2), (10) ----------
    def align(self, claims: list[Claim], steps_by_agent: dict[str, Step], subtask: str, phase: str) -> tuple[list[ClusterRec], list[tuple[int, int]]]:
        n = len(claims)
        if n == 0:
            return [], []
        E = self.emb([c.text for c in claims])
        S = cosine_matrix(E, E)
        cand = [(i, j) for i in range(n) for j in range(i + 1, n) if S[i, j] >= self.p.theta_sim]
        sc = self.nli([(claims[i].text, claims[j].text) for i, j in cand] +
                      [(claims[j].text, claims[i].text) for i, j in cand])
        m = len(cand)
        eq_edges, con_pairs = [], []
        for t, (i, j) in enumerate(cand):
            fwd, bwd = sc[t], sc[m + t]
            if eq.equivalent(fwd.ent, bwd.ent, self.p.theta_e):
                eq_edges.append((i, j))
            elif eq.contradicts(fwd.con, bwd.con, self.p.theta_c):
                con_pairs.append((i, j))
        comps = eq.connected_components(n, eq_edges)
        cid_of = {i: k for k, comp in enumerate(comps) for i in comp}
        contra = sorted({tuple(sorted((cid_of[i], cid_of[j]))) for i, j in con_pairs if cid_of[i] != cid_of[j]})
        contested = {a for pair in contra for a in pair}
        recs = []
        for k, comp in enumerate(comps):
            members = [claims[i] for i in comp]
            if len(comp) > 1:
                sub = S[np.ix_(comp, comp)]
                rep = members[int(np.argmax(sub.sum(axis=1)))]
            else:
                rep = members[0]
            agents = sorted({c.agent_id for c in members})
            exposed = set()
            for a in agents:
                X = self.context_texts(steps_by_agent[a]) if a in steps_by_agent else []
                if X:
                    xs = self.nli([(x, mm.text) for x in X for mm in members])
                    if any(s.ent >= self.p.theta_e for s in xs):
                        exposed.add(a)
            S_set, S_ind = eq.support_sets(agents, exposed)
            label = eq.cluster_label(k in contested, len(S_ind))
            recs.append(ClusterRec(key=f"{subtask}|{phase}|{k}", subtask=subtask, phase=phase,
                                   member_ids=[c.cid for c in members], rep_id=rep.cid, rep_text=rep.text,
                                   S=sorted(S_set), S_ind=sorted(S_ind), label=label))
        for r in recs:
            self.ctx.trace.event("cluster", **{k: v for k, v in asdict(r).items() if k in
                                               ("key", "member_ids", "rep_text", "S", "S_ind", "label")})
        return recs, contra

    # ---------- resolution (Section IV-G) ----------
    def resolve_contested(self, recs: list[ClusterRec], contra: list[tuple[int, int]]):
        groups = eq.connected_components(len(recs), contra)
        for g in groups:
            if len(g) < 2:
                continue
            opts = [recs[i] for i in g]
            if self.sw.arbitration == "majority":                     # A5
                sizes = [len(r.S) for r in opts]
                best = max(sizes)
                if sizes.count(best) == 1:
                    for r in opts:
                        r.status = "resolved" if len(r.S) == best else "rejected"
                        if r.status == "resolved":
                            r.label = eq.RESOLVED
                else:
                    for r in opts:
                        r.status = "unresolved"
                self.ctx.trace.event("arbitration", context="majority", options=[r.rep_text for r in opts],
                                     sizes=sizes, outcome=[r.status for r in opts])
                continue
            passages = _dedupe([ps for r in opts for ps in self.ctx.retriever(r.rep_text, self.p.p)])
            idx, verdict, cited = arbitrate(self.ctx, [r.rep_text for r in opts], passages, key=opts[0].key)
            accepted = False
            if idx is not None and cited is not None:
                accepted = self.nli([(cited.as_text(), opts[idx].rep_text)])[0].ent >= self.p.theta_e
            for i, r in enumerate(opts):
                if accepted:
                    r.status = "resolved" if i == idx else "rejected"
                elif verdict == "neither":
                    r.status = "rejected"
                else:
                    r.status = "unresolved"
            if accepted:
                win = opts[idx]
                win.label = eq.RESOLVED
                win.G, win.evidence = self.grounding(win.rep_text, passages=[cited])
                win.reason = "evidence"
            self.ctx.trace.event("arbitration", context="contested", options=[r.rep_text for r in opts],
                                 verdict=verdict, chosen=idx, cited=cited.pid if cited else None,
                                 accepted=accepted, outcome=[r.status for r in opts])

    # ---------- scoring and write, Eqs. (11)-(16) ----------
    def score_and_write(self, rec: ClusterRec, members: list[Claim], I_j: list[Entry], agents: list[AgentSpec]):
        k = self.cfg.k
        rec.A = eq.agreement(len(rec.S_ind), k, self.rho)
        rec.M, mem_details = self.memory_score(rec.rep_text)
        if not (rec.label == eq.RESOLVED and rec.reason == "evidence"):
            rec.G, rec.evidence = self.grounding(rec.rep_text)
        rec.conf = eq.confidence(rec.A, rec.M, rec.G, self.p.alpha, self.p.beta, self.p.gamma)
        confs, rec.premises = self.premise_confs(members, I_j)
        rec.conf_star = eq.conf_star(rec.conf, confs)
        if rec.status in ("rejected", "unresolved"):
            allowed = False
        elif rec.label == eq.SINGLETON and self.sw.singletons == "reject":     # A4
            allowed = False
        else:
            allowed = eq.write_rule(rec.conf_star, rec.M, rec.G, rec.label, self.p.tau, self.p.delta)
        if not self.sw.r2:                                                     # A2: every claim written
            allowed = True
        self.ctx.trace.event("score", key=rec.key, rep_text=rec.rep_text, label=rec.label, status=rec.status,
                             A=rec.A, M=rec.M, G=rec.G, conf=rec.conf, conf_star=rec.conf_star,
                             premises=rec.premises, write_rule=allowed, memory_details=mem_details,
                             evidence=rec.evidence)
        if not allowed:
            rec.status = rec.status if rec.status in ("rejected", "unresolved") else "not_written"
            return
        role = {a.agent_id: a for a in agents}
        label = "Singleton-grounded" if rec.label == eq.SINGLETON else rec.label
        quals = sorted({q for m in members for q in m.qualifiers})
        eid = self.write(rec.rep_text, rec.conf_star, label, A=rec.A, M=rec.M, G=rec.G,
                         support=[{"agent": a, "role": role[a].role, "model": role[a].model.key()}
                                  for a in rec.S_ind if a in role],
                         evidence=rec.evidence, premises=rec.premises, qualifiers=quals, subtask=rec.subtask,
                         source_claims=[m.cid for m in members], source_steps=sorted({m.step_id for m in members}))
        rec.entry_id = eid
        rec.status = "written" if eid else "not_written"

    # ---------- main loop ----------
    def run(self) -> dict:
        ctx, sw = self.ctx, self.sw
        subtasks = topo_order(decompose_task(ctx)) if sw.use_decomposer else [SubTask("t1", ctx.ex.question, [])]
        pool = self.cfg.agent_pool()
        final_responses: list[Step] = []
        for j, st in enumerate(subtasks):
            agents = assign_agents(pool, self.cfg.k, j)
            parents = set(st.depends_on)
            I_j = self.mem.active(parents) if parents else []
            if sw.structured_handoff:
                prem_block, prem_map = render_entries(I_j)
                inputs_block = f"\nVerified inputs from earlier sub-tasks:\n{prem_block}\n" if I_j else ""
                extra = []
            else:                                                              # A3
                summary = summarize_entries(ctx, I_j)
                prem_block, prem_map = "(none)", {}
                inputs_block = f"\nFindings from earlier sub-tasks:\n{summary}\n" if summary else ""
                extra = [summary] if summary else []
            base_refs = [f"entry:{e.id}" for e in I_j]
            if sw.draft_retrieval:          # same retriever, corpus and p as B2/B5 (Section V-B)
                ps = self.ctx.retriever(st.description, self.p.p)
                if ps:
                    from .components import render_passages
                    inputs_block += f"\nRetrieved passages:\n{render_passages(ps)}\n"
                    base_refs = base_refs + [f"passage:{x.pid}" for x in ps]
            if not sw.verification:
                final_responses += self._unguarded_subtask(st, agents, inputs_block, base_refs)
                continue
            drafts = self._draft(st, agents, inputs_block, base_refs, extra)
            claims = extract_claims(ctx, st.description, drafts, prem_map, prem_block)
            recs, contra = self.align(claims, {s.agent_id: s for s in drafts}, st.sid, "draft")
            self.resolve_contested(recs, contra)
            byid = ctx.trace.claims
            for rec in recs:
                self.score_and_write(rec, [byid[c] for c in rec.member_ids], I_j, agents)
            self.clusters += recs
            if sw.revision:
                final_responses += self._revise(st, agents, drafts, recs, I_j, prem_map, prem_block)
            else:
                final_responses += drafts
        return self._synthesize(final_responses)

    def _draft(self, st, agents, inputs_block, base_refs, extra) -> list[Step]:
        ctx = self.ctx
        ids = [ctx.trace.new_id("s") for _ in agents]
        if self.sw.r1:                                                         # R1: blind, parallel drafts
            with ThreadPoolExecutor(max_workers=len(agents)) as ex:
                steps = list(ex.map(lambda a_id: agent_respond(ctx, a_id[0], st, "draft", inputs_block, base_refs,
                                                                "", a_id[1]), zip(agents, ids)))
            for s in steps:
                self.extra_context_text[s.step_id] = list(extra)
                ctx.trace.add_step(s)
            return steps
        steps = []                                                             # A1: read memory and peers first
        for a, sid in zip(agents, ids):
            all_mem = self.mem.active()
            mem_block, _ = render_entries(all_mem, prefix="M")
            peers = "\n\n".join(f"[{p.agent_id}] {p.output_text}" for p in steps) or "(none)"
            block = (inputs_block + f"\nShared memory (all entries):\n{mem_block}\n"
                     f"\nOther agents' drafts for this sub-task:\n{peers}\n")
            refs = sorted(set(base_refs) | {f"entry:{e.id}" for e in all_mem}) + [f"step:{p.step_id}" for p in steps]
            s = agent_respond(ctx, a, st, "draft", block, refs, "", sid)
            self.extra_context_text[s.step_id] = list(extra)
            ctx.trace.add_step(s)
            steps.append(s)
        # Peer drafts' claims are needed to compute exposure; they are extracted together after drafting.
        return steps

    def _unguarded_subtask(self, st, agents, inputs_block, base_refs) -> list[Step]:
        """B6: agents read the whole memory and earlier peers; every claim is written unverified."""
        ctx, steps = self.ctx, []
        for a in agents:
            all_mem = self.mem.active()
            mem_block, mem_map = render_entries(all_mem, prefix="E")
            peers = "\n\n".join(f"[{p.agent_id}] {p.output_text}" for p in steps) or "(none)"
            block = inputs_block + f"\nShared memory (all entries):\n{mem_block}\n\nOther agents' drafts for this sub-task:\n{peers}\n"
            refs = [f"entry:{e.id}" for e in all_mem] + [f"step:{p.step_id}" for p in steps]
            s = agent_respond(ctx, a, st, "draft", block, refs, "", ctx.trace.new_id("s"))
            ctx.trace.add_step(s)
            steps.append(s)
            for c in extract_claims(ctx, st.description, [s], mem_map, mem_block):
                eid = self.write(c.text, 0.0, UNVERIFIED, A=0.0, M=0.0, G=0.0,
                                 support=[{"agent": a.agent_id, "role": a.role, "model": a.model.key()}],
                                 evidence=[], premises=c.premises, qualifiers=c.qualifiers, subtask=st.sid,
                                 source_claims=[c.cid], source_steps=[s.step_id])
                self.clusters.append(ClusterRec(key=f"{st.sid}|b6|{c.cid}", subtask=st.sid, phase="draft",
                                                member_ids=[c.cid], rep_id=c.cid, rep_text=c.text, S=[a.agent_id],
                                                S_ind=[], label=UNVERIFIED, status="written", entry_id=eid))
        return steps

    def _revise(self, st, agents, drafts, recs, I_j, prem_map, prem_block) -> list[Step]:
        ctx = self.ctx
        read = self.mem.active({st.sid} | set(st.depends_on))
        mem_block, _ = render_entries(read, prefix="M")
        status_of = {}
        for r in recs:
            for c in r.member_ids:
                status_of[c] = r
        def feedback(draft: Step) -> list[str]:
            out = []
            for cid in draft.claim_ids:
                r = status_of.get(cid)
                if r is None or r.status == "written":
                    continue
                why = {"rejected": "contested and rejected by evidence", "unresolved": "contested and unresolved",
                       "not_written": ("unverified (no independent support or evidence)" if r.label == eq.SINGLETON
                                       else "failed verification")}.get(r.status, r.status)
                out.append(f'"{ctx.trace.claims[cid].text}" - {why}')
            return out
        ids = [ctx.trace.new_id("s") for _ in agents]
        by_agent = {d.agent_id: d for d in drafts}
        refs_mem = [f"entry:{e.id}" for e in read]
        with ThreadPoolExecutor(max_workers=len(agents)) as ex:
            revs = list(ex.map(lambda a_id: agent_revise(ctx, a_id[0], st, by_agent[a_id[0].agent_id], mem_block,
                                                          feedback(by_agent[a_id[0].agent_id]),
                                                          [f"step:{by_agent[a_id[0].agent_id].step_id}"] + refs_mem,
                                                          a_id[1]), zip(agents, ids)))
        for s in revs:
            ctx.trace.add_step(s)
        new_claims = extract_claims(ctx, st.description, revs, prem_map, prem_block)
        for c in new_claims:
            hit = self.entailed_by_memory(c.text)
            if hit:
                ctx.trace.event("revision_claim_accepted", cid=c.cid, entry_id=hit)
                continue
            rec = ClusterRec(key=f"{st.sid}|revision|{c.cid}", subtask=st.sid, phase="revision", member_ids=[c.cid],
                             rep_id=c.cid, rep_text=c.text, S=[c.agent_id], S_ind=[c.agent_id], label=eq.SINGLETON)
            self.score_and_write(rec, [c], I_j, agents)
            self.clusters.append(rec)
        return revs

    def _synthesize(self, responses: list[Step]) -> dict:
        ctx = self.ctx
        active = self.mem.active()
        syn = synthesize(ctx, active, responses)
        text = " ".join(x for x in (syn["answer_claim"], syn["statement"]) if x).strip()
        step = Step(step_id=ctx.trace.new_id("s"), phase="synthesis", subtask="final", agent_id="synthesizer",
                    role="synthesizer", model=self.cfg.utility_model.key(),
                    context_refs=[f"entry:{e.id}" for e in active] + [f"step:{s.step_id}" for s in responses],
                    output_text=text)
        ctx.trace.add_step(step)
        unresolved = syn["answer"].strip().upper() in ("", "UNRESOLVED")
        if not self.sw.synth_filter:
            claims = extract_claims(ctx, ctx.ex.question, [step], {}, "(none)")
            final_ids = [c.cid for c in claims]
            return self._result(syn, text if not unresolved else "", unresolved, final_ids)
        claims = extract_claims(ctx, ctx.ex.question, [step], {}, "(none)")
        kept = []
        for c in claims:
            hit = self.entailed_by_memory(c.text)
            ctx.trace.event("final_filter", cid=c.cid, text=c.text, entailed_by=hit)
            if hit:
                kept.append(c)
        single = self.entailed_by_memory(syn["answer_claim"]) if syn["answer_claim"] else None
        multi = False
        if self.sw.multi_entry_synthesis and not single and syn["answer_claim"]:
            cited = [self.mem.get(i) for i in syn.get("support", []) if self.mem.get(i).status == ACTIVE]
            if cited:
                prem = " ".join(e.text for e in cited)
                multi = self.nli([(prem, syn["answer_claim"])])[0].ent >= self.p.theta_e
        ans_ok = (not unresolved) and bool(syn["answer_claim"]) and (single is not None or multi)
        ctx.trace.event("final_answer_check", answer_claim=syn["answer_claim"], entailed=ans_ok,
                        single_entry=single, multi_entry=multi, support=syn.get("support", []))
        if not ans_ok:
            return self._result(syn, "", True, [])
        parts = [syn["answer_claim"]] + [c.text for c in kept if c.text != syn["answer_claim"]]
        return self._result(syn, " ".join(parts), False, [c.cid for c in kept])

    def _result(self, syn, final_text, abstained, final_ids) -> dict:
        return {
            "answer": "" if abstained else syn["answer"],
            "final_text": final_text,
            "abstained": abstained,
            "final_claim_ids": final_ids,
            "synthesis": syn,
            "memory_log": self.mem.snapshot(),
            "clusters": [asdict(c) for c in self.clusters],
            "rho_hat_used": self.rho,
        }


def _dedupe(passages):
    seen, out = set(), []
    for p in passages:
        if p.pid not in seen:
            seen.add(p.pid); out.append(p)
    return out


def run_cvsm(ctx: QueryContext) -> dict:
    return CVSM(ctx).run()


def detect_hallucination(ctx: QueryContext) -> dict:
    """HaluEval QA (Section VI-D): the CVM checks a given response; a block decision on any of its claims is a
    prediction that the response is hallucinated. Each claim is scored as a single-agent assertion
    (Singleton, A = 1/k) with M and G against the item's knowledge passage, then Eq. (16) is applied."""
    sys_ = CVSM(ctx)
    text = ctx.ex.meta["shown_answer"]
    step = Step(step_id=ctx.trace.new_id("s"), phase="response", subtask="t1", agent_id="respondent",
                role="respondent", model="dataset", context_refs=[], output_text=text)
    ctx.trace.add_step(step)
    claims = extract_claims(ctx, ctx.ex.question, [step], {}, "(none)")
    blocked = []
    for c in claims:
        rec = ClusterRec(key=f"detect|{c.cid}", subtask="t1", phase="response", member_ids=[c.cid], rep_id=c.cid,
                         rep_text=c.text, S=[c.agent_id], S_ind=[c.agent_id], label=eq.SINGLETON)
        sys_.score_and_write(rec, [c], [], [])
        sys_.clusters.append(rec)
        blocked.append(rec.status != "written")
    pred = (not claims) or any(blocked)
    return {"answer": "hallucinated" if pred else "correct", "final_text": "", "abstained": False,
            "final_claim_ids": [], "predicted_hallucinated": pred, "gold_hallucinated": ctx.ex.meta["is_hallucinated"],
            "memory_log": sys_.mem.snapshot(), "clusters": [asdict(c) for c in sys_.clusters]}
