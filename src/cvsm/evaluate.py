"""Post-hoc evaluation of a run folder (Section V-E). Every metric is computed from logged records, so it
can be recomputed exactly from the cache.

Equivalence between claims for metrics (presence in the final answer, PD classes, CLR clusters, exposure)
uses Eq. (9) bidirectional entailment with the *labelling* NLI checkpoint at a fixed threshold, identically
for every system (DESIGN_NOTES: metric equivalence)."""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import equations as eq
from .datasets import exact_match, f1_score
from .nli import cosine_matrix


@dataclass
class MetricEquiv:
    nli: object
    encoder: object
    theta: float = 0.5
    theta_sim: float = 0.6

    def classes(self, texts: list[str]) -> list[list[int]]:
        n = len(texts)
        if n == 0:
            return []
        E = self.encoder.embed(texts)
        S = cosine_matrix(E, E)
        cand = [(i, j) for i in range(n) for j in range(i + 1, n)
                if texts[i] == texts[j] or S[i, j] >= self.theta_sim]
        edges = [(i, j) for i, j in cand if texts[i] == texts[j]]
        rest = [(i, j) for i, j in cand if texts[i] != texts[j]]
        sc = self.nli.scores([(texts[i], texts[j]) for i, j in rest] + [(texts[j], texts[i]) for i, j in rest])
        m = len(rest)
        edges += [(i, j) for t, (i, j) in enumerate(rest) if eq.equivalent(sc[t].ent, sc[m + t].ent, self.theta)]
        return eq.connected_components(n, edges)

    def entailed_by_text(self, text: str, hyps: list[str]) -> list[bool]:
        """"Present in the final answer": the final answer text (as one premise) entails the claim."""
        if not text.strip() or not hyps:
            return [False] * len(hyps)
        return [s.ent >= self.theta for s in self.nli.scores([(text, h) for h in hyps])]

    def entails_any(self, premises: list[str], hyps: list[str]) -> list[bool]:
        if not premises or not hyps:
            return [False] * len(hyps)
        sc = self.nli.scores([(p, h) for h in hyps for p in premises])
        k = len(premises)
        return [any(s.ent >= self.theta for s in sc[i * k:(i + 1) * k]) for i in range(len(hyps))]


def _sid(step_id: str) -> int:
    return int(step_id[1:])


def dependency_edges(steps: list[dict], memory_log: list[dict]) -> dict[str, set[str]]:
    """edges[u] = steps v whose context contained u's output or a memory entry derived from u (Sec. III-E)."""
    src_of_entry: dict[str, set[str]] = defaultdict(set)
    for e in memory_log:
        src_of_entry[e["id"]].update(e.get("source_steps", []))
    edges: dict[str, set[str]] = defaultdict(set)
    for s in steps:
        for ref in s["context_refs"]:
            kind, rid = ref.split(":", 1)
            if kind == "step":
                edges[rid].add(s["step_id"])
            elif kind == "entry":
                for u in src_of_entry.get(rid, ()):
                    edges[u].add(s["step_id"])
    return edges


def query_metrics(rec: dict, final_claims: list[str], y: dict[str, int | None], equiv: MetricEquiv) -> dict:
    """All claim/trace metrics for one query. `y` maps claim text -> 0/1/None."""
    tr, out = rec["trace"], rec["output"]
    steps = sorted(tr["steps"], key=lambda s: _sid(s["step_id"]))
    claims = tr["claims"]
    memory_log = out.get("memory_log", [])
    res: dict = {"qid": rec["qid"]}

    # HR inputs (Eq. 19)
    fy = [y.get(c) for c in final_claims]
    res["n_final_claims"] = len(final_claims)
    res["n_final_halluc"] = sum(1 for v in fy if v == 0)
    res["n_final_unlabelled"] = sum(1 for v in fy if v is None)

    # internal claims C_int, blocked(c) (Eq. 21)
    int_ids = [cid for cid, c in claims.items() if c["phase"] != "final"]
    written = {cid for e in memory_log for cid in e.get("source_claims", [])}
    int_texts = [claims[cid]["text"] for cid in int_ids]
    in_final = equiv.entailed_by_text(out.get("final_text", "") if not out.get("abstained") else "", int_texts)
    vsr_items = []
    for cid, present in zip(int_ids, in_final):
        v = y.get(claims[cid]["text"])
        if v is not None:
            vsr_items.append((v, (cid not in written) and not present))
    res["vsr_items"] = vsr_items

    # PD and escape rate (Eq. 6)
    edges = dependency_edges(steps, memory_log)
    halluc = [cid for cid in int_ids if y.get(claims[cid]["text"]) == 0]
    texts = [claims[c]["text"] for c in halluc]
    pds, escapes = [], []
    for comp in equiv.classes(texts):
        members = [halluc[i] for i in comp]
        by_step = sorted({claims[c]["step_id"] for c in members}, key=_sid)
        origin, later = by_step[0], by_step[1:]
        pds.append(eq.propagation_depth(origin, later, edges))
        escapes.append(any(in_final[int_ids.index(c)] for c in members))
    res["pd"], res["escaped"] = pds, escapes

    # MCR(t) (Eq. 7)
    text_of = {e["id"]: e["text"] for e in memory_log}
    mcr_curve = []
    for ev in tr["events"]:
        if ev["kind"] == "memory_write":
            labs = [y.get(text_of[i]) for i in ev["active_ids"]]
            labs = [v for v in labs if v is not None]
            mcr_curve.append((ev["t"], eq.mcr(labs)))
    res["mcr_curve"] = mcr_curve

    # CLR clusters (Eq. 3), computed uniformly from the trace
    res["clr_clusters"] = _clr_clusters(rec, steps, claims, memory_log, y, equiv)

    # calibration pairs (Eq. 22, Brier): system confidence vs truth of the scored cluster
    res["calib"] = [(c["conf_star"], y.get(c["rep_text"])) for c in out.get("clusters", [])
                    if c.get("label") != "Unverified" and y.get(c["rep_text"]) is not None]
    res["label_precision"] = [(("Singleton-grounded" if c["label"] == eq.SINGLETON else c["label"]),
                               y.get(c["rep_text"])) for c in out.get("clusters", [])
                              if c.get("status") == "written" and y.get(c["rep_text"]) is not None]
    return res


def _clr_clusters(rec, steps, claims, memory_log, y, equiv) -> list[dict]:
    """Per sub-task: cluster agent assertions, S = agents asserting, S_ind = agents whose earliest assertion
    step had no context item entailing a member; accepted per the system's own acceptance decision."""
    system = rec["system"]
    agent_phases = ("draft", "revision") if not system.startswith("b5") else tuple(
        p for p in {s["phase"] for s in steps} if p.startswith("debate_r"))
    entry_text = {e["id"]: e["text"] for e in memory_log}
    written = {cid for e in memory_log for cid in e.get("source_claims", [])}
    winners = {ev["subtask"]: ev["winner_step"] for ev in rec["trace"]["events"] if ev["kind"] == "debate_majority"}
    step_by_id = {s["step_id"]: s for s in steps}
    out = []
    by_sub: dict[str, list[str]] = defaultdict(list)
    for cid, c in claims.items():
        if c["phase"] in agent_phases and not c["agent_id"] in ("synthesizer",):
            by_sub[c["subtask"]].append(cid)
    for sub, cids in by_sub.items():
        texts = [claims[c]["text"] for c in cids]
        for comp in equiv.classes(texts):
            members = [cids[i] for i in comp]
            agents = sorted({claims[c]["agent_id"] for c in members})
            if len(agents) < 2:
                continue
            mtexts = [claims[c]["text"] for c in members]
            exposed = set()
            for a in agents:
                first = min((claims[c]["step_id"] for c in members if claims[c]["agent_id"] == a), key=_sid)
                X = []
                for ref in step_by_id[first]["context_refs"]:
                    kind, rid = ref.split(":", 1)
                    if kind == "entry" and rid in entry_text:
                        X.append(entry_text[rid])
                    elif kind == "step" and rid in step_by_id:
                        X += [claims[x]["text"] for x in step_by_id[rid]["claim_ids"]]
                if any(equiv.entails_any(X, mtexts)):
                    exposed.add(a)
            if system.startswith("b5"):
                accepted = any(claims[c]["step_id"] == winners.get(sub) for c in members)
            else:
                accepted = any(c in written for c in members)
            ys = [y.get(t) for t in mtexts if y.get(t) is not None]
            out.append({"subtask": sub, "S": len(agents), "S_ind": len(set(agents) - exposed), "accepted": accepted,
                        "y": (1 if ys and statistics.mean(ys) >= 0.5 else 0) if ys else None, "texts": mtexts})
    return out


def aggregate(per_query: list[dict], records: list[dict], examples: dict) -> dict:
    """Run-level metrics from per-query results."""
    m: dict = {}
    n = len(records)
    if n == 0:
        return {"n_queries": 0, "error": "no successful queries to evaluate"}
    answered = [r for r in records if not r["output"]["abstained"]]
    m["n_queries"] = n
    m["coverage"] = len(answered) / n if n else None
    m["HR"] = eq.hallucination_rate([(q["n_final_halluc"], q["n_final_claims"] - q["n_final_unlabelled"]) for q in per_query])
    m["final_claims_unlabelled"] = sum(q["n_final_unlabelled"] for q in per_query)
    m["claims_per_answer"] = (statistics.mean(q["n_final_claims"] for q, r in zip(per_query, records)
                                              if not r["output"]["abstained"]) if answered else None)
    ems = [exact_match(r["output"]["answer"], examples[r["qid"]].answer) if not r["output"]["abstained"] else 0.0 for r in records]
    f1s = [f1_score(r["output"]["answer"], examples[r["qid"]].answer) if not r["output"]["abstained"] else 0.0 for r in records]
    m["EM"], m["F1"] = statistics.mean(ems), statistics.mean(f1s)
    if answered:
        m["EM_answered"] = statistics.mean(exact_match(r["output"]["answer"], examples[r["qid"]].answer) for r in answered)
        m["F1_answered"] = statistics.mean(f1_score(r["output"]["answer"], examples[r["qid"]].answer) for r in answered)
    vsr, frr = eq.vsr_frr([it for q in per_query for it in q["vsr_items"]])
    m["VSR"], m["FRR"] = vsr, frr
    pds = [d for q in per_query for d in q["pd"]]
    esc = [e for q in per_query for e in q["escaped"]]
    m["PD_mean"] = statistics.mean(pds) if pds else None
    m["PD_distribution"] = {str(k): pds.count(k) for k in sorted(set(pds))}
    m["escape_rate"] = (sum(esc) / len(esc)) if esc else None
    m["n_hallucinated_classes"] = len(pds)
    by_t = defaultdict(list)
    for q in per_query:
        for t, v in q["mcr_curve"]:
            if v is not None:
                by_t[t].append(v)
    m["MCR_curve"] = {str(t): {"mean": statistics.mean(v), "n": len(v)} for t, v in sorted(by_t.items())}
    if len(by_t) >= 2:
        xs = [t for t, v in by_t.items() for _ in v]
        ys = [x for t, v in by_t.items() for x in v]
        m["MCR_slope"] = float(np.polyfit(xs, ys, 1)[0]) if len(set(xs)) > 1 else None
    clr_items = [c for q in per_query for c in q["clr_clusters"] if c["y"] is not None]
    m["CLR"] = eq.clr(clr_items)
    m["CLR_n_false_accepted"] = sum(1 for c in clr_items if c["accepted"] and c["S"] >= 2 and c["y"] == 0)
    cal = [c for q in per_query for c in q["calib"]]
    m["ECE"] = eq.ece([c for c, _ in cal], [y for _, y in cal]) if cal else None
    m["Brier"] = eq.brier([c for c, _ in cal], [y for _, y in cal]) if cal else None
    m["n_calibration_claims"] = len(cal)
    lp = defaultdict(list)
    for q in per_query:
        for lab, v in q["label_precision"]:
            lp[lab].append(v)
    m["label_precision"] = {k: {"precision": statistics.mean(v), "n": len(v)} for k, v in lp.items()}
    m["cost"] = cost_summary(records)
    return m


def cost_summary(records: list[dict]) -> dict:
    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return {"mean": statistics.mean(vals), "median": statistics.median(vals),
                "p95": float(np.percentile(vals, 95)), "total": sum(vals)}
    sysc = [r["cost"]["system"] for r in records]
    return {
        "input_tokens": stats([c["input_tokens"] for c in sysc]),
        "output_tokens": stats([c["output_tokens"] for c in sysc]),
        "llm_calls": stats([c["llm_calls"] for c in sysc]),
        "nli_calls": stats([c["nli_calls"] for c in sysc]),
        "cost_usd": stats([c["cost_usd"] for c in sysc]),
        "latency_s": stats([r["cost"]["wall_time_s"] for r in records]),
        "instrumentation_cost_usd": stats([r["cost"]["instrumentation"]["cost_usd"] for r in records]),
    }
