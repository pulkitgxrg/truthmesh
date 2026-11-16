"""Section V-G analysis and Section VI outputs from the study manifest (runs/study_manifest.jsonl).

Writes results/: metrics.json (every number with its 95% bootstrap CI), table6.tex, table7.tex, table8.tex,
fig3_mcr.pdf ... fig7_pareto.pdf, error_analysis_sample.csv, and RESULTS.md (H1-H5 verdicts by the rules
stated in DESIGN_NOTES #55). Statistics: 10,000 bootstrap resamples with the query as the unit; a system's
point estimate is the mean over its U runs of the per-run statistic; paired tests resample the same queries
for both systems; McNemar on per-query correctness (majority over the U runs); Holm within each table.

  python scripts/analyze.py --config configs/frozen_study.yaml [--n-boot 10000] [--out results]
"""
import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from cvsm import equations as eq
from cvsm import stats
from cvsm.config import load_config
from cvsm.datasets import exact_match, f1_score, load_split
from cvsm.llm import Ledger, LLMClient
from cvsm.runner import load_records
from cvsm.task_metrics import fever_label_correct, truthfulqa_judge

SYSTEM_NAMES = {"b1": "B1 Single agent", "b2": "B2 Single agent + retrieval", "b3": "B3 Self-consistency",
                "b4": "B4 Chain-of-Verification", "b5": "B5 Multi-agent debate", "b6": "B6 Unguarded shared memory",
                "cvsm": "CV-SM", "a1": "A1 $-$R1", "a2": "A2 $-$R2", "a3": "A3 $-$SML", "a4": "A4 Singletons rejected",
                "a5": "A5 Majority arbitration", "a6": r"A6 $\hat\rho = 0$", "a7": "A7 Homogeneous agents",
                "a8": "A8 No premise cap or retraction"}
BASELINES = ["b1", "b2", "b3", "b4", "b5", "b6"]
ABL = [f"a{i}" for i in range(1, 9)]
HR_DATASETS = ("hotpotqa", "fever", "truthfulqa")


# ---------------------------------------------------------------- loading
def load_rows(entry, examples, llm, judge, ledger):
    """One row per query of one run, with everything the statistics need."""
    dirs = [Path(x) for x in entry.get("dirs", [entry["dir"]])]     # one run may be split over folders
    recs, pq = {}, {}
    for d in dirs:
        recs.update({r["qid"]: r for r in load_records(d) if r["status"] == "ok"})
        if (d / "per_query.json").exists():
            pq.update({q["qid"]: q for q in json.loads((d / "per_query.json").read_text())})
    rows = {}
    for qid, r in recs.items():
        ex, out, q = examples[qid], r["output"], pq.get(qid, {})
        ans = "" if out["abstained"] else (out.get("answer") or "")
        c = r["cost"]["system"]
        agent_in = [x["input_tokens"] / x["calls"] for k, x in c.get("by_component", {}).items()
                    if k.startswith("agent:") and x["calls"]]
        row = {"qid": qid, "dataset": entry["dataset"], "answered": int(not out["abstained"]),
               "hr": (q.get("n_final_halluc", 0), q.get("n_final_claims", 0) - q.get("n_final_unlabelled", 0)),
               "n_claims": q.get("n_final_claims", 0), "vsr": q.get("vsr_items", []), "pd": q.get("pd", []),
               "escaped": q.get("escaped", []), "clr": q.get("clr_clusters", []), "calib": q.get("calib", []),
               "mcr": q.get("mcr_curve", []), "lp": q.get("label_precision", []),
               "in_tok": c["input_tokens"], "out_tok": c["output_tokens"], "calls": c["llm_calls"],
               "nli": c["nli_calls"], "usd": c["cost_usd"], "lat": r["cost"]["wall_time_s"],
               "tok": c["input_tokens"] + c["output_tokens"],
               "agent_in_per_call": statistics.mean(agent_in) if agent_in else None}
        if entry["dataset"] == "hotpotqa":
            row["em"], row["f1"] = exact_match(ans, ex.answer), f1_score(ans, ex.answer)
        elif entry["dataset"] == "fever":
            row["fever_acc"] = fever_label_correct(ans, ex.answer)
        elif entry["dataset"] == "truthfulqa":
            j = truthfulqa_judge(llm, judge, ex, ans, ledger) if judge else {"truthful": None, "informative": None}
            row["tqa_ti"] = None if j["truthful"] is None else int(j["truthful"] and j["informative"])
            row["tqa_t"] = j["truthful"]
        elif entry["dataset"] == "halueval":
            row["halu_acc"] = int(out.get("predicted_hallucinated") == out.get("gold_hallucinated"))
        rows[qid] = row
    return rows


# ---------------------------------------------------------------- statistics on rows
def m_hr(rs):
    return eq.hallucination_rate([r["hr"] for r in rs])


def m_mean(key):
    def f(rs):
        v = [r[key] for r in rs if r.get(key) is not None]
        return statistics.mean(v) if v else None
    return f


def m_vsr(idx):
    def f(rs):
        v = eq.vsr_frr([tuple(it) for r in rs for it in r["vsr"]])[idx]
        return v
    return f


def m_clr(rs):
    return eq.clr([c for r in rs for c in r["clr"] if c["y"] is not None])


def m_ece(rs):
    c = [x for r in rs for x in r["calib"]]
    return eq.ece([a for a, _ in c], [b for _, b in c]) if c else None


def m_brier(rs):
    c = [x for r in rs for x in r["calib"]]
    return eq.brier([a for a, _ in c], [b for _, b in c]) if c else None


def m_pd(rs):
    v = [d for r in rs for d in r["pd"]]
    return statistics.mean(v) if v else None


def m_escape(rs):
    v = [e for r in rs for e in r["escaped"]]
    return sum(v) / len(v) if v else None


def m_cpa(rs):
    v = [r["n_claims"] for r in rs if r["answered"]]
    return statistics.mean(v) if v else None


def over_runs(runs, qids, fn):
    """Mean over the U runs of fn on the given queries (point estimate of Table VI cells)."""
    vals = [fn([run[q] for q in qids if q in run]) for run in runs]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def boot(runs, fn, n_boot, seed=0):
    qids = sorted(set().union(*[set(r) for r in runs])) if runs else []
    if not qids:
        return {"est": None, "lo": None, "hi": None, "n_queries": 0}
    point = over_runs(runs, qids, fn)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = [rng.choice(qids) for _ in qids]
        v = over_runs(runs, s, fn)
        if v is not None:
            vals.append(v)
    lo, hi = (np.percentile(vals, [2.5, 97.5]).tolist() if vals else (None, None))
    return {"est": point, "lo": lo, "hi": hi, "n_queries": len(qids)}


def paired_delta(runs_a, runs_b, fn, n_boot, seed=0):
    """Delta = fn(A) - fn(B) on the same queries; two-sided bootstrap p-value."""
    qa = set().union(*[set(r) for r in runs_a]); qb = set().union(*[set(r) for r in runs_b])
    qids = sorted(qa & qb)
    if not qids:
        return None
    def d(s):
        x, y = over_runs(runs_a, s, fn), over_runs(runs_b, s, fn)
        return None if x is None or y is None else x - y
    point = d(qids)
    rng = random.Random(seed)
    vals = [v for v in (d([rng.choice(qids) for _ in qids]) for _ in range(n_boot)) if v is not None]
    if point is None or not vals:
        return {"est": point, "lo": None, "hi": None, "p": None, "n_queries": len(qids)}
    lo, hi = np.percentile(vals, [2.5, 97.5]).tolist()
    p = min(1.0, 2 * min(np.mean([v <= 0 for v in vals]), np.mean([v >= 0 for v in vals])))
    return {"est": point, "lo": lo, "hi": hi, "p": float(p), "n_queries": len(qids)}


def majority_correct(runs, key):
    out = {}
    for q in set().union(*[set(r) for r in runs]):
        v = [r[q][key] for r in runs if q in r and r[q].get(key) is not None]
        if v:
            out[q] = int(sum(1 for x in v if x >= 1) * 2 > len(v))
    return out


# ---------------------------------------------------------------- output helpers
def fmt(c, pct=False, d=2):
    if c is None or c.get("est") is None:
        return "--"
    k = 100 if pct else 1
    s = f"{c['est'] * k:.{d if not pct else 1}f}"
    if c.get("lo") is not None:
        s += f" [{c['lo'] * k:.{d if not pct else 1}f}, {c['hi'] * k:.{d if not pct else 1}f}]"
    return s


def write_tex(path, caption, label, header, rows, note=""):
    cols = "l" + "c" * (len(header) - 1)
    L = [r"\begin{table*}[!t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
         r"\footnotesize", rf"\begin{{tabular}}{{{cols}}}", r"\hline", " & ".join(header) + r" \\", r"\hline"]
    L += [" & ".join(r) + r" \\" for r in rows]
    L += [r"\hline", r"\end{tabular}"]
    if note:
        L += [rf"\par\smallskip\parbox{{\textwidth}}{{\footnotesize {note}}}"]
    L += [r"\end{table*}"]
    Path(path).write_text("\n".join(L) + "\n")


def setup_mpl():
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42, "font.family": "serif",
                         "figure.figsize": (3.5, 2.4), "axes.spines.top": False, "axes.spines.right": False})
    return plt


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", default="runs/study_manifest.jsonl")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--u-min", type=float, default=0.8)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    cfg = load_config(a.config, "cvsm")
    llm, ledger = LLMClient(cfg.cache_dir), Ledger()
    entries = [json.loads(l) for l in Path(a.manifest).read_text().splitlines() if l.strip()]
    ex_cache = {}
    def examples(d):
        if d not in ex_cache:
            ex_cache[d] = {e.qid: e for e in load_split(d, cfg.data_dir, a.split)}
        return ex_cache[d]
    # runs[(block, system, dataset, k, tau)] = [rows_u1, rows_u2, ...]
    runs = defaultdict(list)
    for e in entries:
        runs[(e["block"], e["system"], e["dataset"], e["k"], e["tau"])].append(
            load_rows(e, examples(e["dataset"]), llm, cfg.judge_model, ledger))
    B = a.n_boot
    R = {"manifest": a.manifest, "n_boot": B, "runs": entries, "systems": {}, "paired": {}, "ablations": {}, "cost": {}, "hypotheses": {}}

    def sysruns(system, dataset, block="main", k=3, tau=None):
        if system in ABL and block == "main":
            block = "ablation"
        return runs.get((block, system, dataset, k, tau), [])

    def pooled(system, datasets, block="main"):
        """Pool datasets by prefixing query ids so a run is the union of per-dataset rows."""
        U = max((len(sysruns(system, d, block)) for d in datasets), default=0)
        merged = []
        for u in range(U):
            m = {}
            for d in datasets:
                rs = sysruns(system, d, block)
                if u < len(rs):
                    m.update({f"{d}:{q}": r for q, r in rs[u].items()})
            merged.append(m)
        return merged

    # ---------------- Table VI
    rows6, sys_metrics = [], {}
    for s in BASELINES + ["cvsm"]:
        M = {}
        M["hotpot_em"] = boot(sysruns(s, "hotpotqa"), m_mean("em"), B)
        M["hotpot_f1"] = boot(sysruns(s, "hotpotqa"), m_mean("f1"), B)
        M["fever_acc"] = boot(sysruns(s, "fever"), m_mean("fever_acc"), B)
        M["tqa_ti"] = boot(sysruns(s, "truthfulqa"), m_mean("tqa_ti"), B)
        M["halu_acc"] = boot(sysruns(s, "halueval"), m_mean("halu_acc"), B)
        P = pooled(s, HR_DATASETS)
        M["HR"] = boot(P, m_hr, B); M["claims_per_answer"] = boot(P, m_cpa, B)
        M["coverage"] = boot(P, m_mean("answered"), B); M["tokens"] = boot(P, m_mean("tok"), B)
        sys_metrics[s] = M
        rows6.append([SYSTEM_NAMES[s], f"{fmt(M['hotpot_em'])} / {fmt(M['hotpot_f1'])}", fmt(M["fever_acc"]),
                      fmt(M["tqa_ti"], True), fmt(M["halu_acc"]), fmt(M["HR"], True), fmt(M["claims_per_answer"]),
                      fmt(M["coverage"], True), fmt(M["tokens"], d=0)])
    R["systems"] = sys_metrics
    write_tex(out / "table6.tex", "Main results on the test splits", "tab:main",
              ["System", "HotpotQA EM / F1", "FEVER label acc.", r"TruthfulQA \% T\&I", "HaluEval acc.", r"HR (\%)",
               "Claims/answer", r"Coverage (\%)", "Tokens/query"], rows6,
              f"Means over U runs with 95\\% bootstrap intervals ({B} resamples, query-level). HR, claims per answer, "
              "coverage and tokens pool HotpotQA, FEVER and TruthfulQA. FEVER score is not reported: it requires "
              "predicted FEVER evidence sentences, which the systems do not output. HaluEval: detector accuracy (CV-SM) "
              "vs single-LLM judge (B1); other systems not applicable.")

    # ---------------- paired CV-SM vs baselines (H1), Holm
    cv = pooled("cvsm", HR_DATASETS)
    pairs = {}
    for s in BASELINES:
        pairs[s] = paired_delta(cv, pooled(s, HR_DATASETS), m_hr, B) or {"est": None, "lo": None, "hi": None,
                                                                         "p": None, "note": "system not run"}
        em_cv, em_b = majority_correct(sysruns("cvsm", "hotpotqa"), "em"), majority_correct(sysruns(s, "hotpotqa"), "em")
        common = sorted(set(em_cv) & set(em_b))
        pairs[s]["mcnemar_em_hotpotqa"] = stats.mcnemar([em_cv[q] for q in common], [em_b[q] for q in common]) if common else None
    valid = {s: v["p"] for s, v in pairs.items() if v and v.get("p") is not None}
    holm = stats.holm(valid) if valid else {}
    mc = stats.holm({s: v["mcnemar_em_hotpotqa"]["p_value"] for s, v in pairs.items()
                     if v and v.get("mcnemar_em_hotpotqa")}) if pairs else {}
    for s in pairs:
        if pairs[s]:
            pairs[s]["holm"] = holm.get(s); pairs[s]["mcnemar_holm"] = mc.get(s)
    R["paired"] = pairs

    # ---------------- Table VII (ablations; HotpotQA + TruthfulQA pooled)
    abl_ds = ("hotpotqa", "truthfulqa")
    cv7 = pooled("cvsm", abl_ds)
    rows7, abl = [], {}
    for s in ["cvsm"] + ABL:
        P = cv7 if s == "cvsm" else pooled(s, abl_ds, "ablation")
        M = {"HR": boot(P, m_hr, B), "CLR": boot(P, m_clr, B), "VSR": boot(P, m_vsr(0), B),
             "FRR": boot(P, m_vsr(1), B), "ECE": boot(P, m_ece, B)}
        M["dHR"] = None if s == "cvsm" else paired_delta(P, cv7, m_hr, B)
        M["dCLR"] = None if s == "cvsm" else paired_delta(P, cv7, m_clr, B)
        abl[s] = M
        rows7.append([SYSTEM_NAMES[s], fmt(M["HR"], True), "--" if M["dHR"] is None else fmt(M["dHR"], True),
                      fmt(M["CLR"], True), fmt(M["VSR"], True), fmt(M["FRR"], True), fmt(M["ECE"])])
    pa = {s: abl[s]["dHR"]["p"] for s in ABL if abl[s]["dHR"] and abl[s]["dHR"].get("p") is not None}
    R["ablations"] = abl; R["ablation_holm"] = stats.holm(pa) if pa else {}
    write_tex(out / "table7.tex", "Ablation results on HotpotQA and TruthfulQA", "tab:ablation",
              ["Configuration", r"HR (\%)", r"$\Delta$HR vs CV-SM (95\% CI)", r"CLR (\%)", r"VSR (\%)", r"FRR (\%)", "ECE"],
              rows7, "Pooled over HotpotQA and TruthfulQA; $\\Delta$ is paired by query; Holm-adjusted p-values in metrics.json.")

    # ---------------- Table VIII (cost on HotpotQA, k = 2/3/5)
    rows8, cost = [], {}
    for s in ("b5", "cvsm"):
        for k in (2, 3, 5):
            rs = sysruns(s, "hotpotqa", "main" if k == 3 else "k_sweep", k)
            allrows = [r for run in rs for r in run.values()]
            if not allrows:
                rows8.append([SYSTEM_NAMES[s], str(k)] + ["--"] * 6); continue
            c = {x: statistics.mean(r[x] for r in allrows) for x in ("in_tok", "out_tok", "calls", "nli")}
            lat = [r["lat"] for r in allrows]
            c["lat_median"], c["lat_p95"] = statistics.median(lat), float(np.percentile(lat, 95))
            c["agent_in_per_call"] = [r["agent_in_per_call"] for r in allrows]
            cost[f"{s}@k={k}"] = {x: v for x, v in c.items() if x != "agent_in_per_call"}
            cost[f"{s}@k={k}"]["_agent_in"] = [v for v in c["agent_in_per_call"] if v is not None]
            rows8.append([SYSTEM_NAMES[s], str(k), f"{c['in_tok']:.0f}", f"{c['out_tok']:.0f}", f"{c['calls']:.1f}",
                          f"{c['nli']:.0f}", f"{c['lat_median']:.1f}", f"{c['lat_p95']:.1f}"])
    write_tex(out / "table8.tex", "Cost per query on HotpotQA", "tab:cost",
              ["System", "$k$", "Input tokens", "Output tokens", "LLM calls", "NLI calls", "Median latency (s)",
               "p95 latency (s)"], rows8, "Means over queries and runs; latency is wall-clock per query.")
    R["cost"] = {k: {x: v for x, v in c.items() if x != "_agent_in"} for k, c in cost.items()}

    # ---------------- H5: per-agent input tokens vs k (quadratic fit, bootstrap over queries)
    h5 = {}
    for s in ("b5", "cvsm"):
        pts = [(k, v) for k in (2, 3, 5) for v in cost.get(f"{s}@k={k}", {}).get("_agent_in", [])]
        if len({k for k, _ in pts}) == 3:
            ks, ys = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)
            coef = np.polyfit(ks, ys, 2)
            rng = np.random.default_rng(0); qs = []
            for _ in range(B):
                i = rng.integers(0, len(ks), len(ks))
                if len(set(ks[i])) == 3:
                    qs.append(np.polyfit(ks[i], ys[i], 2)[0])
            h5[s] = {"quad": float(coef[0]), "lin": float(coef[1]), "quad_ci": np.percentile(qs, [2.5, 97.5]).tolist()}
    R["h5_fit"] = h5

    # ---------------- figures
    plt = setup_mpl()
    # Fig 3: MCR(t)
    fig, ax = plt.subplots()
    slopes = {}
    for s, lab in (("cvsm", "CV-SM"), ("b6", "B6"), ("a2", "A2")):
        rs = sysruns(s, "hotpotqa")
        by_t = defaultdict(list)
        pts = []
        for run in rs:
            for r in run.values():
                for t, v in r["mcr"]:
                    if v is not None:
                        by_t[t].append(v); pts.append((t, v, r["qid"]))
        if not by_t:
            continue
        ts = sorted(by_t)
        mean = [statistics.mean(by_t[t]) for t in ts]
        lo, hi = [], []
        rng = np.random.default_rng(0)
        for t in ts:
            arr = np.array(by_t[t]); bs = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(min(B, 2000))]
            lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        ax.plot(ts, mean, label=lab); ax.fill_between(ts, lo, hi, alpha=0.2)
        qids = sorted({q for _, _, q in pts})
        sl = []
        for _ in range(B):
            pick = set(rng.choice(qids, len(qids)))
            sub = [(t, v) for t, v, q in pts if q in pick]
            if len({t for t, _ in sub}) > 1:
                sl.append(np.polyfit([t for t, _ in sub], [v for _, v in sub], 1)[0])
        if len({t for t, _, _ in pts}) > 1:
            slopes[s] = {"slope": float(np.polyfit([t for t, _, _ in pts], [v for _, v, _ in pts], 1)[0]),
                         "ci": np.percentile(sl, [2.5, 97.5]).tolist() if sl else None}
    ax.set_xlabel("Write index $t$"); ax.set_ylabel("MCR($t$)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig3_mcr.pdf"); plt.close(fig)
    R["mcr_slopes"] = slopes
    # Fig 4: PD box plots + escape rate
    fig, ax = plt.subplots()
    data, labels = [], []
    for s in ["b5", "b6", "cvsm", "a1", "a2"]:
        v = [d for run in sysruns(s, "hotpotqa") for r in run.values() for d in r["pd"]]
        if v:
            esc = [e for run in sysruns(s, "hotpotqa") for r in run.values() for e in r["escaped"]]
            data.append(v); labels.append(f"{s.upper() if s != 'cvsm' else 'CV-SM'}\nesc={sum(esc) / len(esc):.2f}")
    if data:
        ax.boxplot(data); ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_ylabel("Propagation depth"); fig.tight_layout(); fig.savefig(out / "fig4_pd.pdf"); plt.close(fig)
    # Fig 5: precision of agreement labels
    fig, ax = plt.subplots()
    lp = {}
    present = [s for s in ("cvsm", "a1", "b5") if sysruns(s, "hotpotqa")]
    for i, s in enumerate(present):
        rs = sysruns(s, "hotpotqa")
        if s == "b5":
            items = [("Majority-accepted", c["y"]) for run in rs for r in run.values() for c in r["clr"]
                     if c["accepted"] and c["S"] >= 2 and c["y"] is not None]
        else:
            items = [(l, y) for run in rs for r in run.values() for l, y in r["lp"]]
        groups = defaultdict(list)
        for l, y in items:
            groups[l].append(y)
        for j, (l, ys) in enumerate(sorted(groups.items())):
            p = statistics.mean(ys); rng = np.random.default_rng(0)
            bs = [np.mean(rng.choice(ys, len(ys))) for _ in range(min(B, 2000))]
            lo_, hi_ = np.percentile(bs, [2.5, 97.5])
            lp[f"{s}:{l}"] = {"precision": p, "ci": [float(lo_), float(hi_)], "n": len(ys)}
            nm = {"cvsm": "CV-SM"}.get(s, s.upper())
            ax.errorbar(i + 0.15 * j, p, yerr=[[p - lo_], [hi_ - p]], fmt="o", label=f"{nm} {l}")
    ax.set_ylabel("Fraction correct")
    ax.set_xticks(range(len(present)), [{"cvsm": "CV-SM", "a1": "A1", "b5": "B5"}[s] for s in present])
    ax.set_xlim(-0.5, len(present) - 0.2); ax.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(out / "fig5_label_precision.pdf"); plt.close(fig)
    R["label_precision"] = lp
    # Fig 6: reliability diagram
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], ls=":", c="grey")
    for s, lab in (("cvsm", "CV-SM"), ("a6", "A6")):
        c = [x for run in sysruns(s, "hotpotqa") for r in run.values() for x in r["calib"]]
        if not c:
            continue
        conf, y = np.array([x for x, _ in c]), np.array([v for _, v in c])
        bins = np.minimum((conf * 10).astype(int), 9)
        xs = [conf[bins == b].mean() for b in range(10) if (bins == b).any()]
        ys = [y[bins == b].mean() for b in range(10) if (bins == b).any()]
        ax.plot(xs, ys, marker="o", label=f"{lab} (ECE {eq.ece(conf, y):.3f}, Brier {eq.brier(conf, y):.3f})")
    ax.set_xlabel("Conf$^*$"); ax.set_ylabel("Empirical accuracy"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "fig6_reliability.pdf"); plt.close(fig)
    # Fig 7: HR vs tokens with Pareto frontier (HotpotQA)
    fig, ax = plt.subplots()
    pts = []
    for (block, s, d, k, tau), rs in runs.items():
        if d != "hotpotqa":
            continue
        hr = over_runs(rs, sorted(set().union(*[set(r) for r in rs])), m_hr)
        tok = over_runs(rs, sorted(set().union(*[set(r) for r in rs])), m_mean("tok"))
        cov = over_runs(rs, sorted(set().union(*[set(r) for r in rs])), m_mean("answered"))
        if hr is not None and tok is not None:
            lab = s.upper() if s != "cvsm" else "CV-SM"
            lab += f" (cov. {cov * 100:.0f}%)" if cov is not None and cov < 0.995 else ""
            lab += f" k={k}" if k != 3 else ""
            lab += f" τ={tau}" if tau else ""
            pts.append((tok, hr, lab))
    for tok, hr, lab in pts:
        ax.scatter(tok, hr * 100, s=12); ax.annotate(lab, (tok, hr * 100), fontsize=6, xytext=(3, 3),
                                                     textcoords="offset points", ha="right" if tok > 5000 else "left")
    front, best = [], float("inf")
    for tok, hr, _ in sorted(pts):
        if hr < best:
            front.append((tok, hr)); best = hr
    if front:
        ax.step([f[0] for f in front], [f[1] * 100 for f in front], where="post", c="k", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("Tokens per query"); ax.set_ylabel("HR (%)")
    fig.tight_layout(); fig.savefig(out / "fig7_pareto.pdf"); plt.close(fig)
    R["pareto"] = [{"tokens": t, "HR": h, "config": l} for t, h, l in pts]

    # ---------------- error-analysis sample (Section VI-F); categories left blank for the authors
    sample = []
    for e in entries:
        if e["system"] != "cvsm" or e["block"] != "main" or e["dataset"] == "halueval":
            continue
        for d in map(Path, e.get("dirs", [e["dir"]])):
          labs = {(j["qid"], j["claim"]): j for j in map(json.loads, (d / "labels.jsonl").read_text().splitlines())}
          finals = {j["qid"]: j["claims"] for j in map(json.loads, (d / "final_claims.jsonl").read_text().splitlines())}
          for qid, cl in finals.items():
            for c in cl:
                lab = labs.get((qid, c["text"]))
                if lab and lab["y"] == 0:
                    sample.append({"dataset": e["dataset"], "u": e["u"], "qid": qid, "claim": c["text"],
                                   "label_verdict": lab["verdict"], "run_dir": str(d), "category": "", "notes": ""})
    random.Random(13).shuffle(sample)
    sample = sample[:100]
    with open(out / "error_analysis_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sample[0]) if sample else ["claim"]); w.writeheader(); w.writerows(sample)
    R["error_analysis"] = {"n_sampled": len(sample), "categories": ["decomposition error", "alignment error",
                           "shared misconception", "arbitration error", "grounding error", "synthesis error"],
                           "note": "categories to be assigned by the authors; not assigned automatically"}

    # ---------------- hypotheses
    R["hypotheses"] = decide(R, a.u_min)
    R["analysis_eval_cost"] = ledger.summary()
    (out / "metrics.json").write_text(json.dumps(R, indent=1, default=str))
    (out / "RESULTS.md").write_text(results_md(R, a.u_min))
    print("written", out)


def decide(R, u_min):
    """Verdicts by the rules in DESIGN_NOTES #55 (fixed before the data were seen)."""
    H = {}
    p, cov = R["paired"], R["systems"].get("cvsm", {}).get("coverage", {})
    ok = [s for s in BASELINES if p.get(s) and p[s].get("est") is not None and p[s].get("holm")]
    if len(ok) < len(BASELINES):
        H["H1"] = ("inconclusive", "not every baseline has a paired HR comparison")
    else:
        all_lower = all(p[s]["est"] < 0 and p[s]["holm"]["reject"] for s in BASELINES)
        any_higher = any(p[s]["est"] > 0 and p[s]["holm"]["reject"] for s in BASELINES)
        cov_ok = cov.get("est") is not None and cov["est"] >= u_min
        H["H1"] = (("supported" if cov_ok else "inconclusive (coverage below u_min)") if all_lower
                   else ("not supported" if any_higher or not all_lower else "inconclusive"),
                   "; ".join(f"{s}: ΔHR={p[s]['est']:+.4f} [{p[s]['lo']:+.4f}, {p[s]['hi']:+.4f}], Holm p={p[s]['holm']['p_adj']:.4g}" for s in BASELINES))
    # H2: B6 vs B5 on PD (MCR is undefined for memoryless B5; B6's MCR level and slope are reported)
    b5, b6 = R["systems"].get("b5"), R["systems"].get("b6")
    H["H2"] = ("see note", "PD comparison B6 vs B5 and MCR of B6 are in metrics.json; MCR(B5) is undefined because B5 "
                           "has no memory, so H2 is judged on PD (B6 > B5 with CI excluding 0) and MCR(B6) > MCR(CV-SM).")
    a1 = R["ablations"].get("a1", {}).get("dCLR")
    if not a1 or a1.get("lo") is None:
        H["H3"] = ("inconclusive", "CLR difference A1 - CV-SM undefined (no falsely accepted multi-agent clusters)")
    else:
        v = "supported" if a1["lo"] > 0 else ("not supported" if a1["hi"] < 0 else "inconclusive")
        H["H3"] = (v, f"CLR(A1) - CLR(CV-SM) = {a1['est']:+.4f} [{a1['lo']:+.4f}, {a1['hi']:+.4f}]")
    s = R.get("mcr_slopes", {})
    if "cvsm" in s and "b6" in s and s["cvsm"]["ci"] and s["b6"]["ci"]:
        flat = s["cvsm"]["ci"][0] <= 0 <= s["cvsm"]["ci"][1] or s["cvsm"]["ci"][1] < 0
        rising = s["b6"]["ci"][0] > 0
        H["H4"] = ("supported" if flat and rising else ("not supported" if not flat else "inconclusive"),
                   f"slope CV-SM {s['cvsm']['slope']:+.5f} {s['cvsm']['ci']}, B6 {s['b6']['slope']:+.5f} {s['b6']['ci']}")
    else:
        H["H4"] = ("inconclusive", "MCR slopes unavailable")
    h5 = R.get("h5_fit", {})
    if "b5" in h5 and "cvsm" in h5:
        quad_b5 = h5["b5"]["quad_ci"][0] > 0
        lin_cv = h5["cvsm"]["quad_ci"][0] <= 0 <= h5["cvsm"]["quad_ci"][1]
        H["H5"] = ("supported" if quad_b5 and lin_cv else "not supported" if not lin_cv else "inconclusive",
                   f"quadratic coef B5 {h5['b5']['quad']:+.2f} {h5['b5']['quad_ci']}, CV-SM {h5['cvsm']['quad']:+.2f} {h5['cvsm']['quad_ci']}")
    else:
        H["H5"] = ("inconclusive", "k sweep incomplete")
    return H


def results_md(R, u_min):
    L = ["# Results", "", f"Generated by scripts/analyze.py from {R.get('manifest')} "
         f"({len(R['runs'])} runs, {R['n_boot']} bootstrap resamples). Every number below is read from results/metrics.json.", "",
         "## Hypotheses", ""]
    for h, (v, detail) in R["hypotheses"].items():
        L += [f"- **{h}: {v}.** {detail}"]
    L += ["", f"Decision rules: DESIGN_NOTES.md #55 (coverage floor u_min = {u_min}).", "",
          "## Tables and figures", "", "results/table6.tex, table7.tex, table8.tex; results/fig3_mcr.pdf … fig7_pareto.pdf; "
          "results/error_analysis_sample.csv (categories to be filled by the authors)."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
