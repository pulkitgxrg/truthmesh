"""Label claims and compute metrics for a run folder. Writes final_claims.jsonl, labels.jsonl,
per_query.json, metrics.json and eval_cost.json into the run folder."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .components import extract_text_claims
from .config import RunConfig
from .evaluate import MetricEquiv, aggregate, query_metrics
from .labelling import LabelConfig, Labeller
from .llm import Ledger, LLMClient
from .runner import load_records


def evaluate_run(run_dir: Path, cfg: RunConfig, examples: dict, llm: LLMClient, label_nli, encoder,
                 theta_equiv: float = 0.5, concurrency: int = 4) -> dict:
    run_dir = Path(run_dir)
    recs = [r for r in load_records(run_dir) if r["status"] == "ok"]
    families = {a.model.family() for a in cfg.agent_pool()}
    labeller = Labeller(label_nli, llm, cfg.judge_model, LabelConfig(), agent_families=families)
    equiv = MetricEquiv(label_nli, encoder, theta=theta_equiv, theta_sim=cfg.params.theta_sim)
    eval_ledger = Ledger()

    def finals(r):
        return extract_text_claims(llm, cfg.eval_decomposer_model, r["question"], r["output"]["final_text"], eval_ledger)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        final_claims = list(pool.map(finals, recs))

    def texts_for(r, fc):
        out = list(dict.fromkeys([c["text"] for c in fc] + [c["text"] for c in r["trace"]["claims"].values()] +
                                 [e["text"] for e in r["output"].get("memory_log", [])] +
                                 [c["rep_text"] for c in r["output"].get("clusters", [])]))
        return out

    def label_query(args):
        r, fc = args
        ex = examples[r["qid"]]
        return {t: labeller.label(ex, t, ledger=eval_ledger) for t in texts_for(r, fc)}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        labels = list(pool.map(label_query, zip(recs, final_claims)))

    per_query = []
    for r, fc, lab in zip(recs, final_claims, labels):
        y = {t: v["y"] for t, v in lab.items()}
        per_query.append(query_metrics(r, [c["text"] for c in fc], y, equiv))
    metrics = aggregate(per_query, recs, examples)
    stage_counts: dict[str, int] = {}
    for lab in labels:
        for v in lab.values():
            stage_counts[f"{v['stage']}:{v['verdict']}"] = stage_counts.get(f"{v['stage']}:{v['verdict']}", 0) + 1
    metrics["label_stage_counts"] = stage_counts
    metrics["eval_cost"] = eval_ledger.summary()

    with open(run_dir / "final_claims.jsonl", "w") as f:
        for r, fc in zip(recs, final_claims):
            f.write(json.dumps({"qid": r["qid"], "claims": fc}, ensure_ascii=False) + "\n")
    with open(run_dir / "labels.jsonl", "w") as f:
        for r, lab in zip(recs, labels):
            for v in lab.values():
                f.write(json.dumps({"qid": r["qid"], **v}, ensure_ascii=False) + "\n")
    (run_dir / "per_query.json").write_text(json.dumps(per_query, ensure_ascii=False))
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, ensure_ascii=False))
    return metrics
