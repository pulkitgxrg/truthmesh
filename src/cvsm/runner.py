"""Run a configuration over a dataset split and write a self-contained results folder:

runs/<timestamp>_<name>/
  config.json        resolved configuration (all switches and parameters)
  environment.json   model IDs, providers, prices used, NLI/encoder revisions, dataset hashes, versions, dates
  records/<qid>.json raw outputs, full execution trace (steps, claims, exposures, memory writes), cost
  records.jsonl      one line per query (same content)
  run_log.json       start/end time, failures
Evaluation (labels, metrics) is added by evaluate_run()."""
from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from .baselines import run_baseline
from .components import QueryContext, extract_text_claims
from .config import RunConfig
from .datasets import RAW_FILES, SOURCES, file_sha256
from .llm import Ledger, LLMClient, PRICES
from .nli import CountingNLI, Encoder, NLIModel
from .pipeline import run_cvsm
from .retrieval import Retriever
from .trace import Trace
from .types import Example


class Resources:
    """Models shared across queries (loaded once)."""

    def __init__(self, cfg: RunConfig, llm: LLMClient | None = None, nli=None, encoder=None):
        self.llm = llm or LLMClient(cfg.cache_dir)
        self.nli = nli or NLIModel(cfg.nli_checkpoint)
        self.encoder = encoder or Encoder(cfg.encoder)


def run_query(cfg: RunConfig, res: Resources, ex: Example) -> dict:
    ledger, instr = Ledger(), Ledger()
    trace = Trace(ex.qid)
    ctx = QueryContext(cfg=cfg, llm=res.llm, nli=CountingNLI(res.nli, ledger), encoder=res.encoder,
                       retriever=Retriever(ex.passages), ex=ex, trace=trace, ledger=ledger, instr_ledger=instr)
    t0 = time.perf_counter()
    status, err, output = "ok", None, None
    try:
        if cfg.dataset == "halueval":                      # detector evaluation (Section VI-D)
            from .baselines import run_halueval_judge
            from .pipeline import detect_hallucination
            output = detect_hallucination(ctx) if cfg.switches.system == "cvsm" else run_halueval_judge(ctx)
        else:
            output = run_cvsm(ctx) if cfg.switches.system in ("cvsm", "b6") else run_baseline(ctx)
    except Exception as e:  # recorded, never hidden: failures are reported in the run summary
        status, err = "error", "".join(traceback.format_exception(e))
    wall = time.perf_counter() - t0
    return {
        "qid": ex.qid, "system": cfg.preset, "status": status, "error": err, "errors": ctx.errors,
        "question": ex.question, "gold_answer": ex.answer, "output": output, "trace": trace.to_dict(),
        "cost": {"system": ledger.summary(), "instrumentation": instr.summary(), "wall_time_s": wall},
    }


def environment(cfg: RunConfig, res: Resources | None) -> dict:
    models = {a.model.key() for a in cfg.agents} | {cfg.utility_model.key(), cfg.arbiter_model.key(),
                                                     cfg.eval_decomposer_model.key()}
    if cfg.judge_model:
        models.add(cfg.judge_model.key())
    pk = ["anthropic", "openai", "google-genai", "torch", "transformers", "sentence-transformers", "numpy", "scipy"]
    raw = Path(cfg.data_dir) / RAW_FILES[cfg.dataset]
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version, "platform": platform.platform(),
        "packages": {p: _ver(p) for p in pk},
        "models": sorted(models),
        "prices_used": {m: PRICES.get(m) for m in sorted(models)},
        "nli_checkpoint": cfg.nli_checkpoint,
        "nli_revision": getattr(res.nli, "revision", None) if res else None,
        "label_nli_checkpoint": cfg.label_nli_checkpoint,
        "encoder": cfg.encoder,
        "encoder_revision": getattr(res.encoder, "revision", None) if res else None,
        "dataset": {"name": cfg.dataset, "file": str(raw), "sha256": file_sha256(raw) if raw.exists() else None,
                    "source": SOURCES[cfg.dataset]},
    }


def _ver(p):
    try:
        return metadata.version(p)
    except metadata.PackageNotFoundError:
        return None


def run(cfg: RunConfig, examples: list[Example], res: Resources | None = None, out_dir: Path | None = None,
        guard=None) -> Path:
    res = res or Resources(cfg)
    out = out_dir or Path(cfg.runs_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg.name}_{cfg.preset}"
    (out / "records").mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=1))
    (out / "environment.json").write_text(json.dumps(environment(cfg, res), indent=1))
    log = {"start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "qids": [e.qid for e in examples]}

    def one(ex):
        if guard is not None and not guard.allow():
            print(f"[{cfg.preset}] {ex.qid} SKIPPED: budget would be exceeded", flush=True)
            return {"qid": ex.qid, "system": cfg.preset, "status": "skipped_budget", "error": "budget", "errors": [],
                    "question": ex.question, "gold_answer": ex.answer, "output": None, "trace": {},
                    "cost": {"system": {"cost_usd": 0, "llm_calls": 0}, "instrumentation": {"cost_usd": 0}, "wall_time_s": 0}}
        rec = run_query(cfg, res, ex)
        if guard is not None:
            guard.record((rec["cost"]["system"]["cost_usd"] or 0) + (rec["cost"]["instrumentation"]["cost_usd"] or 0),
                         "query", str(out))
        (out / "records" / f"{ex.qid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        print(f"[{cfg.preset}] {ex.qid} {rec['status']} answer={((rec['output'] or {}).get('answer') or '')[:60]!r} "
              f"calls={rec['cost']['system']['llm_calls']}", flush=True)
        return rec

    with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
        recs = list(pool.map(one, examples))
    with open(out / "records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.update(end=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               failures=[{"qid": r["qid"], "error": r["error"]} for r in recs if r["status"] == "error"],
               skipped_budget=[r["qid"] for r in recs if r["status"] == "skipped_budget"],
               component_errors={r["qid"]: r["errors"] for r in recs if r["errors"]})
    (out / "run_log.json").write_text(json.dumps(log, indent=1))
    return out


def load_records(run_dir: Path) -> list[dict]:
    return [json.loads(l) for l in (Path(run_dir) / "records.jsonl").read_text().splitlines() if l.strip()]
