"""(Re-)evaluate an existing run folder: label claims and compute metrics. System LLM calls are not repeated."""
import argparse
import json

from cvsm.config import load_config
from cvsm.datasets import load_split
from cvsm.eval_run import evaluate_run
from cvsm.llm import LLMClient
from cvsm.nli import Encoder, NLIModel
from cvsm.runner import load_records

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--runs", nargs="+", required=True)
a = ap.parse_args()
label_nli, enc = None, None
for d in a.runs:
    cfgd = json.loads(open(f"{d}/config.json").read())
    cfg = load_config(a.config, cfgd["preset"], {"params": {"rho_hat": cfgd["params"]["rho_hat"]}})
    qids = {r["qid"] for r in load_records(d)}
    exs = {e.qid: e for e in load_split(cfg.dataset, cfg.data_dir, cfg.split) if e.qid in qids}
    label_nli = label_nli or NLIModel(cfg.label_nli_checkpoint)
    enc = enc or Encoder(cfg.encoder)
    m = evaluate_run(d, cfg, exs, LLMClient(cfg.cache_dir), label_nli, enc)
    print(d, json.dumps({k: m.get(k) for k in ("n_queries", "HR", "final_claims_unlabelled", "EM", "F1", "coverage",
                                               "VSR", "FRR", "PD_mean", "escape_rate", "CLR", "ECE", "label_stage_counts")}))
