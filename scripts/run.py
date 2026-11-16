"""Run one or more systems on a dataset split, then label and evaluate each run.

Example (pilot):
  python scripts/run.py --config configs/pilot_hotpotqa.yaml --presets cvsm b1 b5 --rho runs/<..>_rho_cvsm.json
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

from cvsm.config import load_config
from cvsm.datasets import load_split
from cvsm.eval_run import evaluate_run
from cvsm.llm import LLMClient
from cvsm.nli import Encoder, NLIModel
from cvsm.runner import Resources, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--presets", nargs="+", required=True)
    ap.add_argument("--rho", help="calibration json from scripts/estimate_rho.py (sets params.rho_hat)")
    ap.add_argument("--n", type=int)
    ap.add_argument("--offset", type=int, help="start index in the split (default: config 'offset')")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    raw = yaml.safe_load(Path(a.config).read_text())
    overrides = {}
    if a.rho:
        rho = json.loads(Path(a.rho).read_text())["rho_hat"]
        if rho is None:
            sys.exit("rho_hat undefined in calibration file; refusing to guess a value")
        overrides["params"] = {"rho_hat": rho}
    base = load_config(a.config, "cvsm", overrides)
    n = a.n if a.n is not None else raw.get("n")
    exs = load_split(base.dataset, base.data_dir, base.split, n, a.offset if a.offset is not None else raw.get("offset", 0))
    examples = {e.qid: e for e in exs}
    llm = LLMClient(base.cache_dir)
    nli, enc = NLIModel(base.nli_checkpoint), Encoder(base.encoder)
    label_nli = NLIModel(base.label_nli_checkpoint) if not a.no_eval else None
    for preset in a.presets:
        cfg = load_config(a.config, preset, overrides)
        out = run(cfg, exs, Resources(cfg, llm=llm, nli=nli, encoder=enc))
        print("run folder:", out, flush=True)
        if not a.no_eval:
            m = evaluate_run(out, cfg, examples, llm, label_nli, enc)
            print(json.dumps({k: m.get(k) for k in ("n_queries", "HR", "EM", "F1", "coverage", "VSR", "FRR", "PD_mean",
                                                    "escape_rate", "CLR", "ECE", "error")}, indent=1), flush=True)


if __name__ == "__main__":
    sys.exit(main())
