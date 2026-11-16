"""Run the main study (Section V): Table VI systems on every dataset, Table VII ablations on HotpotQA and
TruthfulQA, the k in {2,3,5} and tau sweeps, U seeds each, on the test splits, under a hard budget.

Each finished run folder is appended to runs/study_manifest.jsonl:
  {"block", "system", "dataset", "u", "k", "tau", "dir"}
scripts/analyze.py consumes the manifest. Runs already in the manifest are skipped (resumable).

  python scripts/run_study.py --config configs/frozen_study.yaml --n hotpotqa=491 truthfulqa=491 ... \
     --rho main=runs/<rho>.json a7=runs/<rho_a7>.json k2=... k5=... --budget-usd 200 --fee 1.1 --sc-samples 7
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

from cvsm.budget import BudgetGuard
from cvsm.config import load_config
from cvsm.datasets import load_split
from cvsm.eval_run import evaluate_run
from cvsm.llm import LLMClient
from cvsm.nli import Encoder, NLIModel
from cvsm.runner import Resources, run

MAIN = ["b1", "b2", "b3", "b4", "b5", "b6", "cvsm"]
ABL = [f"a{i}" for i in range(1, 9)]
TAU_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def plan(datasets, U, frozen_tau):
    jobs = []
    for d in datasets:
        # HaluEval evaluates the CVM as a detector against a single LLM judge (Section VI-D); other systems
        # have no detector mode and are reported as not applicable.
        for s in (MAIN if d != "halueval" else ["b1", "cvsm"]):
            for u in range(1, U + 1):
                jobs.append({"block": "main", "system": s, "dataset": d, "u": u, "k": 3, "tau": None})
    for d in [x for x in ("hotpotqa", "truthfulqa") if x in datasets]:
        for s in ABL:
            for u in range(1, U + 1):
                jobs.append({"block": "ablation", "system": s, "dataset": d, "u": u, "k": 3, "tau": None})
    if "hotpotqa" in datasets:
        for s in ("cvsm", "b5"):
            for k in (2, 5):
                for u in range(1, U + 1):
                    jobs.append({"block": "k_sweep", "system": s, "dataset": "hotpotqa", "u": u, "k": k, "tau": None})
        for t in TAU_GRID:
            if abs(t - frozen_tau) > 1e-9:
                jobs.append({"block": "tau_sweep", "system": "cvsm", "dataset": "hotpotqa", "u": 1, "k": 3, "tau": t})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="frozen config from scripts/tune.py")
    ap.add_argument("--n", nargs="+", required=True, help="dataset=n pairs (test split sizes)")
    ap.add_argument("--rho", nargs="+", required=True, help="main=,a7=,k2=,k5= calibration json files")
    ap.add_argument("--U", type=int, default=3)
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--fee", type=float, default=1.0)
    ap.add_argument("--sc-samples", type=int, required=True, help="B3 k', matched to CV-SM token cost")
    ap.add_argument("--fever-wiki-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    raw = yaml.safe_load(Path(a.config).read_text())
    if "frozen_from" not in raw:
        sys.exit("refusing to run the study on an unfrozen config (tune first, Section V-D)")
    n_by = {k: int(v) for k, v in (x.split("=") for x in a.n)}
    rho_by = {k: json.loads(Path(v).read_text())["rho_hat"] for k, v in (x.split("=") for x in a.rho)}
    if "fever" in n_by and not a.fever_wiki_dir:
        sys.exit("FEVER needs --fever-wiki-dir (evidence text for truth labels); refusing to run unlabellable runs")
    if any(v is None for v in rho_by.values()):
        sys.exit("a rho_hat is undefined; refusing to guess")
    jobs = plan(list(n_by), a.U, raw["params"]["tau"])
    manifest = Path("runs/study_manifest.jsonl")
    done = set()
    if manifest.exists():
        for l in manifest.read_text().splitlines():
            j = json.loads(l)
            done.add((j["block"], j["system"], j["dataset"], j["u"], j["k"], j["tau"]))
    todo = [j for j in jobs if (j["block"], j["system"], j["dataset"], j["u"], j["k"], j["tau"]) not in done]
    print(f"{len(jobs)} jobs planned, {len(todo)} to run")
    if a.dry_run:
        for j in todo:
            print(j)
        return
    guard = BudgetGuard(a.budget_usd, fee_multiplier=a.fee)
    base = load_config(a.config, "cvsm")
    llm = LLMClient(base.cache_dir)
    nli, enc, label_nli = NLIModel(base.nli_checkpoint), Encoder(base.encoder), NLIModel(base.label_nli_checkpoint)
    for j in todo:
        if not guard.allow():
            print("BUDGET STOP before", j, "spent", guard.spent())
            break
        rho_key = "a7" if j["system"] == "a7" else (f"k{j['k']}" if j["k"] != 3 else "main")
        ov = {"dataset": j["dataset"], "run_index": j["u"], "k": j["k"], "split": "test",
              "params": {"rho_hat": rho_by[rho_key]}, "switches": {"sc_samples": a.sc_samples}}
        if j["k"] != 3:
            ov["agents"] = raw[f"agents_k{j['k']}"]
        if j["tau"] is not None:
            ov["params"]["tau"] = j["tau"]
        cfg = load_config(a.config, j["system"], ov)
        cfg.name = f"study_{j['block']}_{j['dataset']}_u{j['u']}_k{j['k']}" + (f"_tau{j['tau']}" if j["tau"] else "")
        exs = load_split(cfg.dataset, cfg.data_dir, "test", n_by[j["dataset"]], 0)
        examples = {e.qid: e for e in exs}
        d = run(cfg, exs, Resources(cfg, llm=llm, nli=nli, encoder=enc), guard=guard)
        m = evaluate_run(d, cfg, examples, llm, label_nli, enc)
        guard.record(m.get("eval_cost", {}).get("cost_usd"), "eval", str(d))
        with open(manifest, "a") as f:
            f.write(json.dumps({**j, "dir": str(d)}) + "\n")
        print("done", j, "spent", round(guard.spent(), 3), flush=True)


if __name__ == "__main__":
    main()
