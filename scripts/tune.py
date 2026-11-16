"""Tune CV-SM thresholds and weights on the validation split (Section V-D), then freeze the config.

The full V-D grid (3*3*2*66*6*2*2 = 28,512 configurations) is infeasible, so this runs one pass of
coordinate search over exactly the V-D value sets, in the order tau -> (alpha,beta,gamma) simplex (step 0.1)
-> theta_e -> theta_c -> theta_sim -> delta -> Delta, starting from the config's values (DESIGN_NOTES #53).
Each candidate runs CV-SM on the same validation subset and is labelled; selection criterion (Eq. 8):
lowest validation HR among candidates with coverage >= u_min (and mean cost <= cost_cap if given).
If no candidate is feasible, the most-covering candidate is kept and this is logged.

Every candidate's metrics are written to <out>/tuning_log.jsonl; the winner to configs/frozen_<name>.yaml.
LLM calls are cached, so candidates share every call whose prompt is unchanged (e.g. first-sub-task drafts).

  python scripts/tune.py --config configs/study.yaml --rho runs/<rho>.json --n 100 --offset 100 \
      --u-min 0.8 --budget-usd 20 --fee 1.1
"""
import argparse
import copy
import itertools
import json
import sys
import time
from pathlib import Path

import yaml

from cvsm.budget import BudgetGuard
from cvsm.config import load_config
from cvsm.datasets import load_split
from cvsm.eval_run import evaluate_run
from cvsm.llm import LLMClient
from cvsm.nli import Encoder, NLIModel
from cvsm.runner import Resources, run

GRID = {
    "tau": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    "weights": [(round(a / 10, 1), round(b / 10, 1), round((10 - a - b) / 10, 1))
                for a in range(11) for b in range(11 - a)],            # 66 points on the simplex
    "theta_e": [0.5, 0.7, 0.9],
    "theta_c": [0.5, 0.7, 0.9],
    "theta_sim": [0.6, 0.75],
    "delta": [0.3, 0.5],
    "Delta": [0.1, 0.2],
}
ORDER = ["tau", "weights", "theta_e", "theta_c", "theta_sim", "delta", "Delta"]


def as_params(p: dict, key: str, val) -> dict:
    q = copy.deepcopy(p)
    if key == "weights":
        q["alpha"], q["beta"], q["gamma"] = val
    else:
        q[key] = val
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rho", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--offset", type=int, default=100)
    ap.add_argument("--u-min", type=float, default=0.8)
    ap.add_argument("--cost-cap", type=float, default=None, help="max mean system USD per query (Eq. 8 B)")
    ap.add_argument("--budget-usd", type=float, required=True)
    ap.add_argument("--fee", type=float, default=1.0)
    a = ap.parse_args()

    raw = yaml.safe_load(Path(a.config).read_text())
    rho = json.loads(Path(a.rho).read_text())["rho_hat"]
    if rho is None:
        sys.exit("rho_hat undefined; refusing to guess")
    base = load_config(a.config, "cvsm", {"params": {"rho_hat": rho}})
    exs = load_split(base.dataset, base.data_dir, base.split, a.n, a.offset)
    examples = {e.qid: e for e in exs}
    llm = LLMClient(base.cache_dir)
    nli, enc, label_nli = NLIModel(base.nli_checkpoint), Encoder(base.encoder), NLIModel(base.label_nli_checkpoint)
    guard = BudgetGuard(a.budget_usd, fee_multiplier=a.fee)
    out = Path(base.runs_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{base.name}_tuning"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "tuning_log.jsonl"

    params = {k: getattr(base.params, k) for k in ("theta_e", "theta_c", "theta_sim", "alpha", "beta", "gamma",
                                                    "tau", "delta", "Delta", "r", "p")}
    seen: dict[str, dict] = {}

    def evaluate(p: dict) -> dict | None:
        key = json.dumps(p, sort_keys=True)
        if key in seen:
            return seen[key]
        if not guard.allow():
            return None
        cfg = load_config(a.config, "cvsm", {"params": {**p, "rho_hat": rho}})
        cfg.name = f"{base.name}_tune"
        d = run(cfg, exs, Resources(cfg, llm=llm, nli=nli, encoder=enc), guard=guard)
        m = evaluate_run(d, cfg, examples, llm, label_nli, enc)
        guard.record(m.get("eval_cost", {}).get("cost_usd"), "eval", str(d))
        recs = [json.loads(l) for l in (d / "records.jsonl").read_text().splitlines()]
        complete = all(r["status"] == "ok" for r in recs)
        res = {"params": p, "run": str(d), "HR": m.get("HR"), "coverage": m.get("coverage"), "EM": m.get("EM"),
               "cost_per_query": (m.get("cost") or {}).get("cost_usd", {}) and m["cost"]["cost_usd"]["mean"],
               "complete": complete, "spent_total_usd": guard.spent()}
        res["feasible"] = bool(complete and res["coverage"] is not None and res["coverage"] >= a.u_min and
                               res["HR"] is not None and (a.cost_cap is None or (res["cost_per_query"] or 0) <= a.cost_cap))
        with open(log, "a") as f:
            f.write(json.dumps(res) + "\n")
        print(json.dumps(res), flush=True)
        seen[key] = res
        return res

    def better(x, y):           # is x better than y under Eq. (8)?
        if y is None:
            return x is not None
        if x is None:
            return False
        if x["feasible"] != y["feasible"]:
            return x["feasible"]
        if x["feasible"]:
            return x["HR"] < y["HR"]
        return (x["coverage"] or 0) > (y["coverage"] or 0)

    best = evaluate(params)
    stopped = best is None
    for key in ORDER:
        if stopped:
            break
        for val in GRID[key]:
            cand = as_params(best["params"], key, val)
            r = evaluate(cand)
            if r is None:
                stopped = True
                break
            if better(r, best):
                best = r
    summary = {"best": best, "stopped_by_budget": stopped, "u_min": a.u_min, "cost_cap": a.cost_cap,
               "n_validation": len(exs), "offset": a.offset, "rho_hat": rho, "n_candidates": len(seen),
               "spent_total_usd": guard.spent(), "feasible_found": bool(best and best["feasible"])}
    (out / "tuning_summary.json").write_text(json.dumps(summary, indent=1))
    if best:
        frozen = copy.deepcopy(raw)
        frozen["params"] = {**raw.get("params", {}), **best["params"], "rho_hat": rho}
        frozen["frozen_from"] = {"tuning_dir": str(out), "date": time.strftime("%Y-%m-%d"),
                                 "feasible": best["feasible"], "stopped_by_budget": stopped}
        dest = Path("configs") / f"frozen_{base.name}.yaml"
        dest.write_text(yaml.safe_dump(frozen, sort_keys=False))
        print("frozen config:", dest)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
