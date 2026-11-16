"""Estimate rho_hat by Eq. (4) on a calibration subset of the validation split (Section V-D).

Each pool agent answers each calibration question alone (role prompt, closed book, same answer format as
drafts); e_i(x) = 1 when its answer is not an exact match (HotpotQA normalisation). rho_hat is the mean
pairwise phi, clipped to [0, 1] for Eq. (5), with a bootstrap CI over items."""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from cvsm import equations as eq
from cvsm import prompts as P
from cvsm.config import load_config
from cvsm.datasets import exact_match, load_split
from cvsm.jsonutil import parse_answer
from cvsm.llm import Ledger, LLMClient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preset", default="cvsm")
    ap.add_argument("--n_boot", type=int, default=10_000)
    a = ap.parse_args()
    raw = yaml.safe_load(Path(a.config).read_text())
    cfg = load_config(a.config, a.preset)
    exs = load_split(cfg.dataset, cfg.data_dir, cfg.split, raw.get("calibration_n", 50), raw.get("calibration_offset", 50))
    llm, ledger = LLMClient(cfg.cache_dir), Ledger()
    pool = cfg.agent_pool()

    def ask(args):
        agent, ex = args
        user = P.DRAFT_USER.format(question=ex.question, subtask=ex.question, inputs="", fmt=P.ANSWER_FORMAT)
        r = llm.complete(agent.model, P.ROLE_PROMPTS[agent.role], user, temperature=cfg.draft_temperature,
                         max_tokens=cfg.max_tokens, tag=f"calib|seed{cfg.seed}|{ex.qid}|{agent.agent_id}",
                         ledger=ledger, component="calibration")
        ans = parse_answer(r.text)[0]
        return agent.agent_id, ex.qid, ans, int(exact_match(ans, ex.answer) == 0.0)

    jobs = [(ag, ex) for ag in pool for ex in exs]
    with ThreadPoolExecutor(max_workers=cfg.concurrency) as tp:
        rows = list(tp.map(ask, jobs))
    errors = {ag.agent_id: [r[3] for r in rows if r[0] == ag.agent_id] for ag in pool}
    rho, pairs = eq.mean_pairwise_rho(errors)
    rng = np.random.default_rng(0)
    boots = []
    n = len(exs)
    for _ in range(a.n_boot):
        idx = rng.integers(0, n, n)
        v, _ = eq.mean_pairwise_rho({k: [e[i] for i in idx] for k, e in errors.items()})
        if v is not None:
            boots.append(v)
    out = {
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "config": a.config, "preset": a.preset,
        "agents": [{"id": ag.agent_id, "role": ag.role, "model": ag.model.key()} for ag in pool],
        "items": [e.qid for e in exs], "error_rate": {k: float(np.mean(v)) for k, v in errors.items()},
        "pairwise_phi": pairs, "rho_hat_raw": rho,
        "rho_hat": None if rho is None else min(1.0, max(0.0, rho)),
        "bootstrap_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
        "n_bootstrap_defined": len(boots), "answers": rows, "cost": ledger.summary(),
    }
    dest = Path(cfg.runs_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg.name}_rho_{a.preset}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("error_rate", "pairwise_phi", "rho_hat_raw", "rho_hat", "bootstrap_ci95")}, indent=1))
    print("written", dest)


if __name__ == "__main__":
    sys.exit(main())
