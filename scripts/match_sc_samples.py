"""B3's k' (Table IV): the number of self-consistency samples whose token cost matches CV-SM's.
k' = round(mean CV-SM tokens per query / mean tokens per single B1 sample), from logged runs."""
import argparse
import statistics

from cvsm.runner import load_records

ap = argparse.ArgumentParser()
ap.add_argument("--cvsm", nargs="+", required=True); ap.add_argument("--b1", nargs="+", required=True)
a = ap.parse_args()
tok = lambda dirs: statistics.mean(r["cost"]["system"]["input_tokens"] + r["cost"]["system"]["output_tokens"]
                                   for d in dirs for r in load_records(d) if r["status"] == "ok")
cv, b1 = tok(a.cvsm), tok(a.b1)
print({"cvsm_tokens_per_query": round(cv, 1), "b1_tokens_per_sample": round(b1, 1), "k_prime": max(1, round(cv / b1))})
