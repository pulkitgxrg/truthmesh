"""Precision / recall / F1 / accuracy (with bootstrap CIs) for HaluEval detector runs (Section VI-D)."""
import argparse
import json

from cvsm.runner import load_records
from cvsm.stats import bootstrap_ci


def prf(rows):
    tp = sum(1 for p, g in rows if p and g); fp = sum(1 for p, g in rows if p and not g)
    fn = sum(1 for p, g in rows if not p and g)
    prec = tp / (tp + fp) if tp + fp else None; rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
    acc = sum(1 for p, g in rows if p == g) / len(rows) if rows else None
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
    rows = [(r["output"]["predicted_hallucinated"], r["output"]["gold_hallucinated"]) for r in load_records(a.run)
            if r["status"] == "ok" and r["output"]["predicted_hallucinated"] is not None]
    out = {"n": len(rows), **prf(rows)}
    for k in ("precision", "recall", "f1", "accuracy"):
        try:
            out[k + "_ci95"] = bootstrap_ci(rows, lambda d, k=k: prf(d)[k], n_boot=10_000)[1:]
        except (IndexError, ValueError):
            out[k + "_ci95"] = None
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
