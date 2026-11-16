"""After both annotators fill their sheets: Cohen's kappa, adjudication sheet for disagreements, and
precision/recall of the automatic labeller per system (positive class = hallucinated, y = 0), with
bootstrap CIs. If annotation/adjudicated.xlsx exists (third annotator), its labels resolve disagreements.

  python scripts/annotation_agreement.py --dir annotation
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from openpyxl import Workbook, load_workbook

from cvsm.agreement import cohen_kappa, precision_recall

LABELS = {"supported", "contradicted", "not_supported"}


def read(path):
    ws = load_workbook(path).active
    rows = list(ws.iter_rows(min_row=2, values_only=True))
    return {r[0]: (str(r[5]).strip().lower() if r[5] else None) for r in rows if r[0]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", default="annotation"); ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(); d = Path(a.dir)
    A, B = read(d / "annotator_A.xlsx"), read(d / "annotator_B.xlsx")
    key = {r["item_id"]: r for r in csv.DictReader(open(d / "KEY_do_not_share.csv"))}
    both = [i for i in key if A.get(i) in LABELS and B.get(i) in LABELS]
    missing = [i for i in key if i not in both]
    if not both:
        raise SystemExit("no item has valid labels from both annotators yet")
    k3 = cohen_kappa([A[i] for i in both], [B[i] for i in both])
    kb = cohen_kappa([int(A[i] == "supported") for i in both], [int(B[i] == "supported") for i in both])
    disagree = [i for i in both if A[i] != B[i]]
    adj = read(d / "adjudicated.xlsx") if (d / "adjudicated.xlsx").exists() else {}
    if disagree and not adj:
        src = load_workbook(d / "annotator_A.xlsx").active
        rows = {r[0]: r for r in src.iter_rows(min_row=2, values_only=True)}
        wb = Workbook(); ws = wb.active
        ws.append(["item_id", "question", "gold_answer", "reference_evidence", "claim", "label", "notes"])
        for i in disagree:
            ws.append(list(rows[i][:5]) + ["", ""])
        wb.save(d / "adjudication_needed.xlsx")
    human = {}
    for i in both:
        lab = A[i] if A[i] == B[i] else adj.get(i)
        if lab in LABELS:
            human[i] = 1 if lab == "supported" else 0
    by_sys = defaultdict(list)
    for i, y in human.items():
        by_sys[key[i]["system"]].append((int(key[i]["auto_y"]), y))
    rng = random.Random(0)
    res = {}
    for s, pairs in sorted(by_sys.items()):
        pr = precision_recall([p[0] for p in pairs], [p[1] for p in pairs])
        bs_p, bs_r = [], []
        for _ in range(a.n_boot):
            smp = [rng.choice(pairs) for _ in pairs]
            r = precision_recall([p[0] for p in smp], [p[1] for p in smp])
            if r["precision"] is not None: bs_p.append(r["precision"])
            if r["recall"] is not None: bs_r.append(r["recall"])
        pr["precision_ci"] = np.percentile(bs_p, [2.5, 97.5]).tolist() if bs_p else None
        pr["recall_ci"] = np.percentile(bs_r, [2.5, 97.5]).tolist() if bs_r else None
        res[s] = pr
    out = {"n_items": len(key), "n_double_labelled": len(both), "n_missing": len(missing),
           "cohen_kappa_3way": k3, "cohen_kappa_binary": kb, "n_disagreements": len(disagree),
           "n_adjudicated": sum(1 for i in disagree if adj.get(i) in LABELS),
           "labeller_vs_human_per_system (positive = hallucinated)": res}
    (d / "agreement_results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if disagree and not adj:
        print(f"{len(disagree)} disagreements written to {d}/adjudication_needed.xlsx; have a third annotator fill it, "
              f"save it as adjudicated.xlsx, and re-run.")


if __name__ == "__main__":
    main()
