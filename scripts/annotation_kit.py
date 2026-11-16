"""Build the Section V-F human-annotation kit (no labels are filled in by this script).

Draws a stratified random sample of automatically labelled claims, balanced across systems and datasets
(equal count per (system, dataset) stratum), from the run folders in a manifest. Writes:
  annotation/annotator_A.xlsx, annotation/annotator_B.xlsx  identical blinded sheets (no system, no auto label)
  annotation/KEY_do_not_share.csv                           item -> system, dataset, run, automatic label
  annotation/README.md                                       guidelines for annotators
Run scripts/annotation_agreement.py once both sheets are filled.

  python scripts/annotation_kit.py --manifest runs/study_manifest.jsonl --per-stratum 30
"""
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from cvsm.datasets import load_split

LABELS = ["supported", "contradicted", "not_supported"]
GUIDE = """# Claim annotation guidelines (Section V-F)

You will see a question, the reference evidence for it, and one claim. Decide whether the **reference evidence
(together with the gold answer)** entails the claim:

- **supported**: the evidence states the claim or directly implies it.
- **contradicted**: the evidence states something incompatible with the claim.
- **not_supported**: the evidence neither states nor contradicts the claim, even if you believe it is true.

Judge only against the evidence shown, not your own knowledge. Do not discuss items with the other annotator.
You do not know which system produced a claim, and you should not try to find out. Use the `notes` column for
unclear cases. Fill the `label` column using the drop-down list.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--per-stratum", type=int, default=30)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="annotation")
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    entries = [json.loads(l) for l in Path(a.manifest).read_text().splitlines() if l.strip()]
    pools = defaultdict(list)
    for e in entries:
        if e["dataset"] == "halueval" or e["block"] not in ("main", "ablation"):
            continue
        lab = Path(e["dir"]) / "labels.jsonl"
        if not lab.exists():
            continue
        for j in map(json.loads, lab.read_text().splitlines()):
            if j["y"] is not None:
                pools[(e["system"], e["dataset"])].append({**j, "run": e["dir"], "u": e["u"]})
    rng = random.Random(a.seed)
    items = []
    for (system, dataset), pool in sorted(pools.items()):
        uniq = {(x["qid"], x["claim"]): x for x in pool}             # one copy of each claim per stratum
        pick = rng.sample(sorted(uniq.values(), key=lambda x: (x["qid"], x["claim"])), min(a.per_stratum, len(uniq)))
        items += [{**x, "system": system, "dataset": dataset} for x in pick]
    rng.shuffle(items)                                               # interleave strata: no system order cues
    exs = {}
    out = Path(a.out); out.mkdir(exist_ok=True)
    key_rows, sheet_rows = [], []
    for i, it in enumerate(items, 1):
        d = it["dataset"]
        if d not in exs:
            exs[d] = {e.qid: e for e in load_split(d, "data", a.split)}
        ex = exs[d][it["qid"]]
        item_id = f"C{i:04d}"
        sheet_rows.append([item_id, ex.question, ex.answer or "", "\n".join(ex.gold_evidence), it["claim"], "", ""])
        key_rows.append({"item_id": item_id, "system": it["system"], "dataset": d, "qid": it["qid"], "u": it["u"],
                         "run": it["run"], "auto_y": it["y"], "auto_verdict": it["verdict"], "auto_stage": it["stage"]})
    for who in ("A", "B"):
        wb = Workbook(); ws = wb.active; ws.title = "claims"
        ws.append(["item_id", "question", "gold_answer", "reference_evidence", "claim", "label", "notes"])
        for c in ws[1]:
            c.font = Font(bold=True)
        for r in sheet_rows:
            ws.append(r)
        dv = DataValidation(type="list", formula1='"' + ",".join(LABELS) + '"', allow_blank=True)
        ws.add_data_validation(dv); dv.add(f"F2:F{len(sheet_rows) + 1}")
        for col, w in zip("ABCDEFG", (9, 45, 18, 70, 50, 15, 30)):
            ws.column_dimensions[col].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row:
                c.alignment = Alignment(wrap_text=True, vertical="top")
        wb.save(out / f"annotator_{who}.xlsx")
    with open(out / "KEY_do_not_share.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key_rows[0])); w.writeheader(); w.writerows(key_rows)
    (out / "README.md").write_text(GUIDE + f"\nSample: {len(items)} claims, up to {a.per_stratum} per (system, dataset) "
                                   f"stratum across {len(pools)} strata, seed {a.seed}.\n")
    print(f"{len(items)} items in {len(pools)} strata -> {out}/")


if __name__ == "__main__":
    main()
