"""Audit of grounding vetoes in the v1 pilot (system NLI = cross-encoder/nli-deberta-v3-base): for clusters with
G <= -0.9, re-score the most-contradicting retrieved passage with the ling-wanli checkpoint on a seeded random
sample. Output: results_pilot/nli_veto_audit.json."""
import json
import random
import sys

from cvsm.datasets import load_split
from cvsm.nli import NLIModel

run = sys.argv[1] if len(sys.argv) > 1 else "runs/20261004-140927_pilot_cvsm"
recs = [json.loads(l) for l in open(f"{run}/records.jsonl")]
exs = {e.qid: e for e in load_split("hotpotqa", "data", "validation", 50, 0)}
pairs = []
for r in recs:
    ps = {p.pid: p for p in exs[r["qid"]].passages}
    for e in r["trace"]["events"]:
        if e["kind"] == "score" and e["G"] <= -0.9:
            ev = max(e["evidence"], key=lambda x: x["con"])
            pairs.append((ps[ev["pid"]].as_text(), e["rep_text"], ev["con"]))
random.Random(0).shuffle(pairs)
sample = pairs[:60]
m = NLIModel("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
sc = m.scores([(p, h) for p, h, _ in sample])
out = {"run": run, "n_vetoes_G_le_-0.9": len(pairs), "sample_size": len(sample), "seed": 0,
       "large_contradiction": sum(s.con >= 0.5 for s in sc), "large_neutral": sum(s.neu >= 0.5 for s in sc),
       "large_entailment": sum(s.ent >= 0.5 for s in sc),
       "items": [{"claim": h, "passage": p[:300], "base_con": c, "large": [s.ent, s.neu, s.con]}
                 for (p, h, c), s in zip(sample, sc)]}
open("results_pilot/nli_veto_audit.json", "w").write(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if k != "items"})
