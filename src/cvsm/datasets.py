"""Dataset loaders (Section V-A, Table III) with fixed-seed validation/test splits (Section V-G)."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import random
import re
import string
from pathlib import Path

from .types import Example, Passage

SPLIT_SEED = 13
VAL_FRACTION = 0.2

RAW_FILES = {
    "hotpotqa": "raw/hotpot_qa_distractor_validation.parquet",
    "fever": "raw/fever_shared_task_dev.jsonl",
    "truthfulqa": "raw/TruthfulQA.csv",
    "halueval": "raw/halueval_qa_data.json",
}

SOURCES = {
    "hotpotqa": "https://huggingface.co/datasets/hotpotqa/hotpot_qa (config distractor, split validation; "
                "mirror of hotpot_dev_distractor_v1.json - the CMU host was unreachable on 2026-10-04)",
    "fever": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "truthfulqa": "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv",
    "halueval": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
}


# ---------- HotpotQA official answer normalisation, EM and F1 ----------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred), normalize_answer(gold)
    if p in ("yes", "no", "noanswer") or g in ("yes", "no", "noanswer"):
        if p != g:
            return 0.0
    pt, gt = p.split(), g.split()
    common = collections.Counter(pt) & collections.Counter(gt)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(pt), same / len(gt)
    return 2 * prec * rec / (prec + rec)


# ---------- loaders ----------
def load_hotpotqa(data_dir: Path) -> list[Example]:
    import pyarrow.parquet as pq
    rows = pq.read_table(data_dir / RAW_FILES["hotpotqa"]).to_pylist()
    out = []
    for r in rows:
        titles, sents = r["context"]["title"], r["context"]["sentences"]
        paras = {t: s for t, s in zip(titles, sents)}
        passages = [Passage(pid=f"{r['id']}:{t}", title=t, text=" ".join(s).strip()) for t, s in zip(titles, sents)]
        gold = []
        for t, i in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"]):
            if t in paras and 0 <= i < len(paras[t]):
                gold.append(f"{t}: {paras[t][i].strip()}")
        out.append(Example(qid=r["id"], dataset="hotpotqa", question=r["question"], answer=r["answer"],
                           gold_evidence=gold, passages=passages,
                           meta={"type": r["type"], "level": r["level"],
                                 "supporting_facts": list(zip(r["supporting_facts"]["title"],
                                                              r["supporting_facts"]["sent_id"]))}))
    return out


def load_fever(data_dir: Path, wiki_dir: Path | None = None) -> list[Example]:
    """FEVER labelled dev. Evidence sentence text requires the FEVER wiki-pages dump (wiki_dir); without it
    gold_evidence is empty and claim labelling for FEVER cannot run (reported, not imputed)."""
    wiki = _load_fever_wiki(wiki_dir) if wiki_dir else None
    out = []
    for line in (data_dir / RAW_FILES["fever"]).read_text().splitlines():
        r = json.loads(line)
        ev_sets = [[(e[2], e[3]) for e in s if e[2] is not None] for s in r["evidence"]]
        gold = []
        if wiki:
            for s in ev_sets:
                for page, sid in s:
                    txt = wiki.get(page, {}).get(sid)
                    if txt:
                        gold.append(f"{page.replace('_', ' ')}: {txt}")
        out.append(Example(qid=str(r["id"]), dataset="fever",
                           question=f"Is the following claim supported or refuted, or is there not enough "
                                    f"information? Claim: {r['claim']}",
                           answer=r["label"], gold_evidence=sorted(set(gold)),
                           meta={"claim": r["claim"], "evidence_sets": ev_sets}))
    return out


def _load_fever_wiki(wiki_dir: Path) -> dict:
    pages: dict[str, dict[int, str]] = {}
    for f in sorted(Path(wiki_dir).glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            sents = {}
            for ln in r.get("lines", "").split("\n"):
                parts = ln.split("\t")
                if len(parts) >= 2 and parts[0].isdigit():
                    sents[int(parts[0])] = parts[1]
            pages[r["id"]] = sents
    return pages


def load_truthfulqa(data_dir: Path) -> list[Example]:
    out = []
    with open(data_dir / RAW_FILES["truthfulqa"], newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            correct = [x.strip() for x in r["Correct Answers"].split(";") if x.strip()]
            incorrect = [x.strip() for x in r["Incorrect Answers"].split(";") if x.strip()]
            out.append(Example(qid=f"tqa{i}", dataset="truthfulqa", question=r["Question"], answer=r["Best Answer"],
                               gold_evidence=correct,
                               meta={"incorrect_answers": incorrect, "category": r["Category"], "type": r["Type"],
                                     "source": r["Source"]}))
    return out


def load_halueval(data_dir: Path) -> list[Example]:
    out = []
    lines = (data_dir / RAW_FILES["halueval"]).read_text().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        r = json.loads(line)
        rng = random.Random(f"halueval|{SPLIT_SEED}|{i}")
        hallucinated = rng.random() < 0.5            # one of the two answers is shown per item, fixed seed
        shown = r["hallucinated_answer"] if hallucinated else r["right_answer"]
        out.append(Example(qid=f"halu{i}", dataset="halueval", question=r["question"], answer=r["right_answer"],
                           gold_evidence=[r["knowledge"]],
                           passages=[Passage(pid=f"halu{i}:k", title="", text=r["knowledge"])],
                           meta={"shown_answer": shown, "is_hallucinated": hallucinated,
                                 "hallucinated_answer": r["hallucinated_answer"]}))
    return out


LOADERS = {"hotpotqa": load_hotpotqa, "fever": load_fever, "truthfulqa": load_truthfulqa, "halueval": load_halueval}


def split_ids(ids: list[str], seed: int = SPLIT_SEED, val_fraction: float = VAL_FRACTION) -> dict[str, list[str]]:
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    n_val = round(len(ids) * val_fraction)
    return {"validation": ids[:n_val], "test": ids[n_val:]}


def load_split(name: str, data_dir: str | Path, split: str, n: int | None = None, offset: int = 0,
               seed: int = SPLIT_SEED) -> list[Example]:
    """Deterministic: the split file is written once and re-used; `n` takes a prefix of the shuffled split."""
    data_dir = Path(data_dir)
    exs = LOADERS[name](data_dir)
    split_file = data_dir / "splits" / f"{name}_seed{seed}.json"
    if split_file.exists():
        splits = json.loads(split_file.read_text())
    else:
        splits = split_ids([e.qid for e in exs], seed)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        split_file.write_text(json.dumps(splits))
    by_id = {e.qid: e for e in exs}
    ids = splits[split][offset:]
    if n is not None:
        ids = ids[:n]
    return [by_id[i] for i in ids]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
