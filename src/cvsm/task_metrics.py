"""Dataset-specific task metrics (Section V-E): EM/F1 (HotpotQA, in datasets.py), FEVER label accuracy and
FEVER score, TruthfulQA % truthful and % truthful-and-informative (LLM judge from a family no agent uses,
as in V-F), HaluEval recognition accuracy (scripts/halueval_detect_metrics.py)."""
from __future__ import annotations

from . import prompts as P
from .jsonutil import parse_json

FEVER_LABELS = ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO")


def normalize_fever_label(answer: str) -> str | None:
    a = answer.strip().upper()
    if "NOT ENOUGH" in a or a in ("NEI", "UNKNOWN", "INSUFFICIENT"):
        return "NOT ENOUGH INFO"
    if a.startswith("REFUT") or a in ("FALSE", "NO", "REFUTED"):
        return "REFUTES"
    if a.startswith("SUPPORT") or a in ("TRUE", "YES", "SUPPORTED"):
        return "SUPPORTS"
    return None


def fever_label_correct(answer: str, gold: str) -> int:
    return int(normalize_fever_label(answer) == gold)


def fever_score(pred_label: str | None, pred_evidence: list[tuple[str, int]], gold: str,
                gold_sets: list[list[tuple[str, int]]], max_evidence: int = 5) -> int:
    """FEVER score [51]: label correct and, unless NEI, the first `max_evidence` predicted sentences contain
    every sentence of at least one gold evidence set."""
    if pred_label != gold:
        return 0
    if gold == "NOT ENOUGH INFO":
        return 1
    pred = set(map(tuple, pred_evidence[:max_evidence]))
    return int(any(s and set(map(tuple, s)) <= pred for s in gold_sets))


def truthfulqa_judge(llm, judge_model, ex, answer: str, ledger=None) -> dict:
    user = P.TQA_JUDGE_USER.format(question=ex.question, correct="\n".join(f"- {c}" for c in ex.gold_evidence),
                                   incorrect="\n".join(f"- {c}" for c in ex.meta.get("incorrect_answers", [])),
                                   answer=answer or "I have no comment.")
    for suffix in ("", "\n\nReturn valid JSON only, with no other text."):
        res = llm.complete(judge_model, P.TQA_JUDGE_SYSTEM, user + suffix, temperature=0.0, max_tokens=64,
                           ledger=ledger, component="eval:tqa_judge")
        try:
            o = parse_json(res.text)
            return {"truthful": bool(o["truthful"]), "informative": bool(o["informative"])}
        except (ValueError, KeyError, TypeError):
            continue
    return {"truthful": None, "informative": None}
