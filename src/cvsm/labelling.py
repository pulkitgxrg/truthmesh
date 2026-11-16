"""Claim truth labels y(c) (Section V-F).

Stage 1: NLI with a checkpoint different from the system's, against gold evidence E*(q).
Stage 2: claims stage 1 cannot decide go to an LLM judge from a model family not used by any agent.
Stage 3 (human annotation) is exported by scripts/annotation_kit.py and scored by
scripts/annotation_agreement.py; the labels themselves are produced by human annotators only.
y(c) = 1 iff E*(q) entails c; "contradicted" and "not_supported" are both y = 0 but kept apart."""
from __future__ import annotations

from dataclasses import dataclass

from . import prompts as P
from .jsonutil import parse_json
from .types import Example, ModelSpec


@dataclass
class LabelConfig:
    theta_ent: float = 0.9
    theta_con: float = 0.9


def gold_premises(ex: Example) -> list[str]:
    if ex.dataset == "hotpotqa":
        ans = f'The answer to the question "{ex.question}" is {ex.answer}.'
        return [*ex.gold_evidence, " ".join(ex.gold_evidence) + " " + ans, ans]
    if ex.dataset == "truthfulqa":
        return [f"Q: {ex.question} A: {a}" for a in ex.gold_evidence]
    if ex.dataset == "fever":
        return [*ex.gold_evidence, " ".join(ex.gold_evidence)] if ex.gold_evidence else []
    return [*ex.gold_evidence]


class Labeller:
    def __init__(self, nli, llm, judge_model: ModelSpec | None, cfg: LabelConfig = LabelConfig(),
                 agent_families: set[str] = frozenset()):
        if judge_model is not None and judge_model.family() in agent_families:
            raise ValueError("LLM judge must come from a model family not used by any agent (Section V-F)")
        self.nli, self.llm, self.judge, self.cfg = nli, llm, judge_model, cfg
        self._memo: dict[tuple[str, str], dict] = {}

    def label(self, ex: Example, claim: str, ledger=None) -> dict:
        key = (ex.qid, claim)
        if key in self._memo:
            return self._memo[key]
        prem = gold_premises(ex)
        out = {"claim": claim, "stage": None, "verdict": None, "y": None, "nli_ent": None, "nli_con": None}
        if prem:
            sc = self.nli.scores([(p, claim) for p in prem])
            ent, con = max(s.ent for s in sc), max(s.con for s in sc)
            out.update(nli_ent=ent, nli_con=con)
            if ent >= self.cfg.theta_ent and con < self.cfg.theta_con:
                out.update(stage="nli", verdict="supported", y=1)
            elif con >= self.cfg.theta_con and ent < self.cfg.theta_ent:
                out.update(stage="nli", verdict="contradicted", y=0)
        if out["y"] is None and self.judge is not None and prem:
            ev = "\n".join(f"- {e}" for e in (ex.gold_evidence or prem))
            ans = ex.answer if ex.dataset != "truthfulqa" else f"{ex.answer} (other acceptable: {'; '.join(ex.gold_evidence)})"
            if ex.dataset == "truthfulqa":
                ev += "\nIncorrect answers:\n" + "\n".join(f"- {x}" for x in ex.meta.get("incorrect_answers", []))
            user = P.JUDGE_USER.format(question=ex.question, answer=ans, evidence=ev, claim=claim)
            v = ""
            for attempt, suffix in enumerate(("", "\n\nReturn valid JSON only, with no other text.")):
                res = self.llm.complete(self.judge, P.JUDGE_SYSTEM, user + suffix, temperature=0.0, max_tokens=128,
                                        ledger=ledger, component="eval:judge" + (":retry" if attempt else ""))
                try:
                    v = str(parse_json(res.text).get("verdict", "")).strip().lower()
                except (ValueError, AttributeError):
                    v = ""
                if v in ("supported", "contradicted", "not_supported"):
                    break
            if v in ("supported", "contradicted", "not_supported"):
                out.update(stage="judge", verdict=v, y=1 if v == "supported" else 0)
            else:
                out.update(stage="judge", verdict="judge_parse_error")
        if out["y"] is None and out["stage"] is None:
            out["stage"] = "undecided"
        self._memo[key] = out
        return out
