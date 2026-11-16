"""Run configuration. Baselines B1-B6 and ablations A1-A8 (Table IV) are switch presets on one config."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

from .types import AgentSpec, ModelSpec


@dataclass
class Params:
    """Section V-D. Defaults are mid-grid starting points until validation tuning freezes them."""
    theta_e: float = 0.7
    theta_c: float = 0.7
    theta_sim: float = 0.6
    alpha: float = 0.4
    beta: float = 0.2
    gamma: float = 0.4
    tau: float = 0.5
    delta: float = 0.5
    Delta: float = 0.1
    r: int = 5
    p: int = 5
    rho_hat: float = 0.0       # set from the calibration run (Eq. 4); A6 forces 0


@dataclass
class Switches:
    """Each flag corresponds to one row of Table IV; CV-SM is all defaults."""
    system: str = "cvsm"             # cvsm | b1 | b2 | b3 | b4 | b5 | b6
    r1: bool = True                  # A1 -> False: agents read memory and peer drafts before drafting
    r2: bool = True                  # A2 -> False: every claim written; labels still used for the answer
    structured_handoff: bool = True  # A3 -> False: downstream gets a free-text summary of verified claims
    singletons: str = "evidence"     # A4 -> "reject"
    arbitration: str = "evidence"    # A5 -> "majority"
    correlation_discount: bool = True  # A6 -> False (rho_hat = 0 in Eq. 11)
    homogeneous: bool = False        # A7 -> True: one base model for all agents
    premise_cap: bool = True         # A8 -> False: Eq. (15) and transitive retraction disabled
    retraction: bool = True
    verification: bool = True       # B6 -> False (no alignment/arbitration/scoring)
    revision: bool = True            # B6 -> False
    synth_filter: bool = True        # B6 -> False
    use_decomposer: bool = True      # B1-B4 -> False
    retrieval_in_context: bool = False  # B2, B4, B5 -> True (retrieved passages in the agent prompt)
    draft_retrieval: bool = True     # CV-SM agents see the same top-p passages as B2/B5 when drafting (approved
                                     # interpretation of R1, DESIGN_NOTES #51); False = strictly closed-book
    multi_entry_synthesis: bool = True  # answer claim may be entailed by the set of entries the Synthesizer cites
    debate_rounds: int = 2           # B5: R
    sc_samples: int = 5              # B3: k' (to be matched to CV-SM token cost)


SYSTEM_PRESETS: dict[str, dict] = {
    "cvsm": {},
    "b1": {"system": "b1", "use_decomposer": False},
    "b2": {"system": "b2", "use_decomposer": False, "retrieval_in_context": True},
    "b3": {"system": "b3", "use_decomposer": False},
    "b4": {"system": "b4", "use_decomposer": False, "retrieval_in_context": True},
    "b5": {"system": "b5", "retrieval_in_context": True},
    "b6": {"system": "b6", "r1": False, "r2": False, "verification": False, "revision": False,
           "synth_filter": False},
    "a1": {"r1": False},
    "a2": {"r2": False},
    "a3": {"structured_handoff": False},
    "a4": {"singletons": "reject"},
    "a5": {"arbitration": "majority"},
    "a6": {"correlation_discount": False},
    "a7": {"homogeneous": True},
    "a8": {"premise_cap": False, "retraction": False},
}


@dataclass
class RunConfig:
    name: str
    preset: str
    dataset: str
    split: str
    n: int | None
    seed: int
    run_index: int                   # u in 1..U (Section V-G); enters the sampling tag
    k: int
    agents: list[AgentSpec]
    utility_model: ModelSpec         # decomposer, claim extraction, arbiter, synthesizer
    arbiter_model: ModelSpec
    eval_decomposer_model: ModelSpec
    judge_model: ModelSpec | None
    nli_checkpoint: str
    label_nli_checkpoint: str
    encoder: str
    params: Params = field(default_factory=Params)
    switches: Switches = field(default_factory=Switches)
    draft_temperature: float = 0.7
    max_tokens: int = 1024
    max_subtasks: int = 4
    cache_dir: str = "cache/llm"
    runs_dir: str = "runs"
    data_dir: str = "data"
    concurrency: int = 4

    def to_dict(self) -> dict:
        return asdict(self)

    def agent_pool(self) -> list[AgentSpec]:
        if self.switches.homogeneous:
            m = self.agents[0].model
            return [AgentSpec(a.agent_id, a.role, m) for a in self.agents]
        return self.agents

    def effective_rho(self) -> float:
        return self.params.rho_hat if self.switches.correlation_discount else 0.0


def _model(d: dict | None) -> ModelSpec | None:
    if d is None:
        return None
    return ModelSpec(provider=d["provider"], model=d["model"],
                     supports_temperature=d.get("supports_temperature", True))


def load_config(path: str | Path, preset: str | None = None, overrides: dict | None = None) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text())
    raw = copy.deepcopy(raw)
    if overrides:
        _deep_update(raw, overrides)
    preset = preset or raw.get("preset", "cvsm")
    sw = Switches(**{**raw.get("switches", {}), **SYSTEM_PRESETS[preset]})
    params = Params(**raw.get("params", {}))
    agents = [AgentSpec(a["agent_id"], a["role"], _model(a["model"])) for a in raw["agents"]]
    util = _model(raw["utility_model"])
    return RunConfig(
        name=raw.get("name", preset), preset=preset, dataset=raw["dataset"], split=raw.get("split", "validation"),
        n=raw.get("n"), seed=raw.get("seed", 13), run_index=raw.get("run_index", 1), k=raw.get("k", 3),
        agents=agents, utility_model=util, arbiter_model=_model(raw.get("arbiter_model")) or util,
        eval_decomposer_model=_model(raw.get("eval_decomposer_model")) or util,
        judge_model=_model(raw.get("judge_model")),
        nli_checkpoint=raw.get("nli_checkpoint", "cross-encoder/nli-deberta-v3-base"),
        label_nli_checkpoint=raw.get("label_nli_checkpoint",
                                     "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"),
        encoder=raw.get("encoder", "sentence-transformers/all-MiniLM-L6-v2"),
        params=params, switches=sw, draft_temperature=raw.get("draft_temperature", 0.7),
        max_tokens=raw.get("max_tokens", 1024), max_subtasks=raw.get("max_subtasks", 4),
        cache_dir=raw.get("cache_dir", "cache/llm"), runs_dir=raw.get("runs_dir", "runs"),
        data_dir=raw.get("data_dir", "data"), concurrency=raw.get("concurrency", 4),
    )


def _deep_update(d: dict, u: dict) -> None:
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
