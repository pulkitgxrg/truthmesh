# CV-SM implementation

This is an implementation of the system and evaluation protocol in `research.md` Part B (Sections III–V).
Every place where the specification was ambiguous, and the choice made, is listed in `DESIGN_NOTES.md`.

## Layout

```
src/cvsm/
  equations.py   Eqs. (2)-(7), (9)-(17), (19)-(22) as pure functions
  config.py      RunConfig; Table IV baselines/ablations as switch presets (SYSTEM_PRESETS)
  llm/           provider-agnostic client, disk cache keyed by prompt+params, providers, official prices
  nli.py         DeBERTa-v3 NLI and sentence encoder
  retrieval.py   BM25 retriever
  components.py  Task Decomposer, agents, CVM claim extraction, Arbiter, Synthesizer
  pipeline.py    CV-SM (Algorithm 1), with A1-A8 and B6 on the same code path; HaluEval detector mode
  baselines.py   B1-B5
  memory.py      Shared Memory Layer (Table II schema, append-only, versioned, transitive retraction)
  trace.py       execution-trace logging (steps, claims, exposures, memory writes)
  datasets.py    HotpotQA, FEVER, TruthfulQA and HaluEval loaders; seeded validation/test splits
  labelling.py   claim truth labels (NLI, then a cross-family LLM judge)
  evaluate.py    HR, VSR/FRR, PD and escape rate, MCR(t), CLR, ECE, Brier, coverage, EM/F1, cost
  stats.py       query-level bootstrap, McNemar, Holm, power analysis
  runner.py      writes a results folder per run
scripts/  run.py, estimate_rho.py, pilot_report.py, halueval_detect_metrics.py
configs/  pilot_hotpotqa.yaml (pilot), smoke_ollama.yaml (local plumbing test only)
tests/    unit tests for every equation, the write rule, retraction, each metric, and every Table IV switch
```

## Setup

```
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev]' pyarrow pandas
.venv/bin/python -m pytest -q
```

Datasets go in `data/raw/`; `datasets.SOURCES` gives the official URLs. API keys are read from
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `GEMINI_API_KEY`.

## Pilot

```
.venv/bin/python scripts/estimate_rho.py --config configs/pilot_hotpotqa.yaml
.venv/bin/python scripts/run.py --config configs/pilot_hotpotqa.yaml --presets cvsm b1 b5 --rho runs/<..>_rho_cvsm.json
.venv/bin/python scripts/pilot_report.py --runs runs/<cvsm> runs/<b1> runs/<b5> --rho runs/<..>.json
```

## Reproducibility

Every LLM response is cached at `cache/llm/<hash>.json`. The key covers provider, model, system prompt,
messages, temperature, max_tokens and a sampling tag (seed, run index, query, agent). With
`LLMClient(offline=True)`, any result can be recomputed from the cache, and a cache miss raises an error
instead of calling an API. Each run folder contains `config.json`, `environment.json` (model IDs, prices
used, NLI and encoder revisions, dataset hashes, package versions, date), `records/` (raw outputs, full
traces, cost), `labels.jsonl`, `per_query.json` and `metrics.json`.

## Reproducing the numbers in the paper

The paper reports a pilot only (50 HotpotQA validation questions; CV-SM, B1, B5; `configs/pilot_v2.yaml`).
Everything below runs offline from the logged runs in `runs/` and the LLM cache in `cache/`:

```
.venv/bin/python scripts/analyze.py --config configs/pilot_v2.yaml --manifest results_pilot/manifest.jsonl \
    --n-boot 10000 --out results_pilot --split validation
.venv/bin/python scripts/nli_veto_audit.py          # NLI veto audit of the first pilot run
.venv/bin/python scripts/pilot_results.py           # results_pilot/paper_numbers.json, RESULTS.md, LaTeX rows, figures
cp paper/code_original.tex paper/code.tex && .venv/bin/python scripts/update_paper.py   # writes CHANGES_paper.md
cd paper && tectonic -X compile code.tex
```

Not done, so not in the paper: B2–B4, B6, A1–A8, the k and τ sweeps, FEVER, TruthfulQA, HaluEval, parameter
tuning and human annotation. The code for all of them exists and is tested (`scripts/tune.py`, `run_study.py`,
`annotation_kit.py`, `annotation_agreement.py`).
