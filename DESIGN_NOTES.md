# Design notes: interpretations of the specification

This file records every place where research.md Part B left a choice open, and the choice the code makes.
The rule throughout was to take the simplest reading that stays faithful to the text. Items marked
**[DECIDE]** change results materially and need the authors' confirmation before the main runs.

Section and equation numbers refer to research.md Part B. Code locations are given as `module:function`.

## A. Data and environment

1. **HotpotQA source.** The official file `hotpot_dev_distractor_v1.json` (curtis.ml.cmu.edu) timed out on
   2026-10-04. The loader reads the Hugging Face mirror `hotpotqa/hotpot_qa`, config `distractor`, split
   `validation`. It has 7,405 rows, the official dev count. The file's sha256 is recorded in each run's
   `environment.json`.
2. **TruthfulQA size [DECIDE].** The current official `TruthfulQA.csv` (sylinrl/TruthfulQA, main) has
   **790** rows, not the 817 stated in Table III and Section V-A. Upstream revised the file after the
   original release. Options: (a) use the current 790-row file and correct the paper; (b) pin the
   historical 817-row revision. The code currently loads whatever file is in `data/raw/`.
3. **TruthfulQA split [DECIDE].** Table III says "all 817 questions", but Section V-G says every dataset is
   split into validation and test. The code follows V-G: a seeded 20/80 split (158 / 632 of 790). Reporting
   on "all" questions would include the items used for tuning.
4. **Splits.** Every dataset is shuffled with seed 13 and split 20% validation / 80% test
   (`datasets.split_ids`). The id lists are written once to `data/splits/` and reused. The pilot uses
   validation[0:50]; ρ̂ calibration uses validation[50:100], which does not overlap the pilot.
5. **FEVER evidence text.** `shared_task_dev.jsonl` gives evidence as (page, sentence id). The sentence text
   needs the FEVER `wiki-pages` dump (2017 snapshot, about 1.7 GB), which has not been downloaded. Without
   it, FEVER claim labels cannot be computed, and the code does not impute them.
6. **Retrieval corpus [DECIDE].** Section V-B specifies "a fixed snapshot of English Wikipedia". For
   HotpotQA in the *distractor* setting (Table III), the code retrieves with BM25 over the question's own
   10 paragraphs (2 gold + 8 distractors). That is the closed corpus the distractor setting defines. A
   full-Wikipedia index is not built. Consequences:
   - grounding (Eq. 13) and arbitration see the gold paragraphs, which makes them easier than open
     retrieval would;
   - the same paragraphs contain the gold supporting facts used for truth labels. Section VII-F's
     mitigation still applies: labelling uses a different NLI checkpoint plus an LLM judge.
   - TruthfulQA has no corpus. HaluEval items carry their own knowledge passage. FEVER would use the
     wiki-pages snapshot. **A corpus decision is needed for TruthfulQA.**
7. **Models.** NLI inside CV-SM: `cross-encoder/nli-deberta-v3-base` (DeBERTa-v3, SNLI+MNLI). Labelling NLI,
   which must differ per VII-F: `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`. Sentence
   encoder: `sentence-transformers/all-MiniLM-L6-v2` (Sentence-BERT). Snapshot commit hashes are recorded
   in `environment.json`.
   - Observation: the base cross-encoder gave P_con = 0.69 for a compatible, unrelated pair ("directed by
     X" vs "X was born in 1970"). With θ_c = 0.7 this is a near miss, so the choice of θ_c matters.
8. **Prices** were copied from the official pricing pages on 2026-10-04 (`llm/pricing.py`, with URLs). A
   model without a recorded price reports cost `None`, never an estimate.
9. **Anthropic SDK 1.x** removed `temperature` from `messages.create`. Claude Haiku 4.5 still accepts it,
   so it is sent through `extra_body`. Models that reject sampling parameters (Opus 4.7+, Sonnet 5.x) need
   `supports_temperature: false`. Drafts then use the model's default sampling, which must be reported.

## B. CV-SM (Section IV, Algorithm 1)

10. **R1 and retrieval [DECIDE].** R1 says the draft context is "the query, the sub-task description, its
    role prompt and the dependency inputs I_j, and nothing else". The code follows this literally, so
    CV-SM drafts are closed-book. External evidence enters only through grounding (Eq. 13), the Arbiter
    and supersession. The "retriever" role in the agent pool (IV-B) is therefore a role prompt, not a
    tool. Baselines B2, B4 and B5 *do* get retrieved passages in context ("with retrieval", Table IV). If
    the intent was that agents may retrieve before committing their draft, change
    `components.agent_respond` and record the change.
11. **Session = one query.** Algorithm 1 line 1 starts each query with an empty store, and IV-H scopes the
    store to a session. MCR(t) (Eq. 7) is therefore a within-query curve. t counts every appended version
    (writes and status changes) and is averaged over queries at each t. H4 ("growth over session length")
    is then tested over short sessions of about 2 to 30 writes.
12. **Task decomposition.** The decomposer returns at most `max_subtasks` = 4 sub-tasks, and the last one
    must answer the query. Dependencies may only point to earlier ids, which guarantees a DAG. If the
    output cannot be parsed, the fallback is one sub-task equal to the query, logged as
    `decomposition_fallback`.
13. **Agent assignment.** When the pool has exactly k agents, every sub-task gets all of them. Otherwise
    agents are assigned round-robin.
14. **Batched decomposition.** One CVM call decomposes all k drafts of a sub-task. This matches the
    2k + 1 + n_con call count in IV-J. The prompt forbids adding or merging claims across responses.
15. **Exposure (Eq. 2).** X_ij is the texts of the memory entries in the agent's context plus the claims of
    any steps in its context; for A3 it is the summary text. An agent is exposed to κ if the system NLI
    gives P_ent(x ⇒ m) ≥ θ_e for some x ∈ X_ij and some member m of κ, evaluated over all pairs without
    a similarity prefilter. As III-B states, dependency inputs count as exposure.
16. **Alignment (Eq. 9).** The NLI only scores candidate pairs with cosine ≥ θ_sim. A pair that is
    equivalent is not also tested for contradiction. Contradictions inside one cluster are ignored
    (κ′ ≠ κ in Eq. 10).
17. **Cluster representative.** M, G and Conf for a cluster are computed on its medoid claim (the member
    with the highest summed cosine to the others). y(κ) is the label of that claim.
18. **Agreement for singletons.** A(κ) follows Eq. (11) for every cluster. A singleton with
    |S_ind| = 1 gets 1/k, as IV-G states. A cluster whose asserters were all exposed has |S_ind| = 0 and
    gets A = 0 under Eq. (11); IV-G's "A = 1/k" is read as describing the usual one-asserter singleton.
19. **Contested groups with more than two clusters.** The Arbiter receives every claim in the
    contradiction component, anonymised and shuffled, and chooses one, "neither" or "insufficient". With
    two clusters this is exactly IV-G. The winning cluster is labelled Resolved and G is computed from the
    cited passage only. The other clusters are rejected. "Neither" rejects all of them. "Insufficient", a
    parse failure, or a citation that fails the NLI check (P_ent < θ_e) leaves all of them Unresolved.
20. **A5 (majority arbitration).** The cluster with the largest |S| wins and G is computed normally. A tie
    leaves the group Unresolved.
21. **Premise cap (Eq. 15).** 𝒫(c) is the union of the members' recorded premise ids that lie in I_j. If
    none are recorded and I_j is non-empty, the cap is over all of I_j. If I_j is empty, for example
    because nothing was admitted from the parents, there is no cap. A premise that is no longer active
    contributes Conf = 0.
22. **Supersession (Eq. 17).** Contradicted active entries are those with cosine ≥ θ_sim and system-NLI
    P_con ≥ θ_c in either direction. All decisions are made before anything is applied. c is written only
    if every contradicted entry is either superseded by margin or ruled against by the Arbiter (with the
    NLI check). Any other outcome blocks c and leaves e unchanged. Supersession is off when R2 is off
    (A2, B6).
23. **Transitive retraction.** "Flagged for re-verification" is implemented as re-checking Eq. (16) with
    Conf* recomputed from the stored A, M and G and the current status of the premises. Because a
    superseded or retracted premise contributes 0, a flagged entry is retracted whenever τ > 0, and the
    process continues to its descendants.
24. **Revision (IV-I).** Feedback lists the agent's claims that were contested, unresolved, rejected or
    left pending. A revised claim that an active entry entails (top-r by similarity, P_ent ≥ θ_e) is
    accepted. Every other revised claim is scored as a singleton (A = 1/k) and goes through Eq. (16).
    Claims are processed in order, so a duplicate is caught by the entry just written.
25. **Synthesis and abstention.** The Synthesizer returns {answer, answer_claim, statement}. The answer
    and statement are decomposed, and only claims entailed by a single active entry (P_ent ≥ θ_e) are
    kept. The system abstains if the answer is UNRESOLVED or `answer_claim` is not entailed by an active
    entry. The "present competing alternatives" variant is not implemented: an unresolved answer becomes
    an abstention, which counts against coverage.

## C. Baselines and ablations (Table IV)

26. **B1 / B2.** One agent with a generic assistant prompt and agent 1's model, no decomposition. B2 adds
    the top-p BM25 passages for the question.
27. **B3.** k′ samples at the draft temperature, with a majority vote on normalised short answers. k′ = 5
    is a placeholder; Table IV requires k′ to match CV-SM's token cost, so it must be set from the pilot
    cost before main runs.
28. **B4 (CoVe, factored, with retrieval).** The draft is closed-book. The model plans up to 5
    verification questions. Each is answered without the draft in context, using top-p passages
    retrieved for that question. A final call revises the draft.
29. **B5 (debate).** Uses the same decomposer, roles and models as CV-SM. Each sub-task is debated for
    R = 2 rounds, where R counts the independent first round. All agents get top-p passages for the
    sub-task. Later rounds see the peers' previous answers. The final round's majority (normalised short
    answer; ties go to the lowest-index agent) is passed downstream as free text. The query answer is the
    majority of the last sub-task.
30. **B6 (unguarded shared memory).** Agents draft sequentially. Each one reads all of memory and the
    drafts of earlier peers. Every extracted claim is written as `Unverified`. There is no alignment,
    arbitration or revision, and the Synthesizer runs without the entailment filter.
31. **A1.** Sequential drafting in which each agent reads all memory and the earlier peers' drafts. The
    rest of CV-SM is unchanged.
32. **A2.** Every scored cluster is written, including contested and unresolved ones, without
    supersession. Labels are still stored and shown to the Synthesizer, and the final filter still
    applies.
33. **A3.** I_j reaches agents as an LLM-written prose summary with no ids, so claims cannot cite premises
    and are capped over all of I_j (Eq. 15). The structured store is still used for M, the write rule and
    synthesis.
34. **A6.** ρ̂ = 0 in Eq. (11). **A7.** Every agent uses agent 1's model. ρ̂ must be re-estimated for this
    configuration (V-D). **A8.** No Eq. (15) cap and no transitive retraction.

## D. Metrics and labels (Sections III, V-E, V-F)

35. **Truth labels [DECIDE].** Following III-A literally, y(c) = 1 iff the gold evidence E*(q) entails c.
    For HotpotQA, E* is the supporting-fact sentences plus the sentence "The answer to Q is A". A true
    background fact that is not in the supporting facts is therefore y = 0 ("not_supported"). The
    contradicted / not_supported split is logged so Section VI can report both, as III-A requires. This
    definition penalises verbose answers; it is the paper's definition, but the authors should confirm
    it.
36. **Labelling stages.** NLI decides only when one side is ≥ 0.9 and the other is not. Everything else
    goes to an LLM judge from a family not used by any agent; the code enforces this. With no judge
    configured, a claim stays `undecided` and is excluded and counted, never imputed. Human annotation
    (stage 3) is not automated.
37. **Uniform final decomposition.** HR (Eq. 19) uses the claims that the same evaluation decomposer
    (model, prompt, temperature 0) extracts from every system's final output text. CV-SM's internal
    claims are not reused for this.
38. **"Present in the final answer"** (VSR blocking and escape rate) means the final answer *text*, used as
    a single premise, entails the claim (labelling NLI, P_ent ≥ 0.5). An earlier version compared the claim
    with the separately decomposed final claims. The pilot probe showed that this fails when the two
    decompositions phrase the same content differently (one dropped the entity name), which inflated VSR.
39. **Metric equivalence.** Presence in the final answer, PD classes, CLR clusters and post-hoc exposure
    use bidirectional entailment (Eq. 9) under the *labelling* NLI at θ = 0.5. Candidates are pairs that
    are textually identical or have cosine ≥ θ_sim. The same instrument is used for every system, as
    required by VII-F (construct validity of PD).
40. **VSR / FRR (Eq. 21).** C_int is every claim decomposed from every step, including the Synthesizer's
    output before filtering. A claim is blocked if it is in no written entry's source claims and is not
    equivalent to any final claim. Baseline traces are decomposed with the same CVM prompt; those calls
    go to an *instrumentation* ledger and are not counted in the baseline's cost.
41. **PD (Eq. 6).** Steps are agent outputs (drafts, revisions, debate rounds, verification answers) and
    the synthesis step. σ′ depends on σ if σ′'s context held σ's output or a memory entry whose source
    steps include σ. A re-assertion with no dependency path from o(h) is independent, so it is excluded
    from the max. If every re-assertion is excluded, PD(h) = 0. The escape rate is the fraction of
    hallucinated claim classes that have a member in the final answer.
42. **CLR (Eq. 3).** Computed post hoc in the same way for every multi-agent system. Agent assertions are
    clustered within each sub-task. S counts the agents asserting; S_ind counts those whose *earliest*
    asserting step had no context item that entails a member. A cluster is accepted if it was written to
    memory (CV-SM family, B6) or appears in the winning response (B5). y(κ) is the majority label of its
    members.
43. **ECE / Brier (Eq. 22).** Computed over every scored cluster (including revision singletons and
    rejected or unresolved clusters), using Conf* and y(rep), with B = 10 equal-width bins.
44. **ρ̂ (Eq. 4).** Each pool agent answers each calibration question alone, closed-book, with its role
    prompt and the draft answer format. e_i(x) = 1 when its answer does not exactly match the gold answer
    after HotpotQA normalisation. ρ̂ is the mean of the defined pairwise φ values, clipped to [0, 1], with
    a 95% bootstrap CI. If it is undefined, the run refuses to start rather than guess a value.
45. **Cost.** Each query has three ledgers: *system* (the cost metric in Table VIII), *instrumentation*
    (claim extraction on baseline traces) and *evaluation* (final decomposition and judge). Tokens of
    cached calls are still counted, because cost describes the computation, not the bill. Latency is the
    query's wall-clock time in the live run.
46. **RC (Eq. 20)** needs U ≥ 2 runs. Runs differ through `run_index`, which enters the cache key of every
    stochastic call. Deterministic calls (temperature 0) share cache entries across runs.
47. **HaluEval.** For each item one of the two answers is shown, chosen by a fixed seed. The CVM detector
    decomposes the shown answer and scores each claim as a singleton against the item's knowledge
    passage. A block decision means "hallucinated". The comparison is a single LLM judging the same
    answer with the knowledge passage.

## E. Parameters

48. **[DECIDE]** The pilot uses the untuned mid-grid values θ_e = θ_c = 0.7, θ_sim = 0.6,
    (α, β, γ) = (0.4, 0.2, 0.4), τ = 0.5, δ = 0.5, Δ = 0.1, r = p = 5, a draft temperature of 0.7 and
    max_tokens of 1024. Section V-D requires tuning on the validation split. The full grid has
    3·3·2·66·6·2·2 = 28,512 configurations, so full reruns are infeasible and a tuning protocol has to be
    chosen, for example coordinate search, or re-scoring cached drafts where a parameter does not change
    any prompt.

## F. Changes after the pilot (pilot_v2)

49. **System NLI changed** to `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`. In the first pilot,
    `cross-encoder/nli-deberta-v3-base` gave P_con ≈ 1.0 for claims against *off-topic* passages: of 60
    sampled grounding vetoes, the large model judged 24 to be neutral. `cross-encoder/nli-deberta-v3-large`
    and `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` showed the same flaw on a probe pair; only the
    ling-wanli checkpoint judged it correctly.
50. **Labelling NLI changed** to `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` so that it still differs from
    the system NLI (VII-F). Known risk: it shares the off-topic false-contradiction flaw. The premises
    are gold supporting facts, which are on topic, so exposure is lower; but NLI-stage "contradicted"
    labels should be audited in the human-annotation stage.

## G. Decisions for the main study (approved 2026-10-04 unless marked otherwise)

51. **Retrieval before drafting (approved by the authors).** CV-SM agents (and A1-A8, B6) receive the same
    top-p BM25 passages for their sub-task as B2/B5 (`switches.draft_retrieval`). R1 still withholds peer
    drafts and memory outside I_j; passages are external evidence, not exposure (Eq. 2). The strictly
    closed-book reading remains available (`draft_retrieval: false`).
52. **Multi-entry synthesis check (approved by the authors).** The Synthesizer cites the memory entries that
    establish its answer. The answer claim is accepted if a single active entry entails it, *or* the
    concatenation of the cited active entries entails it (P_ent ≥ θ_e). Other claims in the synthesis are
    still filtered entry by entry (IV-I).
53. **Coverage floor u_min = 0.8 (chosen by me when the authors deferred the choice).** Rationale: it keeps
    CV-SM comparable with baselines that always answer. Cost cap B in Eq. (8): none unless given.
54. **Tuning protocol (V-D).** One pass of coordinate search over exactly the V-D value sets, in the order
    τ → (α,β,γ) simplex at step 0.1 (66 points) → θ_e → θ_c → θ_sim → δ → Δ, with r = p = 5. Each
    candidate runs CV-SM on the same validation subset (default validation[100:200], disjoint from the
    pilot and ρ̂ calibration items) and is labelled. Selection: lowest HR among candidates with
    coverage ≥ u_min, otherwise highest coverage (logged as infeasible). The full 28,512-point grid is
    infeasible; this protocol must be stated in the paper.
55. **Analysis rules (fixed before any study data were seen).**
    - Table VI's HR, claims per answer, coverage and tokens pool HotpotQA, FEVER and TruthfulQA. A
      system's estimate is the mean over its U runs; 95% CIs come from 10,000 query-level bootstrap
      resamples.
    - H1 is supported iff CV-SM's ΔHR is negative against every baseline with Holm-adjusted bootstrap
      p < 0.05 **and** CV-SM's coverage ≥ u_min. It is not supported if any baseline has significantly
      lower HR or any ΔHR is not negative. Otherwise it is inconclusive.
    - H2: MCR is undefined for memoryless B5, so H2 is judged on PD (B6 > B5, CI excluding 0) together
      with MCR(B6) > MCR(CV-SM).
    - H3 is supported iff CLR(A1) − CLR(CV-SM) has a CI above 0, and not supported iff the CI is below 0.
    - H4 is supported iff the CV-SM MCR slope CI includes 0 (or lies below it) and the B6 slope CI lies
      above 0.
    - H5: regress per-agent input tokens per call on k ∈ {2, 3, 5} with a quadratic fit. It is supported
      iff B5's quadratic coefficient CI lies above 0 and CV-SM's includes 0.
    - McNemar uses per-query correctness by majority over the U runs (HotpotQA EM), with Holm across
      baselines.
56. **HaluEval** runs only CV-SM (detector) and B1 (single-LLM judge), following VI-D. The other systems
    are marked not applicable in Table VI.
57. **FEVER score** needs predicted FEVER evidence sentences, which no system outputs, so Table VI reports
    FEVER label accuracy. FEVER runs are refused until the wiki-pages evidence text is provided (truth
    labels need it).
58. **B3 k′** = round(mean CV-SM tokens per query / mean tokens per B1 sample), from logged runs
    (`scripts/match_sc_samples.py`).
59. **k sweep.** The k = 5 pool needs five agents (`agents_k5` in the frozen config); k = 2 uses the first
    two agents of the main pool. ρ̂ is estimated separately for the main pool, A7 (homogeneous), k = 2 and
    k = 5 (V-D: "separately for each agent configuration").
