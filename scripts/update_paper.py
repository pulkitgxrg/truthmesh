"""Apply the pilot results to paper/code.tex. Every number inserted is read from results_pilot/paper_numbers.json
or the generated table rows; each edit asserts its target text occurs exactly once. Writes CHANGES_paper.md."""
import json
from pathlib import Path

TEX = Path("paper/code.tex")
N = json.loads(Path("results_pilot/paper_numbers.json").read_text())
S, P = N["systems"], N["paired"]
rows6 = Path("results_pilot/table6_rows.tex").read_text().strip()
rowsP = Path("results_pilot/table_process_rows.tex").read_text().strip()
rows8 = Path("results_pilot/table8_rows.tex").read_text().strip()
lab_rev = (Path.home() / ".cache/huggingface/hub/models--MoritzLaurer--DeBERTa-v3-base-mnli-fever-anli/refs/main").read_text().strip()
env = json.loads(Path("runs/20261004-161732_pilot_v2_cvsm/environment.json").read_text())


def th(n):                # thousands separator as a LaTeX thin space (numbers only)
    return f"{n:,.0f}".replace(",", "\\,")


def pc(c, d=1):          # percent with CI, LaTeX
    return f"{c['est']*100:.{d}f}\\% (95\\% CI {c['lo']*100:.{d}f}--{c['hi']*100:.{d}f})"


def num(c, d=2):
    return f"{c['est']:.{d}f} (95\\% CI {c['lo']:.{d}f}--{c['hi']:.{d}f})"


cv, b5, b1 = S["cvsm"], S["b5"], S["b1"]
d5, d1 = P["b5"]["dHR"], P["b1"]["dHR"]
rho = N["rho"]
pw5, pw1 = N["power"]["b5"]["0.05"]["alpha_0.05"], N["power"]["b1"]["0.05"]["alpha_0.05"]
lp = N["label_precision"]
n_ans = round(cv["coverage"]["est"] * N["n_queries"])
slope = N["mcr_slope_cvsm"]
pk = env["packages"]
py = env["python"].split()[0]
changes = []


def rep(old, new, what):
    global tex
    assert tex.count(old) == 1, f"target not unique ({tex.count(old)}): {what}\n{old[:120]}"
    tex = tex.replace(old, new)
    changes.append(what)


tex = TEX.read_text()

# 0 preamble: let long model identifiers in \url{} break at '/', '-' and '.' (layout only)
rep("\\usepackage{url}\n", "\\usepackage{url}\n\\def\\UrlBreaks{\\do\\/\\do-\\do.}\n",
    "Preamble: allow \\url{} to break at '/', '-' and '.' so long model identifiers in Section V-B do not stretch lines (layout only).")

# 1 abstract
rep(r"\datareq{DATA REQUIRED: one or two sentences stating the main quantitative findings once experiments are complete.}",
    f"A pilot on {N['n_queries']} HotpotQA questions with a single base model shows that the protocol and its metrics run "
    f"end to end, but it does not support the main hypothesis. With untuned thresholds, CV-SM answered "
    f"{cv['coverage']['est']*100:.0f}\\% of questions, and its claim-level hallucination rate, {pc(cv['HR'])}, was not lower "
    f"than that of multi-agent debate, {pc(b5['HR'])}. The verifier stopped hallucinated claims from reaching the answer "
    f"(escape rate {cv['escape']['est']*100:.1f}\\% against {b5['escape']['est']*100:.1f}\\% for debate) but also blocked "
    f"{cv['FRR']['est']*100:.1f}\\% of true claims.",
    "Abstract: replaced the DATA REQUIRED marker with the pilot findings.")

# 2 contribution 5
rep(r"\datareq{DATA REQUIRED: a sentence summarising the empirical findings.}",
    f"A pilot evaluation shows that these quantities can be measured end to end, and that verified write, as configured, "
    f"trades a low escape rate ({cv['escape']['est']*100:.1f}\\% of hallucinated claims reached the answer, against "
    f"{b5['escape']['est']*100:.1f}\\% under debate) for heavy over-blocking of true claims "
    f"(false rejection rate {cv['FRR']['est']*100:.1f}\\%).",
    "Contributions, item 5: replaced the DATA REQUIRED marker with a sentence on the pilot findings.")

# 3 Table III
rep(r"Dev (distractor); \datareq{DATA REQUIRED: N}",
    f"Dev (distractor); {N['n_queries']} questions in the pilot, 50 further questions for $\\hat{{\\rho}}$",
    "Table III, HotpotQA: sample size filled from the pilot.")
rep(r"Dev; \datareq{DATA REQUIRED: N}", r"Dev; not run in this paper", "Table III, FEVER: marker replaced by 'not run in this paper'.")
rep(r"All 817 questions, generation setting", r"Generation setting; not run in this paper",
    "Table III, TruthfulQA: 'All 817 questions' replaced by 'not run in this paper' (the current official file has 790 rows).")
rep(r"QA subset; \datareq{DATA REQUIRED: N}", r"QA subset; not run in this paper", "Table III, HaluEval: marker replaced by 'not run in this paper'.")

# 4 sample size sentence
rep(r"Sample sizes are to be fixed by a power analysis on pilot data before the main runs, so that the minimum effect of practical interest on HR can be detected with paired tests at the chosen significance level.",
    r"Sample sizes are to be fixed by a power analysis on pilot data before the main runs, so that the minimum effect of practical interest on HR can be detected with paired tests at the chosen significance level. "
    f"This paper reports only the pilot. At the variance it observed, detecting a difference of five percentage points in HR "
    f"with paired tests at power 0.8 and $\\alpha = 0.05$ would require about {th(pw5)} questions against B5 and {th(pw1)} against B1.",
    "Section V-A: added the pilot power-analysis result after the existing sample-size sentence.")

# 5 Section V-B
rep(r"\datareq{DATA REQUIRED: base LLMs with exact identifiers and access dates; NLI checkpoint; sentence encoder; retrieval corpus snapshot date, index type and retriever; hardware used for NLI and retrieval; software versions.}",
    "The pilot used \\texttt{gpt-4.1-nano} (model identifier \\url{openai/gpt-4.1-nano}, served through the AICredits "
    "gateway, \\url{https://aicredits.in}, accessed 4 October 2026) for all three agents (retriever, reasoner and critic roles), "
    "the decomposer, CVM claim extraction, the Arbiter and the Synthesizer. The LLM judge for truth labels was "
    "\\url{google/gemini-2.5-flash-lite} through the same gateway. The NLI model inside CV-SM was "
    f"\\url{{MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}} (revision \\texttt{{{env['nli_revision'][:7]}}}); "
    f"truth labelling used a different checkpoint, \\url{{MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli}} (revision "
    f"\\texttt{{{lab_rev[:7]}}}). An earlier pilot run used \\url{{cross-encoder/nli-deberta-v3-base}} inside CV-SM; it was "
    "replaced because it assigned high contradiction probability to off-topic passages (Section~\\ref{sec:results}). "
    f"The sentence encoder was \\url{{sentence-transformers/all-MiniLM-L6-v2}} (revision \\texttt{{{env['encoder_revision'][:7]}}}). "
    "Retrieval was Okapi BM25 ($k_1 = 1.5$, $b = 0.75$) over the ten paragraphs that the HotpotQA distractor setting supplies "
    "with each question; no Wikipedia index was built for the pilot. NLI and the encoder ran on an Apple M2 with 16 GB of "
    f"memory through the PyTorch MPS backend. Software: Python {py}, PyTorch {pk['torch']}, Transformers {pk['transformers']}, "
    f"Sentence-Transformers {pk['sentence-transformers']} and the OpenAI Python SDK {pk['openai']}.",
    "Section V-B: replaced the DATA REQUIRED marker with exact model IDs, access date, NLI checkpoints and revisions, encoder, retrieval, hardware and software versions.")

# 5b planned retrieval sentence must not read as a description of the pilot
rep(r"Retrieval runs over a fixed snapshot of English Wikipedia.",
    r"In the full study, retrieval is to run over a fixed snapshot of English Wikipedia.",
    "Section V-B: 'Retrieval runs over a fixed snapshot of English Wikipedia' reworded as the planned setting, since the pilot did not use one.")

# 6 parameters
rep(r"\datareq{DATA REQUIRED: selected values, temperatures and maximum token limits.}",
    "The pilot did not tune these parameters. It used the initial values $\\theta_e = \\theta_c = 0.7$, $\\theta_{\\mathrm{sim}} = 0.6$, "
    "$(\\alpha, \\beta, \\gamma) = (0.4, 0.2, 0.4)$, $\\tau = 0.5$, $\\delta = 0.5$, $\\Delta = 0.1$ and $r = p = 5$, with "
    f"$\\hat{{\\rho}} = {rho['rho_hat']:.2f}$ (95\\% bootstrap CI {rho['bootstrap_ci95'][0]:.2f}--{rho['bootstrap_ci95'][1]:.2f}) "
    "estimated on 50 validation questions disjoint from the pilot. Drafts and revisions were sampled at temperature 0.7 and all "
    "other prompts at temperature 0. Responses were limited to 512 output tokens, except batched claim extraction "
    "($\\max(1024, 700n)$ for $n$ responses) and the judge (128). Agent drafts were closed-book, as R1 states.",
    "Section V-D: replaced the DATA REQUIRED marker with the parameter values, temperatures and token limits used.")

# 7 labelling
rep(r"\datareq{DATA REQUIRED: sample size and agreement statistics.}",
    "In the pilot, the NLI stage decided a claim when one of the entailment and contradiction probabilities was at least 0.9 "
    "and the other was below 0.9. Human annotation was not carried out, so the automatic labels have not been validated "
    "against human judgement.",
    "Section V-F: replaced the DATA REQUIRED marker; states that human annotation was not done.")

# 8 protocol deviation
rep(r"Effect sizes are reported with their intervals in addition to $p$-values.",
    r"Effect sizes are reported with their intervals in addition to $p$-values. The pilot departs from this protocol in three "
    r"respects: it uses validation questions, a single run ($U = 1$), and only B1 and B5 as baselines.",
    "Section V-G: added one sentence stating how the pilot departs from the protocol.")

# 9 results intro
rep(r"This draft reports no experimental results. The section fixes in advance the tables and figures in which results will be presented and the comparison each must support, so that the analysis plan is settled before the data are seen. Every cell marked ``---'' is \datareq{DATA REQUIRED}; no value in this section may be estimated or filled from any source other than the runs described in Section~\ref{sec:method}.",
    f"This section reports the pilot: {N['n_queries']} HotpotQA validation questions, one run per system, CV-SM against B1 and "
    "B5, all agents on one base model. Intervals are 95\\% bootstrap intervals with 10\\,000 resamples and the query as the "
    "resampling unit. Systems and analyses that were not run in the pilot are stated as such; no value is reported for them.",
    "Section VI intro: replaced the 'no results' paragraph with a description of the pilot.")

# 10 Table VI
old6 = tex[tex.index(r"\caption{Main Results on the Test Splits \datareq{DATA REQUIRED}}"):tex.index(r"\parbox{\textwidth}{\footnotesize Values are means over $U = 3$ runs with 95\% bootstrap intervals.}")]
rep(old6 + r"\parbox{\textwidth}{\footnotesize Values are means over $U = 3$ runs with 95\% bootstrap intervals.}",
    r"""\caption{Pilot Results on 50 HotpotQA Validation Questions}
\label{tab:main}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\textwidth}{@{}P{3.4cm}CCCCCC@{}}
\toprule
\textbf{System} & \textbf{HotpotQA EM} & \textbf{HotpotQA F1} & \textbf{HR (\%)} & \textbf{Claims per answer} & \textbf{Coverage (\%)} & \textbf{Tokens per query} \\
\midrule
""" + rows6.replace("\nCV-SM", "\n\\midrule\nCV-SM") + r"""
\bottomrule
\end{tabularx}

\vspace{3pt}
\parbox{\textwidth}{\footnotesize One run per system; 95\% bootstrap intervals in brackets. EM and F1 count an abstention as incorrect; HR and claims per answer are computed over answered questions. B2--B4, B6 and the FEVER, TruthfulQA and HaluEval columns were not run.}""",
    "Table VI: replaced the empty planned table with the pilot table (B1, B5, CV-SM; EM, F1, HR, claims per answer, coverage, tokens); columns and rows for runs not made were removed.")

# 11 H1 text
rep(r"Table~\ref{tab:main} compares CV-SM with the six baselines on the test splits. H1 is supported only if CV-SM has a lower HR than every baseline with a Holm-adjusted confidence interval for the difference that excludes zero, and if this reduction is not explained by lower coverage or by a change in the number of claims per answer. The text accompanying Table~\ref{tab:main} should state the HR difference to B5 and B6 with intervals, report any dataset on which the ordering reverses, and note where task accuracy and HR disagree.",
    r"Table~\ref{tab:main} compares CV-SM with two of the six baselines. H1 is supported only if CV-SM has a lower HR than every baseline with a Holm-adjusted confidence interval for the difference that excludes zero, and if this reduction is not explained by lower coverage or by a change in the number of claims per answer. "
    f"The pilot does not support H1. The HR difference between CV-SM and B5 was {d5['est']*100:+.1f} percentage points "
    f"(95\\% CI {d5['lo']*100:+.1f} to {d5['hi']*100:+.1f}; Holm-adjusted $p = {d5['holm']['p_adj']:.2f}$), and against B1 it was "
    f"{d1['est']*100:+.1f} points ({d1['lo']*100:+.1f} to {d1['hi']*100:+.1f}; $p = {d1['holm']['p_adj']:.2f}$). Neither difference "
    f"excludes zero, and CV-SM's HR rests on {n_ans} answered questions, because it abstained on "
    f"{100 - cv['coverage']['est']*100:.0f}\\% of them. Task accuracy and HR disagree: CV-SM's exact match, "
    f"{cv['hotpot_em']['est']:.2f}, was below B1's ({b1['hotpot_em']['est']:.2f}; McNemar, Holm-adjusted "
    f"$p = {P['b1']['dHR']['mcnemar_holm']['p_adj']:.4f}$) and B5's ({b5['hotpot_em']['est']:.2f}; "
    f"$p = {P['b5']['dHR']['mcnemar_holm']['p_adj']:.1e}$), whereas B5 had both the highest accuracy and the lowest HR.".replace("e-05", "\\times 10^{-5}").replace("2.2\\times", "2.2 \\times"),
    "Section VI-A: kept the H1 criterion sentence; replaced the instructions with the pilot's HR differences, McNemar tests and the H1 verdict.")

# 12 H2/H4 text and figures
rep(r"Fig.~\ref{fig:mcr} plots $\mathrm{MCR}(t)$ against the write index $t$ for CV-SM, B6 and ablation A2, with bootstrap bands. H4 predicts a flat curve for CV-SM and a rising curve for B6; the slope of a linear fit to each curve, with its interval, should be reported. Fig.~\ref{fig:pd} shows the distribution of propagation depth per system on HotpotQA, together with the escape rate. H2 is supported if B6 shows both a higher MCR and a larger mean PD than the memoryless debate baseline B5.",
    r"Fig.~\ref{fig:mcr} plots $\mathrm{MCR}(t)$ against the write index $t$ for CV-SM, with bootstrap bands. H4 predicts a flat curve for CV-SM and a rising curve for B6. "
    f"The slope of a linear fit for CV-SM was {slope['slope']:+.4f} per write (95\\% CI {slope['ci'][0]:+.4f} to {slope['ci'][1]:+.4f}), "
    "so no growth was detected. B6 and ablation A2 were not run, so H4 was not tested. Fig.~\\ref{fig:pd} shows the distribution "
    f"of propagation depth on HotpotQA, together with the escape rate. Mean PD was {num(b5['PD'])} for B5 and {num(cv['PD'])} for CV-SM "
    f"(paired difference {P['b5']['dPD']['est']:+.2f}, 95\\% CI {P['b5']['dPD']['lo']:+.2f} to {P['b5']['dPD']['hi']:+.2f}). "
    "H2 is supported if B6 shows both a higher MCR and a larger mean PD than the memoryless debate baseline B5; because B6 was "
    "not run, H2 was not tested. Table~\\ref{tab:process} gives the process-level metrics.",
    "Section VI-B: replaced the instructions with the measured MCR slope and PD; H2 and H4 stated as not tested (B6, A2 not run); pointer to the new process-metrics table.")
rep(r"\figplaceholder{$\mathrm{MCR}(t)$ against write index for CV-SM, B6 and A2, with 95\% bootstrap bands.}",
    r"\includegraphics[width=\columnwidth]{figs/fig3_mcr.pdf}", "Fig. 3: placeholder replaced by figs/fig3_mcr.pdf (CV-SM only).")
rep(r"\caption{Memory contamination rate over a session.}",
    r"\caption{Memory contamination rate over a session for CV-SM in the pilot, with 95\% bootstrap bands.}",
    "Fig. 3 caption: states that only CV-SM is shown.")
rep(r"\figplaceholder{Box plots of PD per system; escape rate annotated.}",
    r"\includegraphics[width=\columnwidth]{figs/fig4_pd.pdf}", "Fig. 4: placeholder replaced by figs/fig4_pd.pdf (B5 and CV-SM).")

# 13 H3 text; ablation table replaced by process table
rep(r"Table~\ref{tab:ablation} reports the ablations. The comparison of CV-SM with A1 tests H3 through CLR. Fig.~\ref{fig:precision} shows the precision of each agreement label as a correctness signal, defined as the fraction of clusters with a given label whose truth label is 1, for CV-SM, A1 and B5. If R1 matters, the precision of the Corroborated label should fall when it is removed. The estimated $\hat{\rho}$ for the homogeneous (A7) and heterogeneous configurations should be reported beside the table, since the correlation discount can only help where $\hat{\rho}$ is substantially above zero.",
    r"The comparison of CV-SM with A1 tests H3 through CLR. A1 was not run in the pilot, so H3 was not tested. "
    f"Descriptively, CLR was {pc(cv['CLR'])} for CV-SM and {pc(b5['CLR'])} for B5; the CV-SM estimate rests on few falsely "
    "accepted clusters. Fig.~\\ref{fig:precision} shows the precision of each agreement label as a correctness signal, defined as "
    "the fraction of clusters with a given label whose truth label is 1, for CV-SM and B5. For CV-SM the precision was "
    f"{lp['cvsm:Corroborated']['precision']*100:.1f}\\% for Corroborated ($n = {lp['cvsm:Corroborated']['n']}$), "
    f"{lp['cvsm:Resolved']['precision']*100:.1f}\\% for Resolved ($n = {lp['cvsm:Resolved']['n']}$) and "
    f"{lp['cvsm:Singleton-grounded']['precision']*100:.1f}\\% for Singleton-grounded ($n = {lp['cvsm:Singleton-grounded']['n']}$); "
    f"for clusters accepted by B5's majority it was {lp['b5:Majority-accepted']['precision']*100:.1f}\\% "
    f"($n = {lp['b5:Majority-accepted']['n']}$). With three role prompts on one base model, the estimated error correlation was "
    f"$\\hat{{\\rho}} = {rho['rho_hat']:.2f}$ (95\\% CI {rho['bootstrap_ci95'][0]:.2f}--{rho['bootstrap_ci95'][1]:.2f}). The "
    "heterogeneous configuration and A7 were not run.",
    "Section VI-C: replaced the instructions with the measured CLR, label precision and rho-hat; H3 stated as not tested (A1 not run).")
old7 = tex[tex.index("\\begin{table}[t]\n\\caption{Ablation Results on HotpotQA and TruthfulQA"):tex.index(r"A4 is expected to raise VSR")]
rep(old7, r"""\begin{table*}[t]
\caption{Process-Level Metrics in the Pilot}
\label{tab:process}
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\textwidth}{@{}P{3.4cm}CCCCC@{}}
\toprule
\textbf{System} & \textbf{VSR (\%)} & \textbf{FRR (\%)} & \textbf{Mean PD} & \textbf{Escape (\%)} & \textbf{CLR (\%)} \\
\midrule
""" + rowsP + r"""
\bottomrule
\end{tabularx}

\vspace{3pt}
\parbox{\textwidth}{\footnotesize 95\% bootstrap intervals in brackets. The ablations A1--A8 were not run.}
\end{table*}

""", "Table VII: replaced the empty ablation table (not run) with a process-level metrics table for B1, B5 and CV-SM (VSR, FRR, mean PD, escape rate, CLR).")
rep(r"A4 is expected to raise VSR at the cost of a higher FRR, since rejecting every singleton also discards correct details found by only one agent; the two columns must be read together.",
    r"VSR and FRR must be read together. CV-SM blocked " + f"{pc(cv['VSR'])} of hallucinated internal claims but also "
    f"{pc(cv['FRR'])} of supported ones; for B5 the figures were {b5['VSR']['est']*100:.1f}\\% and {b5['FRR']['est']*100:.1f}\\%. "
    "In the first pilot run, with \\texttt{cross-encoder/nli-deberta-v3-base} as the system NLI, "
    f"{N['nli_audit']['n_vetoes_G_le_-0.9']} clusters were vetoed by grounding with $G \\le -0.9$; in a seeded sample of "
    f"{N['nli_audit']['sample_size']}, the ling-wanli checkpoint judged {N['nli_audit']['large_contradiction']} of the passage--claim "
    f"pairs contradictory, {N['nli_audit']['large_neutral']} neutral and {N['nli_audit']['large_entailment']} entailed. Replacing "
    f"the system NLI raised coverage from {N['v1']['coverage']*100:.0f}\\% to {cv['coverage']['est']*100:.0f}\\% and lowered FRR "
    f"from {N['v1']['FRR']*100:.1f}\\% to {cv['FRR']['est']*100:.1f}\\%.",
    "Section VI-C: replaced the A4 expectation sentence (A4 not run) with the measured VSR/FRR trade-off and the NLI audit.")
rep(r"\figplaceholder{Fraction correct per label (Corroborated, Singleton-grounded, Resolved) for CV-SM, A1 and B5, with intervals.}",
    r"\includegraphics[width=\columnwidth]{figs/fig5_label_precision.pdf}", "Fig. 5: placeholder replaced by figs/fig5_label_precision.pdf (CV-SM and B5; A1 not run).")

# 14 calibration
rep(r"Fig.~\ref{fig:reliability} is a reliability diagram of $\mathrm{Conf}^{*}$ against empirical claim accuracy for CV-SM and A6, with ECE and Brier scores in the legend. On HaluEval, the CVM is evaluated as a detector by treating a block decision as a prediction of hallucination; precision, recall and F1 are reported with intervals and compared with the recognition accuracy of a single LLM prompted to judge the same responses.",
    r"Fig.~\ref{fig:reliability} is a reliability diagram of $\mathrm{Conf}^{*}$ against empirical claim accuracy for CV-SM, with ECE and Brier scores in the legend. "
    f"ECE was {num(cv['ECE'], 3)} and the Brier score {num(cv['Brier'], 3)}. Ablation A6 and the HaluEval detector evaluation were not run.",
    "Section VI-D: replaced the instructions with measured ECE and Brier; A6 and HaluEval stated as not run.")
rep(r"\figplaceholder{$\mathrm{Conf}^{*}$ bins against empirical accuracy for CV-SM and A6; ECE and Brier in legend.}",
    r"\includegraphics[width=\columnwidth]{figs/fig6_reliability.pdf}", "Fig. 6: placeholder replaced by figs/fig6_reliability.pdf (CV-SM only).")

# 15 cost
rep(r"Table~\ref{tab:cost} reports cost per query. H5 is tested by regressing per-agent input tokens on $k$ for B5 and CV-SM and comparing the fitted growth with the forms in \eqref{eq:tokens}. Fig.~\ref{fig:pareto} plots HR against tokens per query for every configuration, with $\tau$ and $k$ varied for CV-SM, and marks the Pareto frontier.",
    r"Table~\ref{tab:cost} reports cost per query at $k = 3$. H5 is tested by regressing per-agent input tokens on $k$ for B5 and CV-SM and comparing the fitted growth with the forms in \eqref{eq:tokens}; the runs at $k = 2$ and $k = 5$ were not made, so H5 was not tested. "
    f"At $k = 3$, CV-SM used fewer input tokens per query than B5 ({th(cv['cost']['input_tokens'])} against {th(b5['cost']['input_tokens'])}) "
    + f"but more LLM calls ({cv['cost']['llm_calls']:.1f} against {b5['cost']['llm_calls']:.1f}) and about {cv['cost']['nli_calls']:.0f} NLI calls, "
    f"and its median latency was {cv['cost']['latency_median']:.0f} s against {b5['cost']['latency_median']:.0f} s. "
    "Fig.~\\ref{fig:pareto} plots HR against tokens per query for the three systems and marks the Pareto frontier; the $\\tau$ sweep was not run.",
    "Section VI-E: replaced the instructions with the k = 3 cost comparison; H5 and the tau sweep stated as not run.")
old8 = tex[tex.index(r"\caption{Cost per Query on HotpotQA \datareq{DATA REQUIRED}}"):tex.index(r"\end{table*}", tex.index(r"\caption{Cost per Query on HotpotQA"))]
rep(old8, r"""\caption{Cost per Query on HotpotQA in the Pilot}
\label{tab:cost}
\centering
\footnotesize
\renewcommand{\arraystretch}{1.12}
\begin{tabularx}{\textwidth}{@{}P{3.6cm}CCCCCCCC@{}}
\toprule
\textbf{System} & \textbf{$k$} & \textbf{Input tokens} & \textbf{Output tokens} & \textbf{LLM calls} & \textbf{NLI calls} & \textbf{Median latency (s)} & \textbf{p95 latency (s)} & \textbf{USD (list price)} \\
\midrule
""" + rows8 + r"""
\bottomrule
\end{tabularx}

\vspace{3pt}
\parbox{\textwidth}{\footnotesize Means over the 50 pilot questions. USD at the gateway's list price for \texttt{gpt-4.1-nano} (\$0.10 and \$0.40 per million input and output tokens), excluding evaluation. The runs at $k = 2$ and $k = 5$ were not made.}
""", "Table VIII: replaced the empty k = 2/3/5 table with measured k = 3 cost for B1, B5 and CV-SM, plus a USD column.")
rep(r"\figplaceholder{HR against tokens per query for all systems; CV-SM swept over $\tau$ and $k$; Pareto frontier marked.}",
    r"\includegraphics[width=\columnwidth]{figs/fig7_pareto.pdf}", "Fig. 7: placeholder replaced by figs/fig7_pareto.pdf (three systems; no tau or k sweep).")

# 16 error analysis
rep(r"\datareq{DATA REQUIRED: sample size, counts and examples.}",
    f"In the pilot, {len(Path('results_pilot/error_analysis_sample.csv').read_text().strip().splitlines()) - 1} hallucinated claims "
    "reached CV-SM's final answers. They were exported for categorisation, but no categories were assigned, because the "
    "assignment requires human judgement that has not yet been carried out.",
    "Section VI-F: replaced the DATA REQUIRED marker with the pilot count and a statement that the categories were not assigned.")

# 17 conclusion
rep(r"The empirical questions remain open until the evaluation of Section~\ref{sec:method} has been run. \datareq{DATA REQUIRED: two or three sentences stating which hypotheses were supported, with effect sizes, once results are available.}",
    f"A pilot on {N['n_queries']} HotpotQA questions did not support H1: CV-SM's hallucination rate, {pc(cv['HR'])}, was not "
    f"lower than that of multi-agent debate, {pc(b5['HR'])}, and CV-SM answered only {cv['coverage']['est']*100:.0f}\\% of "
    "questions. Verified write kept hallucinated claims out of the answer "
    f"(escape rate {cv['escape']['est']*100:.1f}\\% against {b5['escape']['est']*100:.1f}\\%) but blocked "
    f"{cv['FRR']['est']*100:.1f}\\% of true claims, and agents built on one base model were highly correlated "
    f"($\\hat{{\\rho}} = {rho['rho_hat']:.2f}$). H2--H5 were not tested. The results identify what must change before a full "
    "evaluation: tuning the thresholds against the coverage constraint, heterogeneous agents, and an NLI model that does not "
    "read off-topic evidence as contradiction.",
    "Conclusion: replaced the DATA REQUIRED marker with the pilot findings and the untested hypotheses.")

# 18 limitations: pilot scale
rep(r"An adversarial agent that deliberately coordinated with others, or poisoned the retrieval corpus, is not addressed, and the security literature on memory poisoning \cite{b42} suggests that this setting requires separate defences.",
    r"An adversarial agent that deliberately coordinated with others, or poisoned the retrieval corpus, is not addressed, and the security literature on memory poisoning \cite{b42} suggests that this setting requires separate defences. "
    r"The reported evidence is a pilot: 50 questions from one dataset, one run per system, one base model, untuned parameters and unvalidated automatic labels. It is underpowered for the effects of interest and does not test H2--H5.",
    "Section VII-E (Limitations): added two sentences on the scale of the pilot.")

# 19 remove reference [60] (authors' decision) and the subsection that rests on it
rep(r"because a correct minority can be outvoted (Section~\ref{sec:prototype}).",
    r"because a correct minority can be outvoted.",
    "Section IV-G: removed the cross-reference to Section VII-C (that subsection was removed).")
proto = tex[tex.index(r"\subsection{Observations From a Prototype Verifier}"):tex.index(r"\subsection{Practical Implications}")]
rep(proto, "", "Section VII-C 'Observations From a Prototype Verifier' removed: it rested entirely on reference [60], which the authors removed.")
b60 = tex[tex.index(r"\bibitem{b60}"):tex.index(r"\end{thebibliography}")]
rep(b60, "", "References: removed [60] (Groundtruth GitHub repository) at the authors' request.")

assert r"\datareq{" not in tex.split(r"\begin{document}")[1], "a DATA REQUIRED marker remains"
assert r"\figplaceholder{" not in tex.split(r"\begin{document}")[1], "a figure placeholder remains"
TEX.write_text(tex)
Path("CHANGES_paper.md").write_text("# Changes to paper/code.tex\n\nAll numbers come from results_pilot/paper_numbers.json "
                                    "(generated from logged runs by scripts/pilot_results.py). Original: paper/code_original.tex.\n\n"
                                    + "\n".join(f"{i}. {c}" for i, c in enumerate(changes, 1)) + "\n\n"
                                    + Path("paper/CHANGES_notes.md").read_text())
print(len(changes), "changes applied")
