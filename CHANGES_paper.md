# Changes to paper/code.tex

All numbers come from results_pilot/paper_numbers.json (generated from logged runs by scripts/pilot_results.py). Original: paper/code_original.tex.

1. Preamble: allow \url{} to break at '/', '-' and '.' so long model identifiers in Section V-B do not stretch lines (layout only).
2. Abstract: replaced the DATA REQUIRED marker with the pilot findings.
3. Contributions, item 5: replaced the DATA REQUIRED marker with a sentence on the pilot findings.
4. Table III, HotpotQA: sample size filled from the pilot.
5. Table III, FEVER: marker replaced by 'not run in this paper'.
6. Table III, TruthfulQA: 'All 817 questions' replaced by 'not run in this paper' (the current official file has 790 rows).
7. Table III, HaluEval: marker replaced by 'not run in this paper'.
8. Section V-A: added the pilot power-analysis result after the existing sample-size sentence.
9. Section V-B: replaced the DATA REQUIRED marker with exact model IDs, access date, NLI checkpoints and revisions, encoder, retrieval, hardware and software versions.
10. Section V-B: 'Retrieval runs over a fixed snapshot of English Wikipedia' reworded as the planned setting, since the pilot did not use one.
11. Section V-D: replaced the DATA REQUIRED marker with the parameter values, temperatures and token limits used.
12. Section V-F: replaced the DATA REQUIRED marker; states that human annotation was not done.
13. Section V-G: added one sentence stating how the pilot departs from the protocol.
14. Section VI intro: replaced the 'no results' paragraph with a description of the pilot.
15. Table VI: replaced the empty planned table with the pilot table (B1, B5, CV-SM; EM, F1, HR, claims per answer, coverage, tokens); columns and rows for runs not made were removed.
16. Section VI-A: kept the H1 criterion sentence; replaced the instructions with the pilot's HR differences, McNemar tests and the H1 verdict.
17. Section VI-B: replaced the instructions with the measured MCR slope and PD; H2 and H4 stated as not tested (B6, A2 not run); pointer to the new process-metrics table.
18. Fig. 3: placeholder replaced by figs/fig3_mcr.pdf (CV-SM only).
19. Fig. 3 caption: states that only CV-SM is shown.
20. Fig. 4: placeholder replaced by figs/fig4_pd.pdf (B5 and CV-SM).
21. Section VI-C: replaced the instructions with the measured CLR, label precision and rho-hat; H3 stated as not tested (A1 not run).
22. Table VII: replaced the empty ablation table (not run) with a process-level metrics table for B1, B5 and CV-SM (VSR, FRR, mean PD, escape rate, CLR).
23. Section VI-C: replaced the A4 expectation sentence (A4 not run) with the measured VSR/FRR trade-off and the NLI audit.
24. Fig. 5: placeholder replaced by figs/fig5_label_precision.pdf (CV-SM and B5; A1 not run).
25. Section VI-D: replaced the instructions with measured ECE and Brier; A6 and HaluEval stated as not run.
26. Fig. 6: placeholder replaced by figs/fig6_reliability.pdf (CV-SM only).
27. Section VI-E: replaced the instructions with the k = 3 cost comparison; H5 and the tau sweep stated as not run.
28. Table VIII: replaced the empty k = 2/3/5 table with measured k = 3 cost for B1, B5 and CV-SM, plus a USD column.
29. Fig. 7: placeholder replaced by figs/fig7_pareto.pdf (three systems; no tau or k sweep).
30. Section VI-F: replaced the DATA REQUIRED marker with the pilot count and a statement that the categories were not assigned.
31. Conclusion: replaced the DATA REQUIRED marker with the pilot findings and the untested hypotheses.
32. Section VII-E (Limitations): added two sentences on the scale of the pilot.
33. Section IV-G: removed the cross-reference to Section VII-C (that subsection was removed).
34. Section VII-C 'Observations From a Prototype Verifier' removed: it rested entirely on reference [60], which the authors removed.
35. References: removed [60] (Groundtruth GitHub repository) at the authors' request.

## Notes

- Kept as is: the "---" marks inside the TikZ drawing of Fig. 2 (they are diagram content meaning "no claim", not
  table cells), every section heading, the method (Sections III–IV), Related Work, and all other wording.
- Removed from tables (not run, so no values exist): Table VI rows B2–B4, B6 and the FEVER, TruthfulQA and
  HaluEval columns; Table VII ablation rows A1–A8 (the table now gives process metrics for the systems that ran);
  the k = 2 and k = 5 entries of Table VIII.
- Not filled, because the data do not exist: Cohen's κ and labeller precision/recall (no human labels yet), and the
  error-analysis category counts (need human judgement). The text now says this instead of showing a red marker.
- Compilation: Tectonic (XeTeX engine) with the IEEEtran class. Remaining warnings are the same as for the original
  file (two overfull boxes of 2.5 pt and 2.9 pt at the ends of Tables II and V, and loose lines in three
  bibliography entries); the long model identifiers in Section V-B were made breakable to fix the new ones.
