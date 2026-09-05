# Phase 9 hardening matrix

| Hostile input / failure | Expected behavior | Actual result |
|---|---|---|
| Empty file | friendly ingestion rejection | PASS — We couldn't analyze this file because it is empty. |
| Zero-row/header-only | friendly zero-row rejection | PASS — We couldn't analyze this file because it contains column headers but no data rows. |
| Non-CSV upload | friendly file-type rejection | PASS — Please upload a CSV file. '.txt' files are not supported. |
| Single-row dataset | descriptive stages survive; clustering skips | PASS — EDA returned; clustering skipped=True |
| Single-column dataset | downstream unsupported stages skip | PASS — clustering skipped=True; modeling skipped=True |
| All-numeric | adaptive EDA only runs applicable analyses | PASS — numeric=3, categorical=0, skipped=4 |
| All-categorical | adaptive EDA only runs applicable analyses | PASS — numeric=0, categorical=3, skipped=5 |
| Near-total missingness | quality warnings + safe descriptive result | PASS — quality warnings=8; EDA returned |
| Wide/high-cardinality | bounded charts/relationships; no crash | PASS — 80 source columns; 25 charts |
| Duplicate/garbage headers + CP1252 | normalize headers and decode | PASS — columns=['name', 'name__2', 'column_3', 'city']; warnings=2 |
| Highly imbalanced/tiny target | skip unsafe stratified modeling | PASS — At least one target class has fewer than two rows, so a stratified holdout cannot be created safely. |
| No clusterable features | structured clustering skip | PASS — No usable continuous or safely encodable categorical features remain after structural cleaning. |
| Forced clustering failure | convert exception to structured skip | PASS — Clustering candidate search failed safely (RuntimeError); no segments are being presented. |
| Forced model failure | exclude failed estimators / clean skip if all fail | PASS — All candidate estimators failed safely; no fitted model result is available. |
| Forced export failure | HTML survives; PDF uses placeholders/text fallback | PASS — html=True; fidelity=chart-fallback; placeholders=7 |
| New/failed upload after old report | clear dataset and all downstream state | PASS — dataset=None; report=None |
