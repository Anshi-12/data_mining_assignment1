# Data Mining — Assignment 1

Chatbot-assisted data science and coding. This assignment has two parts, each on its own branch.

## Contents

| Part | Project | Branch |
| --- | --- | --- |
| **Part 1** | CRISP-DM Studio — a dataset-agnostic, leakage-safe CRISP-DM web app that turns any CSV into a full analysis with downloadable HTML/PDF reports | [`part1-crispdm`](https://github.com/Anshi-12/data_mining_assignment1/tree/part1-crispdm) |
| **Part 2** | Anomaly Detection Studio — a full-stack (FastAPI + React) replication and improvement of `06_anomaly_detection`, scoring synthetic server telemetry for anomalies | [`part2-anomaly_detection`](https://github.com/Anshi-12/data_mining_assignment1/tree/part2-anomaly_detection) |

> 👉 The actual project code lives on the two branches linked above. Switch to a branch (or click its link) to see that part's code and README.

---

## 📺 Video Walkthroughs

- **Part 1 — CRISP-DM Studio:** <https://youtu.be/rxYuhJxUmkM?feature=shared>
- **Part 2 — Anomaly Detection Studio:** <https://youtu.be/UH8KraVDyRY?is=vqFhyMfPs57YKYSB>

<!-- Optional: Part 1 Medium article — PASTE MEDIUM LINK HERE -->

---

## Part 1 — CRISP-DM Studio (branch `part1-crispdm`)

A single-command Streamlit app that runs the full CRISP-DM workflow on an uploaded CSV.

- **Leakage-safe by design:** learned preprocessing is fit only on training folds, never the full dataset.
- **Analyze once, render many ways:** the same results and charts drive both the UI and the reports.
- Honest behavior: explicit target confirmation, meaningfulness-gated clustering, baseline-anchored evaluation, clear skip states.
- Downloadable HTML + PDF CRISP-DM reports.

## Part 2 — Anomaly Detection Studio (branch `part2-anomaly_detection`)

A FastAPI + React dashboard scoring high-dimensional server telemetry for anomalies. Replicates and improves on [`dlmastery/data_science_examples/06_anomaly_detection`](https://github.com/dlmastery/data_science_examples/tree/main/06_anomaly_detection):

- **Real live scoring** using the actual fitted RobustScaler + Isolation Forest pipeline (not the original's heuristic).
- **Real experiments** — measured hyperparameter search and genuine retraining with atomic artifact swaps.
- Honest data description (locally synthesized, not Kaggle) and corrected project structure.

---

Built for CMPE 255 / Data Mining, San José State University.
