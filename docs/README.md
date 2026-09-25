# Project Documentation — Business Entity Resolution (ML Challenge 2026)

This folder is the **living record** of the project: what we did at every step,
the metrics and results we got, why we made each choice, the problems we hit and
how we solved them, and what remains to improve. It is updated continuously as
the project proceeds.

## How to read this

Read in order for the full story, or jump to a topic:

| # | Document | What it covers |
|---|----------|----------------|
| 00 | [Challenge Overview](00_challenge_overview.md) | The problem, the F₀.₅ metric, constraints, submission format |
| 01 | [Environment & Repo Setup](01_environment_and_repo_setup.md) | Python/venv choice, git repo situation, tooling decisions & why |
| 02 | [Dataset Exploration (full EDA)](02_dataset_exploration.md) | Scale, fields, country mix, missingness, scripts, ground-truth distribution — every measured number |
| 03 | [Noise Taxonomy](03_noise_taxonomy.md) | Every noise pattern with real examples, and what each implies for the pipeline |
| 04 | [Signal Analysis & Recall Ceiling](04_signal_analysis.md) | Within-cluster similarity, how strong each field is, the blocking recall ceiling |
| 05 | [Cleaning & Normalization Pipeline](05_cleaning_pipeline.md) | The cleaning design: what, why, problems faced, fixes, validation |
| 06 | [Blocking Recall Baseline](06_blocking_recall_baseline.md) | Measured blocking recall, block-size explosion, miss analysis |
| 07 | [Decisions Log](07_decisions_log.md) | Chronological record of every significant decision and its rationale |
| 08 | [Next Steps & Roadmap](08_next_steps.md) | The plan to reach a strong F₀.₅, with rationale and alternatives |
| 09 | [Blocking v2](09_blocking_v2.md) | Candidate generation: experiments, the recall/volume trade-off, IDF top-N via sparse matmul (94.5% recall, 0.66B pairs) |

## Project status

| Phase | Status |
|-------|--------|
| 1. Deep dataset understanding (EDA) | ✅ Complete |
| 2. Data cleaning & normalization | ✅ Complete & validated |
| 3. Candidate generation (blocking) | ✅ v2 complete (94.5% recall, 0.66B candidates) |
| 4. Matching model (classifier) | 🔜 Next |
| 5. Threshold tuning for F₀.₅ + submission | ⏳ Pending |

## One-paragraph summary

We must decide, across three noisy independent data sources, which Source-2 and
Source-3 business records refer to the same real-world business as each
deduplicated Source-1 reference entity. The data is large (24.2M records total)
and noisy (typos, transliterations into 8 Indic scripts, reordered addresses,
legal-suffix drift, domain-form names, an unseen third country — France — in the
test set). Scoring is **F₀.₅** (precision weighted 2× recall), macro-averaged
per Source-1 entity. Our approach is a classic, defensible **blocking →
precision-first classifier** pipeline, built entirely offline with permissively
licensed tools, and every stage is measured against held-out ground truth.

## Reproducibility pointers

- Pipeline code: [`../code/business_entity_resolution/`](../code/business_entity_resolution/)
- Pinned environment: [`../code/business_entity_resolution/requirements.txt`](../code/business_entity_resolution/requirements.txt)
- Cleaned data (gitignored, regenerate with `clean_dataset.py --all`): `../data/processed/`
