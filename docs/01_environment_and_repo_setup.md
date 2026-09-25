# 01 — Environment & Repository Setup

This documents the groundwork before any modelling: the git situation we
inherited, how we made the repo safe, and the Python/tooling choices (with the
reasons, because a couple of them were non-obvious and bit us).

## Git repository — a hazard we found and fixed

**What we found:** the working directory `Downloads/student_resource` was *inside
a git repository rooted at the user's home directory* (`/Users/vanshmundhra`),
whose `origin` pointed at an unrelated project (`vansh007/elpeda.git`).

**Why that is dangerous:** a naive `git add -A` would have staged the **entire
home folder** — `.ssh/` private keys, shell history, cloud credentials, and
dozens of unrelated projects — and could have pushed secrets to a remote.

**How we solved it:**
1. Left the accidental home-directory repo **untouched**.
2. Ran `git init` in `student_resource/` to create a **separate, isolated repo**.
3. Added a `.gitignore` that excludes the raw dataset (2.3 GB of `.tsv`, each
   file >100 MB — far over GitHub's limit), the virtualenv, and processed
   Parquet outputs.
4. Committed only project code/docs and pushed to the requested remote,
   `https://github.com/1417-Ke-Overfits/v_claude.git` (branch `main`).

**Why this is the right call:** the dataset is provided input, not our work, and
would be rejected by GitHub anyway; keeping it out keeps the repo small,
reproducible (anyone regenerates processed data from the raw files), and free of
accidental secret leakage.

> Open item: the home-directory repo (`~/.git`) still exists and still tracks the
> whole home folder. Cleaning it up (`rm -rf ~/.git`) is advisable but was left
> to the user's discretion.

## Python interpreter — why 3.11, not the system 3.14

**Problem:** the default `python3` on this machine is **3.14.3** — bleeding edge.
Many compiled ML wheels (`pyarrow`, `xgboost`, `lightgbm`, `scipy`) do not yet
publish builds for 3.14, so installs fail or fall back to slow source builds.

**Options considered:**
- conda `base` (Python 3.13) — also newer than ideal for some wheels.
- Homebrew `python3.11` (3.11.14) — mature, universal wheel support.

**Decision:** create a project virtualenv with **Homebrew Python 3.11** at
`.venv/` (gitignored). 3.11 is the current sweet spot for the scientific-Python
stack: every dependency we need ships a prebuilt wheel, installs are fast, and
behaviour is stable.

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r code/business_entity_resolution/requirements.txt
```

## Dependencies (all offline, permissively licensed)

| Package | License | Role |
|---------|---------|------|
| numpy, pandas, scipy | BSD | arrays / dataframes / sparse math |
| pyarrow | Apache-2.0 | Parquet I/O (columnar, compressed, fast) |
| RapidFuzz | MIT | fast string similarity (Levenshtein, Jaro-Winkler, token/char ratios) |
| indic-transliteration | MIT | offline native-script → Latin romanization |
| scikit-learn | BSD | TF-IDF vectoriser, calibration |
| xgboost | Apache-2.0 | gradient-boosted matching classifier (≤8B params trivially satisfied) |

**Fair-play note:** none of these performs an *external data lookup*. They are
algorithms operating only on the provided data. Transliteration in particular is
a deterministic script conversion, not an identity lookup, so it complies with
the "no external data" rule — the same category as the string-similarity
features the challenge brief explicitly encourages.

## Repository / directory layout

We adopted the required submission structure from the start so nothing needs
reorganising later:

```
student_resource/
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── normalize.py       # text cleaning & canonicalization (single source of truth)
│       │   └── clean_dataset.py   # streaming .tsv -> Parquet driver
│       ├── README.md
│       └── requirements.txt
├── data/
│   ├── processed/                 # cleaned Parquet (gitignored, regenerable)
│   └── interim/                   # scratch (gitignored)
├── docs/                          # this documentation set
├── dataset/                       # raw provided .tsv (gitignored)
├── output/                        # final matching_results.tsv + candidate_pairs.tsv (later)
└── utils/validate_submission.py   # provided format validator
```

**Why:** `output/` + `code/business_entity_resolution/` + the methodology doc is
exactly the final submission zip layout the brief mandates, so our repo *is* the
submission package minus the raw data.

## Storage format decision: Parquet, not CSV/TSV

Cleaned data is written as **zstd-compressed Parquet**, not TSV.

**Why:** columnar Parquet is ~2–4× smaller, preserves dtypes (no re-parsing
strings vs numbers), and lets later stages read *only the columns they need*
(e.g. blocking reads `name_core, street_number, state` without touching
`raw_name`/`raw_addr`). On 24.2M rows this is the difference between a few
seconds and a minute per pass. Each cleaned file is ~165–520 MB.
