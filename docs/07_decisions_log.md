# 07 — Decisions Log

A chronological record of every significant decision, the options considered, the
choice, and the reason. New entries are appended as the project proceeds.

---

### D1 — Isolate the git repository
**Context:** the working dir sat inside a git repo rooted at the user's *home
directory*, whose remote was an unrelated project. `git add -A` would have staged
`~/.ssh`, credentials, and every home project.
**Options:** (a) reuse the home repo carefully; (b) create a fresh isolated repo
in `student_resource/`.
**Decision:** (b). **Why:** eliminates any risk of leaking home secrets, keeps
the repo scoped to this project, matches the submission layout.

### D2 — Exclude the raw dataset from git
**Context:** raw `.tsv` is 2.3 GB, each file >100 MB (over GitHub's limit).
**Decision:** `.gitignore` the dataset, venv, and processed Parquet.
**Why:** the dataset is provided input, not our work; it would be rejected by
GitHub; processed data is regenerable from code. Keeps the repo small and clean.

### D3 — Python 3.11 venv instead of system 3.14
**Context:** default `python3` is 3.14; many ML wheels (pyarrow, xgboost, scipy)
have no 3.14 build yet.
**Options:** system 3.14 / conda 3.13 / Homebrew 3.11.
**Decision:** Homebrew **3.11** venv. **Why:** mature, universal wheel support;
fast, stable installs for the whole scientific stack.

### D4 — Transliteration via `indic-transliteration` (library, not hand-rolled)
**Context:** ~14% of S2/S3 names are in 8 Indic scripts; must romanize offline.
**Options:** (a) the MIT `indic-transliteration` library; (b) a self-contained
Unicode→Latin rule map.
**Decision:** (a). **Why:** far higher romanisation accuracy across all 8 scripts
and their conjuncts, still fully offline and permissively licensed, one small
dependency. (User chose this explicitly.)
**Fair-play note:** deterministic script conversion is normalisation, not an
external identity lookup — compliant.

### D5 — Submission-shaped layout from day one
**Decision:** `code/business_entity_resolution/{src,README.md,requirements.txt}`
+ `data/processed/` + `output/`. **Why:** the repo *is* the submission package
(minus raw data); nothing to reorganise later. (User chose this.)

### D6 — Parquet (zstd) for cleaned data, not TSV
**Decision:** write cleaned data as columnar Parquet.
**Why:** 2–4× smaller, typed, and column-selective reads make every downstream
pass fast on 24.2M rows.

### D7 — Keep both raw and canonical fields
**Decision:** store `raw_name/raw_addr` alongside canonical fields.
**Why:** canonicalisation helps recall but hides exact agreement; keeping raw
lets the matching model reward exact matches (a strong precision signal).

### D8 — Strip only true legal suffixes, keep descriptive words (revised)
**Context:** first version stripped Ventures/Enterprises/Services/Partners/…
**Problem:** "Red Ventures"→"red" collapses distinct businesses → precision loss
+ block explosion.
**Decision:** strip only genuine legal-entity types + stopwords; keep descriptive
words in the core; handle over-common tokens via **DF pruning at blocking time**.
**Why:** cleanly separates *identity signal* (keep) from *blocking selectivity*
(prune later); protects precision.

### D9 — Order-invariant address token bag + all-field state scan
**Context:** addresses are heavily reordered and contain literal `null`.
**Decision:** represent address as a sorted token set (drop `null`, normalise
abbreviations/ordinals/zeros), and scan **all** comma fields for the state.
**Why:** matches the observed reordering noise; position-based parsing would fail.

### D10 — Multiprocessing streaming cleaner
**Decision:** process pool over 50k-line chunks, one Parquet row-group per chunk.
**Why:** string normalisation is CPU-bound/GIL-heavy → processes beat threads;
row-group streaming keeps memory flat. Result: 24.2M rows in ~2.5 min.

### D11 — Simple 3-key blocking as the measured baseline
**Decision:** ship v1 blocking (name-token, first-2, street#+state) and measure
it before adding complexity.
**Why:** a fully understood, diagnostic baseline (94.9%, US 97.9% / India 90.5%)
localises the real problems (India + domains + generic-token explosion) and gives
each future change a falsifiable target. Complexity is added only where the data
shows it is needed.

### D12 — Comprehensive living documentation
**Decision:** maintain this `docs/` set, updated at every step.
**Why:** the challenge requires a methodology write-up and reproducibility; a
running record also prevents re-deriving decisions and makes the final
Documentation_template.md a compilation task, not a scramble.

---

*(append future decisions below)*
