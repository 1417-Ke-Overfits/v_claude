# Business Entity Resolution — Pipeline

Matches noisy Source-2 / Source-3 business records to each deduplicated
Source-1 reference entity (ML Challenge 2026). Fully offline; permissively
licensed dependencies only; no external data lookup.

## Environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r code/business_entity_resolution/requirements.txt
```

## Pipeline stages

Run everything from the `student_resource/` project root.

### 1. Clean & normalize  (`src/clean_dataset.py`, `src/normalize.py`)

Streams every raw `*.tsv` into a canonical Parquet table, applying:
accent stripping, **native-script → Latin transliteration** (Devanagari, Tamil,
Telugu, Kannada, Bengali, Gujarati, Malayalam, Gurmukhi), business-name
canonicalization (legal-suffix separation, domain/phone/`&`↔`and` handling),
and order-invariant address parsing with multi-country **state/region
normalization** (US states, Indian states/UTs, French regions).

```bash
python code/business_entity_resolution/src/clean_dataset.py --all
# -> data/processed/{train,test}_source{1,2,3}.parquet
```

Output columns: `entity_id, country, raw_name, raw_addr, name_canon,
name_core, name_legal, addr_canon, street_number, state, is_domain, is_native`.

Throughput ≈ 150–210k rows/s (8 workers); full 24.2M records ≈ 2.5 min.

### 2. Candidate generation / blocking  *(next)*

Multi-key blocking (name tokens, street#+state, address tokens, name n-gram
LSH) with document-frequency pruning of non-discriminative keys. Emits
`output/candidate_pairs.tsv`.

### 3. Matching model  *(next)*

Precision-first classifier (XGBoost) over string-similarity features
(RapidFuzz Levenshtein/Jaro-Winkler, token & char-n-gram Jaccard, TF-IDF
cosine, address-token overlap, street-number exactness), threshold tuned for
**F₀.₅**. Emits `output/matching_results.tsv`.

## Data-understanding highlights (see methodology doc)

- 94.4% of S1 entities have ≥1 match (5.6% singletons); avg 3.67 matches.
- S1 is clean/romanized; ~12–15% of S2/S3 names are in native scripts.
- Test adds **France** (~15%), unseen in training — nothing is hard-coded to a
  closed country set.
- Blocking-recall baseline (name+street keys) on train GT: **94.9%**
  (US 97.9%, India 90.5%).
