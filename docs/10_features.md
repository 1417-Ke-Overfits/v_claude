# 10 — Feature Engineering (Stage 4)

For every candidate pair `(S1, S2/S3)` from blocking, we compute a numeric
feature vector that the matching classifier will score. Features are grounded in
the signal analysis (doc 04): character n-grams beat tokens on typos, address is
the strongest anchor *when present*, and transliteration is phonetic (so fuzzy >
exact for native names).

**Code:** `src/features.py` (the features) and `src/build_training.py` (join +
label + featurize driver).

---

## 1. The 29 features

| Group | Feature | What it captures |
|-------|---------|------------------|
| **Name** | `name_token_jaccard` | overlap of core token sets |
| | `name_trigram_jaccard` | overlap of name char-trigrams (typo-robust) |
| | `name_ratio` | RapidFuzz Levenshtein ratio |
| | `name_jaro_winkler` | Jaro-Winkler (prefix-weighted) |
| | `name_token_sort` | token-sort ratio (order-invariant) |
| | `name_token_set` | token-set ratio (extra-word-robust) |
| | `name_partial_ratio` | best substring alignment |
| | `name_len_ratio` | length ratio of canonical names |
| | `name_prefix4` | first 4 chars equal |
| | `name_exact_canon` | canonical names identical |
| | `name_core_len_a/b` | token counts (context) |
| **Address** | `addr_token_jaccard` | overlap of address token sets |
| | `addr_ratio` | Levenshtein ratio on canonical address |
| | `addr_token_set` | token-set ratio |
| | `street_num_exact` | same house/building number |
| | `street_num_both` | both have a street number |
| | `state_match` | same canonical state/region |
| | `state_both` | both have a state |
| | `addr_len_ratio` | address length ratio |
| | `addr_present_a/b/both` | address present flags |
| **Context** | `same_country` | same country label |
| | `b_is_native` | candidate name in native script |
| | `b_is_domain` | candidate name is a web domain |
| | `legal_overlap` | shared legal-suffix tokens |
| | `legal_conflict` | disjoint legal suffixes (weak negative) |
| | `src_is_s3` | candidate is from Source 3 |

Design choices:
- **Both fuzzy scores and set Jaccards** are kept — they are complementary (doc
  04: 52% of matches share an identical token set, but char-level scores rescue
  the typo'd/scrambled 14.7% that share no token).
- **Address-present flags** let the model *down-weight* the address when it is
  empty (3.3% of records) and lean on the name instead.
- **`is_native` / `is_domain`** flag the two structurally hard cases so the model
  can learn different score calibrations for them.
- Only offline string computation — no external lookup (fair-play compliant).

## 2. How the labeled dataset is built

`build_training.py`:
1. Loads the cleaned S1/S2/S3 fields (Parquet) and builds `id → row` maps.
2. Reads ground truth → the positive match set per S1.
3. **Entity-level train/val split** by a hash of the S1 id (~20% validation) —
   so no S1 entity appears in both splits (prevents leakage).
4. For each S1's candidate list:
   - **Training S1:** keep *all* positive candidates + a sample of ≤30 negatives
     (candidates are ~1% positive; subsampling negatives keeps the training set
     small and balanced enough to learn from — the true base rate is restored at
     threshold-tuning time).
   - **Validation S1:** keep **all** candidates, so the F₀.₅ threshold is tuned
     on the *realistic* ~1% base rate the test set will have.
5. Featurizes each pair and writes Parquet (`label, s1, cand, <29 features>`).

**Performance:** the naive per-S1 `arrow.take` was ~2k pairs/s (hours at scale).
Restructured to **batch many S1 per `take` and dedup record prep within a
chunk** → ~25k pairs/s. Full build: **~18 min**.

### Produced datasets

| Dataset | Rows | Positives | Note |
|---------|-----:|----------:|------|
| `train_feats.parquet` | 9,978,780 | 981,426 (9.84%) | 300k S1, ≤30 neg/S1 |
| `val_feats.parquet` | 17,940,414 | 195,331 (1.09%) | 60k S1, all candidates |

## 3. Feature quality check (validation set, realistic base rate)

Mean feature value for true matches vs non-matches — every feature separates the
classes cleanly, and the pattern matches the EDA:

| Feature | positive mean | negative mean |
|---------|--------------:|--------------:|
| `addr_token_jaccard` | **0.745** | 0.141 |
| `street_num_exact` | 0.736 | 0.220 |
| `name_jaro_winkler` | 0.934 | 0.667 |
| `name_token_set` | 0.926 | 0.608 |
| `name_ratio` | 0.885 | 0.525 |
| `addr_ratio` | 0.875 | 0.479 |
| `name_token_jaccard` | 0.729 | 0.264 |
| `state_match` | 0.953 | 0.652 |

**Address token overlap is the single strongest discriminator** (0.745 vs 0.141)
— exactly as the signal analysis predicted (address is the best anchor when
present). Name fuzzy scores are strong across the board. The classifier has
clean, well-separated signal to work with.

## 4. Problems faced & fixes

| Problem | Fix |
|---------|-----|
| Per-S1 `arrow.take` too slow (~2k pairs/s) | batch many S1 per take; dedup record prep per chunk (~25k pairs/s) |
| Candidate record occasionally missing | filter `emit` to existing rows before featurizing (keeps pair/record alignment) |
| ~1% positive base rate (extreme imbalance) | subsample negatives for *training*; keep full base rate for *validation* |
| Leakage risk | split train/val by S1-id hash, not by pair |

## 5. Limits & possible additions (later)

- **Cross-source corroboration** (does the *other* source's record for this S1
  agree?) is not yet a feature — 85% of matches have both S2 & S3, so this is a
  promising precision signal to add in the modelling/error-analysis loop.
- **Transliteration-aware name score** (compare romanized forms explicitly) could
  help the native-script cases beyond the current fuzzy scores.
- **TF-IDF cosine** (corpus-level rarity weighting of name/address n-grams) is a
  natural add if error analysis shows the current fuzzy scores plateau.

Next: train the precision-first classifier on `train_feats` and tune the F₀.₅
threshold on `val_feats` (docs 11–12).
