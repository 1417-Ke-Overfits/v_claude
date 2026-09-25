# 08 — Next Steps & Roadmap

Where we are, what's next, why, and the alternatives we weighed. This is the plan
of record; it is updated as steps complete.

## Status recap

- ✅ **Phase 1 — EDA** (docs 02–04): dataset fully characterised; recall ceiling
  99.3% established.
- ✅ **Phase 2 — Cleaning** (doc 05): canonical Parquet for all 24.2M records;
  validated; blocking-recall baseline **94.9%** (doc 06).
- 🔜 **Phase 3 — Blocking v2**, then **matching model**, then **F₀.₅ tuning +
  submission**.

---

## Step 1 — Blocking v2: recall 94.9% → ~99%, kill the explosion

Three changes, in priority order (rationale in doc 06 §5):

1. **DF pruning (highest priority, cheapest).** Drop `nt:` keys whose S2+S3
   posting count exceeds a threshold (candidate ~1–2k). Removes the ~3,400
   generic-token blocks (`center`=424k, `services`, `group`, …) that cause the
   n² blow-up, at ~zero recall cost (a match sharing *only* "services" is
   unusable). **Success metric:** max block < ~3,000; total candidate pairs down
   by orders of magnitude; recall unchanged.
2. **Address-token blocking.** Key on low-DF address tokens (distinctive
   street/locality names), optionally paired with the street number. **Why:**
   doc 04 shows address overlaps strongly for matches; Indian native-name records
   usually have intact addresses. **Success metric:** India recall 90.5% → ~96%+.
3. **Domain splitting + char-n-gram (MinHash) LSH.** Split domain labels into
   word-pieces; add MinHash banding over name char-trigrams for typo'd/scrambled/
   transliterated names with no shared token. **Why:** targets the residual
   domain and native-script misses. **Success metric:** overall recall → ~99%.

Deliverable: a `blocking.py` that emits, per S1 entity, a candidate list — and
writes `output/candidate_pairs.tsv`. We will **re-measure recall and block-size
after each change** and log the numbers (new doc: `09_blocking_v2.md`).

**Trade-off watched:** every added key raises recall but also candidate volume
(hurting downstream cost and, if we auto-accepted candidates, precision). Since a
classifier sits after blocking, we bias blocking toward **recall**, then let the
model deliver precision. We cap candidate volume via DF pruning and per-S1
candidate limits.

---

## Step 2 — Matching model (precision-first classifier)

**Approach:** for each (S1, candidate) pair, compute a feature vector and score
it with a **gradient-boosted tree classifier (XGBoost)**; keep pairs above an
F₀.₅-tuned threshold.

**Why a GBT, not a neural/transformer matcher (yet):**
- Interpretable, fast to train/infer on tens of millions of pairs, strong on
  tabular similarity features, trivially satisfies the ≤8B-param / Apache rule.
- Needs no GPU; fully reproducible.
- A learned text encoder (e.g. a small sentence embedder) is a *possible later
  add* as an extra feature, but the brief bans external data and rewards
  precision — a well-featured GBT is the high-ROI, low-risk first model. We
  revisit embeddings only if error analysis shows the fuzzy features plateau.

**Feature families** (justified by doc 04):
- **Name:** token Jaccard, char-trigram Jaccard, Levenshtein ratio, Jaro-Winkler,
  token-sort/token-set ratios (RapidFuzz), TF-IDF cosine (char n-gram), length
  ratio, `is_native`/`is_domain` flags, legal-suffix conflict flag.
- **Address:** token Jaccard, street-number exact match, state match, TF-IDF
  cosine, `address_present` flag (so the model down-weights address when empty).
- **Corroboration (later):** does another source's record for the same S1 agree?
  (85% of matches have both sources — doc 02.)

**Labels:** from ground truth — a candidate pair is positive iff it's in the GT
match list for that S1. Negatives are the non-matching candidates from blocking
(realistic hard negatives, not random pairs).

**Validation:** a held-out split of **S1 entities** (not pairs), so no entity
leaks between train/val. Score with the exact **F₀.₅ macro** formula.

Deliverable: `features.py`, `train_matcher.py`, `predict.py`; docs
`10_features.md`, `11_matching_model.md`.

---

## Step 3 — Decision rule & F₀.₅ threshold tuning

- Sweep the classifier threshold on the validation split to **maximise macro
  F₀.₅**, exploiting its precision-heavy shape (expect a fairly high threshold).
- Consider **per-country thresholds** (US/India/France behave differently) and a
  **top-k cap** per S1 (clusters are mostly ≤4 — doc 02).
- **Singleton handling:** emit an empty list when no candidate clears the
  threshold — this is where the 5.6% singletons earn their 1.0.
- Enforce all format rules and run `utils/validate_submission.py` before every
  submission.

Deliverable: `output/matching_results.tsv` + `output/candidate_pairs.tsv`;
doc `12_thresholding_and_results.md` with the final validation F₀.₅ and error
analysis.

---

## Step 4 — Error analysis loop & iteration

- Break false positives / false negatives down by country, cluster size, native
  vs. Latin, address-present vs. empty.
- Feed findings back into blocking keys, features, and thresholds.
- Track every experiment's validation F₀.₅ in a results table
  (`13_experiments.md`).

---

## Longer-horizon ideas (only if error analysis justifies)

- **Transitive/graph resolution:** if S2-a↔S1 and S3-b↔S2-a strongly agree,
  corroborate S3-b↔S1. Must be done carefully — false transitivity hurts
  precision under F₀.₅.
- **Small offline text embedder** as an extra feature for semantic name
  similarity (must be MIT/Apache, ≤8B, no external data).
- **Better transliteration** (schwa-deletion, per-script tuning) to close the
  `epeksa`≠`apex` gap.
- **Learned blocking** (e.g. supervised LSH thresholds) if hand-tuned keys
  plateau.

---

## Guardrails we will not cross

- No external data lookup (APIs, registries, geocoders, internet augmentation).
- Final model MIT/Apache, ≤8B params.
- Country-agnostic — no hard-coding to {US, India}; France and any label handled.
- Don't overfit the public leaderboard; trust the held-out F₀.₅.
