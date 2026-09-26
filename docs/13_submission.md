# 13 — Test Inference & Submission (Stage 8)

This stage runs the full pipeline on the **test** set and produces the files you
upload / submit.

**Code:** `src/predict.py` (+ the same `precompute_keys` / `generate_candidates`
used for training).

---

## 1. What gets submitted

| File | Purpose |
|------|---------|
| **`output/matching_results.tsv`** | **The leaderboard file** — the only file scored. Upload this to the Portal. |
| `output/candidate_pairs.tsv` | Blocking candidate set (for auditing); goes in the final zip. |
| `1417_ke_overfits_submission.zip` | Final package: `output/` (both TSVs) + `code/business_entity_resolution/` + filled `Documentation_template.md`. |

## 2. Pipeline run (test)

```bash
python src/precompute_keys.py    --split test
python src/generate_candidates.py --split test --topn 300 --out output/candidate_pairs.tsv
python src/predict.py --candidates output/candidate_pairs.tsv --split test \
    --model models_rel/xgb.json --out output/matching_results.tsv \
    --threshold 0.99 --cand-cap 100
python utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate /nonexistent --test-dir dataset/test    # fast format check
```

- **Blocking (test):** N=300 candidates for all 1,732,544 test S1 → `candidate_pairs.tsv` (6.2 GB).
- **Scoring:** `predict.py` computes the 35 features (relative features over the
  *full* candidate list, matching training) but only featurises/scores the
  **top-`cand-cap`** candidates per S1 — valid because the high F₀.₅ threshold
  only accepts strong, top-ranked candidates. Threshold 0.99.

## 3. Result

| Metric | Value |
|--------|------:|
| S1 rows written | 1,732,544 (every test S1) |
| Predicted singletons (empty) | 101,682 (**5.9%**) |
| Rows with ≥1 match | 1,630,862 |
| Total matches | 6,181,721 (**3.57 / S1**) |
| S2 / S3 matches | 2,864,813 / 3,316,908 |
| Duplicate S1 rows / within-list dupes / bad prefixes | 0 / 0 / 0 |
| **Validator** | **PASS — safe to submit** |

**Calibration sanity:** the 5.9% predicted-singleton rate closely matches the
training singleton rate (5.6%), and 3.57 matches/S1 matches the training average
(~3.46–3.67) — strong signs the model transfers well to test (including the
unseen France records, handled by the country-agnostic pipeline).

## 4. Expected score

Held-out validation gave macro **F₀.₅ = 0.9226** (honest) / **0.9423**
(candidate-only). The public leaderboard number will land in that range depending
on how it counts blocking misses (see doc 12).

## 5. Notes / gotchas

- `validate_submission.py` **defaults to reading `output/candidate_pairs.tsv`**
  (6.2 GB) for an optional subset cross-check — pass `--candidate /nonexistent`
  to skip it for a fast matching-only format check.
- Full test inference took ~3.4 h at `--cand-cap 100`; `--cand-cap 60` is ~3×
  faster with almost identical matches (both capture the strong, top-ranked
  candidates the threshold accepts).
- To raise the score further, see the levers in doc 12 (India-specific matching,
  true pairwise cross-source corroboration, more training data).
