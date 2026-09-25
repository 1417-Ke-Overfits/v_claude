# 00 — Challenge Overview

## The problem: Business Entity Resolution (ER)

Business identity data arrives from **three independent sources**, each a
partial, noisy view of the same real-world businesses. The sources share **no
common identifier**. We must determine which records across sources refer to the
same real business.

- **Source 1 (S1)** is the *deduplicated reference*: one row per real entity.
- **Source 2 (S2)** and **Source 3 (S3)** are noisy and may contain multiple
  records for the same entity, or none.

**Task:** for every S1 entity, output the list of S2/S3 records that match it. A
S1 entity may match **zero, one, or many** S2/S3 records.

Each record has four columns:

| Column | Notes |
|--------|-------|
| `entity_id` | Unique per record. Prefix encodes the source: `S1-`, `S2-`, `S3-`. |
| `business_name` | May contain abbreviations, legal suffixes, typos, transliterations, domain forms. |
| `business_address` | May be partial, reordered, missing components, landmark-based. |
| `country` | An **open set** of string labels. Train = {US, India}; test **adds France**. Never hard-code to a closed set. |

Files are **tab-separated (`.tsv`)** because names and addresses contain commas.
Reading without `sep="\t"` collapses everything into one column.

The ground truth (`train_ground_truth.tsv`) has two columns:
`source1_entity_id` and a comma-separated `matched_entity_ids` (empty when the
entity has no match).

## The metric: F₀.₅ (precision-heavy), macro-averaged

```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

- Computed **per S1 entity**, then **averaged over all S1 entities** (macro).
- **Singletons count.** A S1 entity with no true matches scores **1.0** if we
  correctly predict an empty list, and **0.0** if we predict any match for it.
- β = 0.5 ⇒ **precision weighted 2× over recall**. A *false merge* (linking two
  different businesses) hurts more than a *missed link*.

**Worked example** (from the brief):
- Predicted: S1-00001 → [S2-00047, S2-00193, S3-00812]
- Truth:     S1-00001 → [S2-00047, S3-00812]
- Precision = 2/3, Recall = 2/2 = 1.0 → **F₀.₅ = 0.714**

### What the metric implies for strategy

1. **Precision first.** Because a wrong merge is doubly penalised, we prefer to
   emit a match only when confident. This drives a **high decision threshold**.
2. **Singletons are free points if handled well** (5.6% of entities — see EDA).
   Predicting empty for a true singleton scores 1.0; a single false merge on it
   scores 0.0. Do not spray matches.
3. **Recall still matters** (94.4% of entities *do* have matches), so blocking
   must not throw away true pairs — but the final classifier can be conservative.
4. The macro average means **every entity is worth the same**, whether it has 1
   match or 10. Getting the many small/medium clusters right matters more than
   perfecting a few large ones.

## Outputs

Two tab-separated files in `output/`:

1. **`matching_results.tsv`** — the final matches. **This is the only file scored
   on the leaderboard.** Columns: `source1_entity_id`, `matched_entity_ids`.
2. **`candidate_pairs.tsv`** — the candidate set fed to the matching model
   (the last blocking stage, just before the model scores). Not scored, but used
   to audit blocking quality (recall ceiling, reduction ratio). Final matches
   must be a **subset** of candidates.

Formatting rules (enforced by the scorer and by `utils/validate_submission.py`):
- Exactly one row per S1 entity; empty `matched_entity_ids` = no match.
- Only `S2-`/`S3-` IDs; no self-matches; no duplicate IDs within a list; no
  duplicate S1 rows; every test S1 entity present.

## Constraints & fair play

- **No external data lookup** — no commercial ER APIs, no government registries,
  no geocoding APIs, no internet augmentation. *Strictly enforced;* violation =
  disqualification. Offline string processing and transliteration are fine
  (they are normalization, not identity lookup).
- **Final model:** MIT/Apache-licensed, **≤ 8 billion parameters**.
- Submissions that fail format validation are not scored.

## Leaderboard

- **Public** leaderboard: a subset of the test set (live feedback).
- **Private** leaderboard: the remaining test set, revealed at the end —
  **final rankings are private-leaderboard**. We predict the full test set in
  both cases; the split is applied at scoring time.

⇒ **Do not overfit the public split.** Validate with our own held-out split
using the exact F₀.₅ formula above.
