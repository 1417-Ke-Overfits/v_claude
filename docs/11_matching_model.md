# 11 — Matching Model & F₀.₅ Tuning (Stage 5)

The matching model scores every candidate pair and decides which are true
matches. Because the metric is **precision-heavy F₀.₅ macro-averaged per S1
entity**, the model is evaluated *exactly* that way — not by pair-level AUC — and
the decision threshold is tuned on a realistic-base-rate validation set.

**Code:** `src/train_model.py`. **Models saved:** `models/xgb.json`,
`models/lgbm.txt`, `models/bakeoff.json`.

---

## 1. Honest evaluation: end-to-end macro F₀.₅

The validation set (`val_feats`, 60k held-out S1 entities, ~1.09% positive) is
scored the way the leaderboard scores:

- Per S1 entity: predicted = candidates with `score ≥ threshold`.
- **Recall is against the FULL ground-truth set** — true matches that *blocking
  missed* (not in the candidate set) are counted as forced false negatives. So
  the reported F₀.₅ already includes the ~5.5% blocking loss; it is the real
  end-to-end number, not an optimistic pair-level score.
- Singletons: F = 1.0 if we predict nothing, 0.0 if we predict anything.
- macro = mean F₀.₅ over all 60k entities.

The threshold is swept on a grid sampled finely at the high end (precision-heavy
F₀.₅ favours high thresholds), globally and **per country**.

## 2. Model bake-off (held-out macro F₀.₅)

| Model | macro F₀.₅ | Precision | Recall | best threshold |
|-------|-----------:|----------:|-------:|---------------:|
| **XGBoost** | **0.9206** | 0.959 | 0.849 | 0.993 |
| LightGBM | 0.8944 | 0.931 | 0.849 | 0.990 |
| Logistic Regression | 0.8298 | 0.890 | 0.753 | 0.996 |

**XGBoost wins clearly.** The logistic baseline confirms the gradient-boosted
trees add real value (+0.09 F₀.₅) — the signal is non-linear (e.g. "low name
similarity *but* exact street#+state ⇒ match"), which trees capture and a linear
model cannot.

Per country (XGBoost): **US = 0.9434**, **India = 0.8867**. India remains the
harder country (native-script names, noisier house numbers) — consistent with
every earlier stage. Per-country thresholds (US 0.993 / India 0.991) were almost
identical to the global one, so a single global threshold ≈ **0.99** is used.

### Reading the operating point
At threshold 0.993, **precision = 0.959, recall = 0.849**. This is the F₀.₅
sweet spot: the metric weights precision 2×, so the model deliberately trades a
little recall for high precision. The recall is also capped by blocking (94.5%);
the classifier keeps ~0.849/0.945 ≈ 90% of the true matches it *could* see.

## 3. Why XGBoost (and the training setup)

- **Tabular, separable, interacting features** (doc 10) → gradient-boosted trees
  are the right tool; they dominate this regime.
- Config: `max_depth=8, eta=0.2, 300 rounds, subsample/colsample=0.8,
  scale_pos_weight = neg/pos` (handles the training imbalance). Trains in ~45 s
  on 10M rows, CPU-only.
- **≤8B params, Apache-2.0** — constraints satisfied trivially.
- The absolute probabilities need not be calibrated: we pick the threshold on the
  realistic-base-rate validation set, so only the *ranking* matters.

## 4. Feature importance (XGBoost gain)

Top features: `addr_present_both`, `addr_present_b`, `addr_token_set`,
`addr_token_jaccard`, `state_both`, `name_partial_ratio`, `name_jaro_winkler`,
`street_num_exact`, `b_is_native`, `addr_ratio`.

The model's structure matches the domain: it **gates on whether an address is
present**, then uses **address similarity** (token-set / Jaccard / exact street
number / state) as the primary evidence, falling back to **name fuzzy scores**
when the address is weak or absent. This is exactly the strategy the EDA implied
(address is the strongest anchor when present; name carries the empty-address
cases).

## 5. Problems faced & fixes

| Problem | Fix |
|---------|-----|
| Optimal threshold hit the sweep ceiling (0.99) | extend grid to 0.999, sampled finely at the top |
| Naive AUC would mislead | evaluate true **macro F₀.₅ per entity**, with blocking misses as forced FNs |
| Training base rate ≠ test base rate (negatives subsampled) | tune threshold on the full-base-rate validation set (ranking is what matters) |
| LightGBM/XGBoost need OpenMP | `brew install libomp` |

## 6. Where the score can still go up (Stage 7 iteration)

Current **0.9206**. Highest-leverage next steps, in order:
1. **Cross-source corroboration feature** — 85% of matches appear in both S2 & S3;
   an agreement signal should lift precision *and* recall. (Not yet a feature.)
2. **Raise blocking N** (300→500) — lifts the recall ceiling 94.5%→95.2%, so the
   F₀.₅ recall term can rise.
3. **More training data / negatives** — better precision calibration near the
   threshold, where the score is decided.
4. **TF-IDF cosine & transliteration-aware name features** — help the residual
   India / native-script cases.
5. **XGBoost tuning + XGB/LGBM ensemble** — modest, low-risk gains.
6. **Per-country / France threshold** — France is unseen in training; pick a
   conservative (high) threshold for it to protect precision.

## 7. Reproduce

```bash
python code/business_entity_resolution/src/train_model.py \
    --train data/interim/train_feats.parquet \
    --val   data/interim/val_feats.parquet \
    --gt    dataset/train/train_ground_truth.tsv --outdir models
```
Outputs the bake-off table, the chosen threshold, feature importances, and the
saved models.
