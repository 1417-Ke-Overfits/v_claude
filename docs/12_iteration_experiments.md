# 12 — Iteration Experiments (Stage 7)

After the first end-to-end model (honest macro F₀.₅ = 0.9206, doc 11), we ran a
set of targeted experiments to find where the score is actually bottlenecked and
how to push toward a top-of-leaderboard number. This doc records each experiment,
its result, and the conclusion — including the negative results, which were the
most informative.

---

## Experiment A — Blocking N-sweep (does more recall help?)

**Hypothesis:** our N=300 candidate cap is throttling recall; raising it will lift
F₀.₅.

**Method:** regenerate candidates at N=1000 (score-sorted), re-featurize, retrain,
re-evaluate the same honest macro F₀.₅.

| Setting | Blocking recall | F₀.₅ | Precision | Recall |
|---------|----------------:|-----:|----------:|-------:|
| N=300 | 94.5% | 0.9206 | 0.959 | 0.849 |
| N=1000 | 96.1% | **0.9201** | 0.958 | 0.854 |

**Result: no improvement.** Blocking recall rose +1.6 points, but F₀.₅ was flat.

**Conclusion — the candidate cap is NOT the bottleneck.** The extra candidates
(ranks 301–1000 by IDF) are *weak* matches the classifier will not confidently
accept at the precision-heavy operating threshold, so they never convert to
score. **The bottleneck is matching *confidence* on hard cases, not candidate
recall.** This redirected all further effort to features/model.

## Experiment B — Relative & cross-source features

**Hypothesis:** competitive context (how a candidate ranks *among its S1's other
candidates*) and cross-source agreement will sharpen precision on borderline
pairs.

**Added 6 features** (computed per S1 over all its candidates, from cheap core
token sets so it stays fast):
`namejac_rank`, `is_top_namejac`, `namejac_margin`, `other_src_best`,
`cross_twin` (shares a name signature with an other-source candidate), `n_cands_log`.

**Feature quality** (val, pos vs neg means): `is_top_namejac` 0.248 vs 0.001,
`namejac_margin` −0.248 vs −0.710, `cross_twin` 0.652 vs 0.446 — all discriminate,
and `namejac_rank` appears in the model's top features.

| Model | F₀.₅ (honest) | vs baseline |
|-------|--------------:|------------:|
| XGBoost + relative feats | **0.9229** | +0.0023 |

**Result: a small but real gain** (0.9206 → 0.9229) — and achieved with *half*
the training data (150k vs 300k S1). Modest; not a breakthrough on its own.

## Experiment C — Evaluation methodology (the big reframe)

**Question:** a teammate reported ~0.95 "without data engineering". Is that a
better model, or a different measurement?

We compute F₀.₅ two ways on the same predictions:
- **Honest** — recall denominator = the *full* ground-truth match set per S1, so
  blocking-missed matches count as false negatives (what docs 11–12 report).
- **Lenient** — recall denominator = only the true matches that *survived
  blocking*, i.e. blocking misses are not penalised (what many pipelines report).

| Eval mode | XGBoost F₀.₅ |
|-----------|-------------:|
| Honest (blocking misses = FN) | 0.9226 |
| **Lenient (ignore blocking misses)** | **0.9423** |

**The "blocking tax" is ~0.02.** Our lenient number is **0.9423** — much closer to
0.95. So a large part of the apparent 0.92-vs-0.95 gap is almost certainly a
**measurement difference**: whether blocking-missed matches are counted. The
action item is to **confirm the teammate's methodology** (and whether France is
included) before treating the gap as a real model deficit.

## Experiment D — true cross-source corroboration + more training data

**Hypothesis:** the residual recall gap (India native-script / weak-address
cases) can be closed by *corroboration*: a candidate that barely matches S1
directly should be accepted if it agrees with a confident cross-source match of
the same S1.

**Added 3 features** (computed per S1 in O(k) via a "best other-source anchor"):
`xsrc_name_agree`, `xsrc_addr_agree`, `xsrc_anchor_s1sim` — the candidate's
name/address agreement with the other source's most-S1-similar candidate.
Also retrained on **250k** S1 (vs 150k in Exp B).

**Feature quality** (val): `xsrc_name_agree` 0.684 vs 0.256, `xsrc_addr_agree`
0.466 vs 0.092 — strong new discriminators.

| Version | F₀.₅ (honest) | Precision | Recall | India |
|---------|--------------:|----------:|-------:|------:|
| Base (29 feats) | 0.9206 | 0.959 | 0.849 | — |
| + relative (Exp B) | 0.9229 | 0.961 | 0.850 | 0.890 |
| **+ cross-source + 250k (Exp D)** | **0.9250** | 0.958 | **0.868** | 0.892 |

**Result:** honest F₀.₅ 0.9229 → **0.9250**; lenient 0.9423 → **0.9444**. The
important shift is **recall 0.849 → 0.868 at held precision** — corroboration
pushes borderline true matches over the threshold, as intended. Gains are
monotonic but incremental; India remains the hardest country.

## Experiment E — XGBoost hyperparameter tuning

On the Exp D feature set (no rebuild — model-only), swept XGBoost configs:

| Config | F₀.₅ | Precision | Recall | India |
|--------|-----:|----------:|-------:|------:|
| d8 / 300 (previous) | 0.9250 | 0.958 | 0.868 | 0.892 |
| **d10 / 600 / min_child_weight=5 / eta 0.1** | **0.9276** | 0.962 | 0.866 | 0.897 |
| d12 / 500 / mcw10 | 0.9261 | 0.959 | 0.871 | 0.895 |

Deeper + slower + mildly regularised is best; d12 starts to over-fit. Adopted
**d10/600/mcw5** as the production config. Lenient F₀.₅ = **0.9474**.

## Cumulative progress (honest macro F₀.₅)

| Stage | F₀.₅ | Lenient | Recall |
|-------|-----:|--------:|-------:|
| Base model (29 feats) | 0.9206 | 0.9423 | 0.849 |
| + relative feats | 0.9229 | — | 0.850 |
| + cross-source + 250k | 0.9250 | 0.9444 | 0.868 |
| **+ tuned XGBoost** | **0.9276** | **0.9474** | 0.866 |

Net: **+0.0070 honest / +0.0051 lenient**, recall 0.849 → 0.866 at precision
0.96. The tuned + cross-source model is used to regenerate the submission.

---

## Where we stand after Stage 7

- **Honest** macro F₀.₅ = **0.9229** (US 0.945, India 0.890).
- **Lenient** macro F₀.₅ = **0.9423**.
- Precision is already high (0.96); the binding constraint is **recall on hard
  matches**, dominated by India (native-script names, noisy house numbers).

## What actually remains to reach a true 0.95 (honest)

1. **Align the metric** with the leaderboard / teammate — resolve whether blocking
   misses and France are counted. This alone may close most of the gap.
2. **Matching quality on India / native-script** — the real model bottleneck:
   TF-IDF char-n-gram cosine, a from-scratch Siamese/char-CNN name encoder,
   stronger (true pairwise) cross-source corroboration.
3. **More training data** — retrain the new-feature model on 300k+ S1 (Exp B used
   only 150k); cheap to test, likely a fraction of a point.
4. **Reduce the blocking tax at the source** — not by adding weak candidates
   (Exp A showed that fails) but by *better candidate quality* for India
   (locality-name keys, PIN codes) so more true matches arrive as *strong*
   candidates the model will accept.
5. **Ensemble + calibration + per-country/France thresholds** — low-risk polish.

**Key lesson from this stage:** the score is limited by *confident identification
of hard matches*, not by recall ceiling or training-data volume. Effort should go
to India-specific matching signal and to nailing down the evaluation, not to
bigger candidate sets.
