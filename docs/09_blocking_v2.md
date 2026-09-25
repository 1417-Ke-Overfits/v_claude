# 09 — Blocking v2: From 94.9% (Unusable Volume) to a Tractable Candidate Set

Blocking v1 (doc 06) reached **94.9% recall** but with a fatal flaw: giant
generic-token blocks implied ~**172 billion** candidate comparisons — impossible
to featurise. Blocking v2's job was to reach comparable-or-better recall **at a
volume a classifier can actually process**. This doc records the full
experimental journey, every metric, the dead-ends, and the final design.

**Code:** `src/blocking.py` (keys), `src/precompute_keys.py` (cache),
`src/generate_candidates.py` (scalable candidate generation).

---

## 1. The two things we must balance

| Quantity | Meaning | Target |
|----------|---------|--------|
| **Recall** | fraction of true pairs kept as candidates | as high as possible (ceiling 99.3%, doc 04) |
| **Candidate volume** | total (S1, candidate) pairs to featurise | small enough to classify (≈ ≤1B) |

These fight each other: high recall needs to keep pairs that share only weak
evidence; low volume needs to drop weak evidence. The entire doc is about
finding the knee of that trade-off.

---

## 2. Experiment 1 — DF pruning alone (and why it isn't enough)

We DF-prune keys whose S2+S3 posting count exceeds a cap (they are
non-discriminative and cause O(n²) blow-ups). Measured on full train GT:

| config | recall | US | India | cand-volume (upper bound) |
|--------|-------:|---:|------:|--------------------------:|
| v1 no prune (n2,nt,sn) | 94.71% | 97.9% | 89.9% | 172.3 B |
| nt ≤ 2000 | 90.68% | 94.4% | 85.0% | 3.8 B |
| **+ address tokens ≤ 2000** | **96.80%** | 97.5% | **95.7%** | 5.4 B |
| + LSH bands ≤ 2000 | 97.17% | 97.9% | 96.1% | 6.5 B |

**Findings:**
1. **DF-pruning `nt` alone *costs* recall** (94.7→90.7%): some true matches share
   only a moderately-common name token.
2. **Address-token blocking is the recovery lever** — it not only restores recall
   but *fixes India* (85.0→95.7%). Indian native-name records usually have an
   intact, distinctive address.
3. **LSH (char-n-gram MinHash) adds only +0.4%** for heavy compute — dropped as
   not worth it.

## 3. Experiment 2 — looser thresholds trade recall for volume

Scanning name/address DF caps (no LSH):

| config | recall | US | India |
|--------|-------:|---:|------:|
| nt2000 at2000 | 96.80% | 97.5% | 95.7% |
| nt5000 at3000 | 97.37% | 98.1% | 96.2% |
| nt10000 at5000 | 97.95% | 98.7% | 96.9% |
| nt20000 at10000 | 98.59% | 99.2% | 97.7% |

Higher caps → higher recall — **but the real candidate volume is ruinous**:
`nt20000 at10000` yields **~11,300 candidates per S1** (~25 **billion** pairs).
Looser pruning buys recall by keeping large blocks. Recall alone is a trap; we
must measure *real* volume.

## 4. The volume problem is structural — composite keys help but don't solve it

A single common token is too coarse (block up to 424k). We added **composite
keys** that pair a name token with an address anchor, so a common word only
collides *within the same state / house number*:

- `ns:` = name_token + state
- `nn:` = name_token + street_number

Composite-only blocking (n2, sn, ns, nn) still produced **~5,300 candidates/S1**
(median 2,233) at 93.3% recall — because big states still make `ns` blocks huge.
Composite keys help specificity but do not, by themselves, bound volume.

## 5. Experiment 3 — conjunctive rules (require ≥K shared keys)

True matches share *many* keys; random collisions share one. So require a
candidate to share ≥K keys:

| rule | recall | US | India | cand/S1 (mean/median) | total pairs |
|------|-------:|---:|------:|----------------------:|------------:|
| K ≥ 1 | 98.42% | 99.0% | 97.5% | 9399 / 7068 | 20.7 B ❌ |
| **K ≥ 2** | 92.80% | 95.0% | 89.5% | **385 / 121** | 849 M ✅ |
| K ≥ 3 | 85.16% | 88.1% | 80.8% | 57 / 10 | 126 M |

K≥2 is tractable but loses 5.6% recall — the true matches that share exactly one
key (empty-address names, native names with one aligning anchor). K≥3 collapses
recall. Pure conjunction is too blunt.

## 6. Experiment 4–5 — top-N per S1 (rank, then cap)

Keep the top-N candidates per S1, ranked by a score. Two rankers:

**By raw shared-key count** (Exp 4):

| N | recall | total pairs |
|--:|-------:|------------:|
| 100 | 90.68% | 221 M |
| 200 | 92.07% | 441 M |
| 500 | 93.44% | 1.10 B |

**By IDF-weighted shared score** (Exp 5) — a shared *rare* key counts more:

| N | recall | US | India | total pairs |
|--:|-------:|---:|------:|------------:|
| 100 | 92.39% | 94.9% | 88.6% | 221 M |
| 200 | 93.78% | 96.0% | 90.3% | 441 M |
| 300 | ~94.4% | — | — | ~660 M |
| 500 | 95.22% | 97.1% | 92.3% | 1.10 B |

**IDF weighting beats raw count at every N** (a shared exact street#+state or rare
name token is far more indicative than a shared "services"). This is the winning
idea: rank candidates by summed IDF of shared keys, then cap.

## 7. Final design — IDF-weighted top-N via sparse matrix multiply

**Rule:** score(S1, r) = Σ over shared kept keys *k* of **IDF(k)** = log(M/df(k));
keep the **top-N = 300** candidates per S1.

**Keys:** `n2, sn, ns, nn` (composite/specific) + `nt` (df≤10k), `at` (df≤5k),
DF-pruned. No LSH.

**Why this is the right choice:**
- IDF ranking captures "how *specific* the shared evidence is", which correlates
  with true matching far better than a raw key count — so a modest N keeps most
  recall.
- It gives a **hard per-S1 bound** (≤N candidates), so total volume is exactly
  N × |S1|, predictable and tunable. N trades recall for compute continuously
  (N=300 → 94.5%; N=500 → 95.2%).
- A precision-first classifier sits downstream, so blocking is tuned for
  **recall + tractability**, not precision.

**Implementation — why sparse matmul:** a pure-Python per-S1 loop would take
hours at 2.2M entities. Instead we build two sparse matrices —
`A` (S1 × keys, IDF-weighted) and `B` (S2+S3 × keys, binary) — and compute
`A_batch @ Bᵀ` in batches. Each product yields, for a batch of S1 rows, the
summed-IDF score against every S2/S3 record; we `argpartition` the top-N per
row. This is BLAS-backed and runs the whole corpus in minutes.

### Measured production result (full train, N=300)

| Metric | Value |
|--------|------:|
| **Recall (all 7.64M true pairs)** | **94.48%** (7,216,552 / 7,638,365) |
| Candidate pairs total | 659,787,362 |
| Candidates per S1 | mean 299, median 300, max 300 |
| S1 with zero candidates | 1 |
| Runtime | ~18 min (build + 2.2M-entity generation) |

Compared with v1: **similar recall (94.5% vs 94.9%) at ~1/260th the candidate
volume** (0.66 B vs 172 B) — the whole point of v2. The candidate set is now
small enough to featurise and classify.

---

## 8. Pipeline & reproduction

```bash
# 1. cache blocking keys per source  (data/interim/keys_*.npz)
python code/business_entity_resolution/src/precompute_keys.py --split train
# 2. generate candidates + measure recall
python code/business_entity_resolution/src/generate_candidates.py \
    --split train --topn 300 \
    --out output/candidate_pairs_train.tsv \
    --gt dataset/train/train_ground_truth.tsv
```
For the submission, the same two commands with `--split test` (no `--gt`) produce
`output/candidate_pairs.tsv`.

---

## 9. Problems faced & how we solved them

| Problem | Symptom | Resolution |
|---------|---------|-----------|
| Buffered output hid progress | long runs looked hung | `flush=True` on all logs |
| Re-running key computation each config | ~10 min per experiment | **cache keysets to disk** (`precompute_keys`), then experiment in-memory in seconds |
| Config scoring loop O(all keys) | 216 s per config | precompute each true pair's *shared* keys once; a config is then a fast set filter |
| Pure-Python candidate gen too slow | ~hours projected for 2.2M S1 | **sparse matrix multiply**, batched |
| Recall accidentally ignored pruning (a bug) | one experiment reported 99.3% (unpruned) | intersect shared keys with the `kept` set before scoring |

## 10. Limits & what could push recall higher (future)

Current ceiling for this candidate set is ~94.5% at N=300 (95.2% at N=500). The
missing ~5% are:
- **Empty-address native/alias names** sharing ≤1 weak key — inherently hard.
- **India house-number noise** (`AF-684`, `6(29)`, `G-3/571`) weakens the
  street-number anchor, so `sn`/`nn` fire less reliably than for US.

Ideas (only if error analysis shows they pay off):
1. **Raise N** (or make N adaptive per S1) — direct recall/compute trade.
2. **Stronger address keys for India** — locality-name composites, PIN-code
   extraction where present, or 2-address-token combinations.
3. **Learned candidate scorer** — replace summed-IDF with a tiny model over cheap
   features (a "pre-ranker") to lift recall@N.
4. **Phonetic / n-gram blocking** for domains and heavy transliteration (LSH
   revisited with better parameters), targeted only at the residual misses.

For now, 94.5% recall at 0.66B candidates is a strong, tractable base for the
matching model (doc 10+).
